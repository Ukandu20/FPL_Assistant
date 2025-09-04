#!/usr/bin/env python3
r"""
goals_assists_forecast.py — leak-free scorer for future GWs using trained G/A heads,
now with optional roster gating from master_teams.json.

Inputs
------
• Trained artifacts from goals_assists_model_builder.py
• Minutes forecast CSV from minutes_forecast.py
• Registry features (players_form, team_form)
• Optional roster file (master_teams.json) to restrict scoring to players on the future-season roster

Output
------
<out-dir>/<SEASON>/GW<from>_<to>.csv with columns:
  season, gw_orig, date_sched, player_id, team_id, player, pos,
  pred_minutes, team_att_z_venue, opp_def_z_venue,
  pred_goals_p90_mean, pred_assists_p90_mean,
  pred_goals_mean, pred_assists_mean,
  pred_goals_p90_poisson, pred_assists_p90_poisson,
  pred_goals_poisson,  pred_assists_poisson,
  p_goal, p_assist, p_return_any
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

# ----------------------------- helpers ----------------------------------------

def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def _pick_gw_col(cols: List[str]) -> Optional[str]:
    for k in ("gw_played","gw_orig","gw"):
        if k in cols: return k
    return None

def _coerce_ts(s: pd.Series, tz: Optional[str]) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if tz:
        if out.dt.tz is None:
            out = out.dt.tz_localize(tz)
        else:
            out = out.dt.tz_convert(tz)
    return out

def _load_players_form(features_root: Path, form_version: str, seasons: List[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        fp = features_root / form_version / s / "players_form.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Missing players_form: {fp}")
        t = pd.read_csv(fp, parse_dates=["date_played"])
        t["season"] = s
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    need = {"season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes"}
    miss = need - set(df.columns)
    if miss: raise KeyError(f"players_form missing: {miss}")
    return df

def _load_team_form(features_root: Path, form_version: str, seasons: List[str]) -> Optional[pd.DataFrame]:
    frames = []
    for s in seasons:
        fp = features_root / form_version / s / "team_form.csv"
        if fp.exists():
            t = pd.read_csv(fp, parse_dates=["date_played"])
            t["season"] = s
            frames.append(t)
    if not frames: return None
    return pd.concat(frames, ignore_index=True)

def _team_z_venue(team_form: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if team_form is None: return None
    if {"team_att_z_venue","opp_def_z_venue"}.issubset(team_form.columns):
        t = team_form[["season","gw_orig","team_id","team_att_z_venue","opp_def_z_venue"]].drop_duplicates()
        for c in ["team_att_z_venue","opp_def_z_venue"]:
            t[c] = pd.to_numeric(t[c], errors="coerce")
        return t
    need = {"season","gw_orig","team_id","venue",
            "att_xg_home_roll_z","att_xg_away_roll_z","def_xga_home_roll_z","def_xga_away_roll_z"}
    if need.issubset(team_form.columns):
        t = team_form[list(need)].copy()
        t["team_att_z_venue"] = np.where(t["venue"].str.lower().eq("home"),
                                         t["att_xg_home_roll_z"], t["att_xg_away_roll_z"])
        t["opp_def_z_venue"] = np.where(t["venue"].str.lower().eq("home"),
                                         t["def_xga_away_roll_z"], t["def_xga_home_roll_z"])
        t = t.drop(columns=["venue","att_xg_home_roll_z","att_xg_away_roll_z","def_xga_home_roll_z","def_xga_away_roll_z"])
        t = t.drop_duplicates(subset=["season","gw_orig","team_id"])
        for c in ["team_att_z_venue","opp_def_z_venue"]:
            t[c] = pd.to_numeric(t[c], errors="coerce")
        return t
    return None

def _load_team_fixtures(fix_root: Path, season: str, filename: str) -> pd.DataFrame:
    path = fix_root / season / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing team fixtures: {path}")
    tf = pd.read_csv(path)
    for dc in ("date_sched","date_played"):
        if dc in tf.columns: tf[dc] = pd.to_datetime(tf[dc], errors="coerce")
    if "is_home" not in tf.columns:
        if "was_home" in tf.columns:
            tf["is_home"] = tf["was_home"].astype(str).str.lower().isin(["1","true","yes"]).astype(int)
        elif "venue" in tf.columns:
            tf["is_home"] = tf["venue"].astype(str).str.lower().eq("home").astype(int)
        else:
            tf["is_home"] = 0
    for c in ("gw_played","gw_orig","gw"):
        if c in tf.columns: tf[c] = pd.to_numeric(tf[c], errors="coerce")
    if "team_id" not in tf.columns:
        for alt in ["team","teamId","team_code"]:
            if alt in tf.columns:
                tf = tf.rename(columns={alt: "team_id"})
                break
    tf["season"] = season
    return tf

def _gw_for_selection(df: pd.DataFrame) -> pd.Series:
    def num(s): return pd.to_numeric(df.get(s), errors="coerce")
    gwp = num("gw_played"); gwo = num("gw_orig"); gwa = num("gw")
    return gwo.where(gwp.isna() | (gwp <= 0), gwp).where(lambda x: x.notna(), gwa)

def _ewm_shots_per_pos(hist: pd.DataFrame,
                       hl_map: Dict[str, float],
                       min_periods: int,
                       adjust: bool) -> pd.DataFrame:
    """Compute past-only EWM shots/SOT p90, with venue splits, per pos."""
    df = hist.copy()
    shots = next((c for c in ("shots","sh") if c in df.columns), None)
    sot   = next((c for c in ("sot","shots_on_target") if c in df.columns), None)
    if shots is None and sot is None:
        for col in ["shots_p90_ewm","sot_p90_ewm","shots_p90_home_ewm","shots_p90_away_ewm",
                    "sot_p90_home_ewm","sot_p90_away_ewm"]:
            df[col] = np.nan
        return df
    df["pos"] = df["pos"].astype(str).str.upper()
    m = df["minutes"].fillna(0).clip(lower=0)
    denom = (m / 90.0).replace(0, np.nan)
    if shots is not None: df["_shots_p90_raw"] = df[shots] / denom
    if sot   is not None: df["_sot_p90_raw"]   = df[sot]   / denom

    def _ewm_series(s: pd.Series, pos_tag: str) -> pd.Series:
        hl = float(hl_map.get(pos_tag, list(hl_map.values())[0]))
        return s.shift(1).ewm(halflife=hl, min_periods=min_periods, adjust=adjust).mean()

    if "_shots_p90_raw" in df.columns:
        df["shots_p90_ewm"] = (
            df.groupby(["pos","player_id","season"], sort=False)["_shots_p90_raw"]
              .transform(lambda s: _ewm_series(s, s.index.get_level_values(0)[0] if isinstance(s.index, pd.MultiIndex) else "MID"))
        )
    if "_sot_p90_raw" in df.columns:
        df["sot_p90_ewm"] = (
            df.groupby(["pos","player_id","season"], sort=False)["_sot_p90_raw"]
              .transform(lambda s: _ewm_series(s, s.index.get_level_values(0)[0] if isinstance(s.index, pd.MultiIndex) else "MID"))
        )

    mask_h = df["venue"].astype(str).str.lower().eq("home")
    mask_a = df["venue"].astype(str).str.lower().eq("away")
    for base_raw, base in (("_shots_p90_raw","shots_p90"), ("_sot_p90_raw","sot_p90")):
        if base_raw not in df.columns: continue
        df[f"{base}_home_ewm"] = np.nan
        df.loc[mask_h, f"{base}_home_ewm"] = (
            df.loc[mask_h].groupby(["pos","player_id","season"], sort=False)[base_raw]
              .transform(lambda s: _ewm_series(s, s.index.get_level_values(0)[0] if isinstance(s.index, pd.MultiIndex) else "MID"))
        )
        df[f"{base}_away_ewm"] = np.nan
        df.loc[mask_a, f"{base}_away_ewm"] = (
            df.loc[mask_a].groupby(["pos","player_id","season"], sort=False)[base_raw]
              .transform(lambda s: _ewm_series(s, s.index.get_level_values(0)[0] if isinstance(s.index, pd.MultiIndex) else "MID"))
        )
    drop = [c for c in ["_shots_p90_raw","_sot_p90_raw"] if c in df.columns]
    return df.drop(columns=drop)

def _last_snapshot_per_player(df: pd.DataFrame, feature_cols: List[str], as_of_ts: pd.Timestamp, tz: Optional[str]) -> pd.DataFrame:
    """Take last known row per (season, player_id) strictly before as_of_ts and keep requested feature_cols."""
    du = pd.to_datetime(df["date_played"], errors="coerce")
    if tz:
        if du.dt.tz is None: du = du.dt.tz_localize(tz)
        else: du = du.dt.tz_convert(tz)
    hist = df[du < as_of_ts].copy()
    if hist.empty:
        return pd.DataFrame(columns=["season","player_id"] + feature_cols)
    gw_key = _pick_gw_col(hist.columns.tolist())
    sort_cols = ["player_id","season","date_played"] + ([gw_key] if gw_key else [])
    hist = hist.sort_values(sort_cols)
    last = hist.groupby(["season","player_id"], as_index=False).tail(1).copy()

    keep = ["season","player_id"] + [c for c in feature_cols if c in last.columns]
    for c in feature_cols:
        if c not in keep:
            last[c] = np.nan
    return last[["season","player_id"] + feature_cols].copy()

def _load_booster(p: Path) -> Optional[lgb.Booster]:
    if not p.exists(): return None
    return lgb.Booster(model_file=str(p))

def _predict_reg(booster: Optional[lgb.Booster], X: pd.DataFrame) -> np.ndarray:
    if booster is None or X.empty:
        return np.zeros(len(X), dtype=float)
    Xn = X.select_dtypes(include=[np.number]).fillna(0)
    return np.clip(booster.predict(Xn), 0, None)

def _predict_per_pos(goals_or_assists: str,
                     X: pd.DataFrame,
                     pos: pd.Series,
                     model_dir: Path) -> np.ndarray:
    glob = _load_booster(model_dir / f"{goals_or_assists}_global_lgbm.txt")
    out = np.zeros(len(X), dtype=float)
    used = np.zeros(len(X), dtype=bool)
    for tag in ["GK","DEF","MID","FWD"]:
        m = _load_booster(model_dir / f"{goals_or_assists}_{tag}_lgbm.txt")
        idx = pos.str.upper().eq(tag).to_numpy()
        if idx.any() and m is not None:
            out[idx] = _predict_reg(m, X.iloc[idx])
            used[idx] = True
    if (~used).any() and glob is not None:
        out[~used] = _predict_reg(glob, X.iloc[~used])
    return out

def _predict_poisson_per_pos(name: str,
                             X: pd.DataFrame,
                             pos: pd.Series,
                             model_dir: Path,
                             med: Optional[pd.Series]) -> np.ndarray:
    """Joblib Tweedie regressors trained on med-imputed X."""
    def _load(path: Path):
        try:
            return joblib.load(path) if path.exists() else None
        except Exception:
            return None
    glob = _load(model_dir / f"{name}_global_poisson.joblib")
    out = np.full(len(X), np.nan, dtype=float)
    used = np.zeros(len(X), dtype=bool)
    Xp = X.apply(pd.to_numeric, errors="coerce")
    if med is not None:
        Xp = Xp.fillna(med)
    Xp = Xp.fillna(0.0)
    for tag in ["GK","DEF","MID","FWD"]:
        mdl = _load(model_dir / f"{name}_{tag}_poisson.joblib")
        idx = pos.str.upper().eq(tag).to_numpy()
        if mdl is not None and idx.any():
            out[idx] = np.clip(mdl.predict(Xp.iloc[idx].to_numpy(dtype=float)), 0, None)
            used[idx] = True
    if (~used).any() and glob is not None:
        out[~used] = np.clip(glob.predict(Xp.iloc[~used].to_numpy(dtype=float)), 0, None)
    return out

# ----------------------------- roster gating ----------------------------------

def _load_roster_pairs(teams_json: Optional[Path],
                       season: str,
                       league_filter: Optional[str]) -> Optional[Set[Tuple[str, str]]]:
    """
    Returns a set of allowed (team_id, player_id) pairs for the given season.
    If teams_json is None / missing / invalid, returns None (no gating).
    """
    if not teams_json:
        return None
    p = Path(teams_json)
    if not p.exists():
        logging.warning("teams_json not found at %s — skipping roster gate.", p)
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning("Failed to parse teams_json (%s): %s — skipping roster gate.", p, e)
        return None

    allowed: Set[Tuple[str, str]] = set()
    for team_id, obj in (data or {}).items():
        career = (obj or {}).get("career", {})
        season_info = career.get(season)
        if not season_info:
            continue
        if league_filter:
            if str(season_info.get("league", "")).strip().lower() != str(league_filter).strip().lower():
                continue
        players = season_info.get("players", []) or []
        for pl in players:
            pid = str(pl.get("id", "")).strip()
            if pid:
                allowed.add((str(team_id), pid))
    if not allowed:
        logging.warning("Roster map for %s produced 0 allowed pairs (league=%r).", season, league_filter)
    return allowed or None

def _apply_roster_gate(df: pd.DataFrame,
                       allowed_pairs: Optional[Set[Tuple[str, str]]],
                       season: str,
                       where: str,
                       out_artifacts_dir: Optional[Path] = None,
                       require_on_roster: bool = False) -> pd.DataFrame:
    """
    Keep only rows with (team_id, player_id) inside allowed_pairs for the season.
    If require_on_roster is True and any rows are dropped, raise RuntimeError.
    """
    if allowed_pairs is None or df.empty:
        return df
    tid = df.get("team_id").astype(str)
    pid = df.get("player_id").astype(str)
    mask_ok = tid.combine(pid, lambda a, b: (a, b) in allowed_pairs)

    dropped = int((~mask_ok).sum())
    if dropped:
        logging.info("Roster gate dropped %d %s row(s) not present on the %s roster.", dropped, where, season)
        if out_artifacts_dir is not None:
            out_artifacts_dir.mkdir(parents=True, exist_ok=True)
            df.loc[~mask_ok].to_csv(out_artifacts_dir / f"roster_dropped_{where}.csv", index=False)
        if require_on_roster:
            raise RuntimeError(f"--require-on-roster set: {dropped} {where} rows are not on the {season} roster.")
    return df.loc[mask_ok].copy()

# ----------------------------- main -------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # Windows
    ap.add_argument("--history-seasons", required=True, help="Comma list of past seasons")
    ap.add_argument("--future-season", required=True, help="Season to score (contains future GWs)")
    ap.add_argument("--as-of", default="now", help='ISO timestamp or "now"')
    ap.add_argument("--as-of-tz", default="Africa/Lagos")
    ap.add_argument("--as-of-gw", type=int, required=True, help="First GW not yet played at --as-of (next GW)")
    ap.add_argument("--n-future", type=int, default=3)
    ap.add_argument("--gw-from", type=int, default=None)
    ap.add_argument("--gw-to", type=int, default=None)
    ap.add_argument("--strict-n-future", action="store_true")

    # IO
    ap.add_argument("--features-root", type=Path, default=Path("data/processed/registry/features"))
    ap.add_argument("--form-version", required=True)
    ap.add_argument("--fix-root", type=Path, default=Path("data/processed/registry/fixtures"))
    ap.add_argument("--team-fixtures-filename", default="fixture_calendar.csv")
    ap.add_argument("--minutes-csv", type=Path, required=True, help="Output CSV from minutes_forecast.py")
    ap.add_argument("--model-dir", type=Path, required=True, help="Folder with trained G/A artifacts (a specific version)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/predictions/goals_assists"))
    ap.add_argument("--apply-calibration", action="store_true")
    ap.add_argument("--skip-gk", action="store_true")
    ap.add_argument("--log-level", default="INFO")

    # Roster gating
    ap.add_argument("--teams-json", type=Path, help="Path to master_teams.json containing per-season rosters")
    ap.add_argument("--league-filter", type=str, default="", help="Optional league name (e.g., 'ENG-Premier League')")
    ap.add_argument("--require-on-roster", action="store_true",
                    help="If set, error out when any scoring rows are not on the future-season roster")

    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    # Seasons & GW window
    history = [s.strip() for s in args.history_seasons.split(",") if s.strip()]
    seasons_all = history + [args.future_season]
    gw_from_req = args.gw_from if args.gw_from is not None else args.as_of_gw
    gw_to_req = args.gw_to if args.gw_to is not None else (gw_from_req + max(1, args.n_future) - 1)

    # Pre-compute season dirs (for artifacts)
    season_dir = args.out_dir / f"{args.future_season}"
    artifacts_dir = season_dir / "artifacts"

    # as-of
    as_of_ts = (pd.Timestamp.now(tz=args.as_of_tz)
                if str(args.as_of).lower() in ("now","auto","today")
                else pd.Timestamp(args.as_of, tz=args.as_of_tz))

    # --- Load artifacts ---
    feat_path = args.model_dir / "artifacts" / "features.json"
    feat_cols: List[str] = _load_json(feat_path) or []
    if not feat_cols:
        raise FileNotFoundError(f"features.json not found or empty: {feat_path}")

    meta = _load_json(args.model_dir / "meta.json")
    train_args = meta.get("args", {})
    hl_default = float(train_args.get("ewm_halflife", 3.0))
    hl_pos_map = {"GK": hl_default, "DEF": hl_default, "MID": hl_default, "FWD": hl_default}
    for k in ["GK","DEF","MID","FWD"]:
        key = "ewm_halflife_pos"
        if isinstance(train_args.get(key), str) and f"{k}:" in train_args[key]:
            try:
                parts = dict(p.split(":") for p in train_args[key].split(","))
                hl_pos_map[k] = float(parts.get(k, hl_default))
            except Exception:
                pass
    ewm_min_periods = int(train_args.get("ewm_min_periods", 1))
    ewm_adjust = bool(train_args.get("ewm_adjust", False))

    # --- Load registry features up to --as-of ---
    pf = _load_players_form(args.features_root, args.form_version, seasons_all)
    tf = _load_team_form(args.features_root, args.form_version, seasons_all)
    tz = args.as_of_tz

    du = _coerce_ts(pf["date_played"], tz)
    pf_hist = pf[(pf["season"].isin(history)) | ((pf["season"] == args.future_season) & (du < as_of_ts))].copy()
    pf_hist = _ewm_shots_per_pos(pf_hist, hl_map=hl_pos_map, min_periods=ewm_min_periods, adjust=ewm_adjust)

    # --- Minutes forecast for target GWs ---
    minutes = pd.read_csv(args.minutes_csv, parse_dates=["date_sched"])
    if "season" not in minutes.columns:
        minutes["season"] = args.future_season

    gw_sel = _gw_for_selection(minutes)
    avail_gws = sorted(pd.unique(gw_sel.dropna().astype(int)))
    avail_gws = [int(x) for x in avail_gws]
    target_gws = [int(g) for g in avail_gws if g >= int(gw_from_req)][:int(args.n_future)]
    if not target_gws:
        raise RuntimeError(f"No target GWs >= {gw_from_req} in minutes CSV. Available: {avail_gws}")
    if args.strict_n_future and len(target_gws) < args.n_future:
        raise RuntimeError(f"Only {len(target_gws)} GW(s) available; wanted {args.n_future}. Available: {avail_gws}")

    minutes = minutes[gw_sel.isin(target_gws)].copy()
    if minutes.empty:
        raise RuntimeError("No minute rows after filtering target GWs.")

    # Merge venue from team fixtures for venue_bin calculation
    team_fix = _load_team_fixtures(args.fix_root, args.future_season, args.team_fixtures_filename)
    gw_key_m = _pick_gw_col(minutes.columns.tolist()) or "gw_orig"
    gw_key_t = _pick_gw_col(team_fix.columns.tolist()) or "gw_orig"
    venue_cols = ["season", "team_id", gw_key_t, "is_home"]
    vmap = team_fix[venue_cols].dropna(subset=[gw_key_t, "team_id"]).drop_duplicates()
    vmap = vmap.rename(columns={gw_key_t: gw_key_m})
    minutes = minutes.merge(vmap, how="left", on=["season","team_id",gw_key_m], validate="many_to_one")

    # Build future frame (one row per player fixture)
    fut = minutes.copy()
    fut["venue_bin"] = fut["is_home"].fillna(0).astype(int)
    if "fdr" not in fut.columns:
        fut["fdr"] = 0.0

    # --- Roster gating (authoritative filter for scoring set) ---
    allowed_pairs = _load_roster_pairs(
        teams_json=args.teams_json,
        season=args.future_season,
        league_filter=(args.league_filter.strip() or None)
    )
    fut = _apply_roster_gate(
        fut,
        allowed_pairs=allowed_pairs,
        season=args.future_season,
        where="ga_scoring",
        out_artifacts_dir=artifacts_dir,
        require_on_roster=args.require_on_roster
    )
    if fut.empty:
        raise RuntimeError("All rows were dropped by roster gating; nothing to score.")

    # Team z merge (by GW)
    tz_map = _team_z_venue(tf)
    if tz_map is not None and (gw_key_m in fut.columns):
        fut = fut.merge(tz_map.rename(columns={"gw_orig": gw_key_m}),
                        how="left", on=["season", gw_key_m, "team_id"], validate="many_to_one")
    else:
        fut["team_att_z_venue"] = np.nan
        fut["opp_def_z_venue"] = np.nan

    # Last-known snapshot features (days_since_last; prev_minutes if required by features)
    pull_cols = set([c for c in pf_hist.columns if ("_ewm" in c or "_roll" in c)])
    pull_cols |= {"is_active","minutes"}
    snap_cols = list(pull_cols)
    gw_cols = [c for c in ("gw_played","gw_orig","gw") if c in pf_hist.columns]
    base_cols = ["season","player_id","date_played","minutes","is_active"] + snap_cols + gw_cols
    last = _last_snapshot_per_player(pf_hist[base_cols].copy(),
                                     feature_cols=list(pull_cols),
                                     as_of_ts=as_of_ts, tz=tz)

    last_play = (pf_hist.sort_values(["player_id","season","date_played"])
                       .groupby(["season","player_id"], as_index=False).tail(1)
                       .loc[:, ["season","player_id","date_played"]])
    last_play["date_played"] = _coerce_ts(last_play["date_played"], tz)
    fut["date_sched"] = _coerce_ts(fut["date_sched"], tz)
    fut = fut.merge(last_play, how="left", on=["season","player_id"])
    fut["days_since_last"] = (fut["date_sched"] - fut["date_played"]).dt.days.clip(lower=0)
    fut.drop(columns=["date_played"], inplace=True)

    # (Optional) add prev_minutes / is_active if present in features
    last_small = last[["season","player_id"] + [c for c in last.columns if c not in ("season","player_id")]].copy()
    if "minutes" in last_small.columns:
        last_small = last_small.rename(columns={"minutes":"prev_minutes"})
    else:
        last_small["prev_minutes"] = np.nan
    fut = fut.merge(last_small[["season","player_id","prev_minutes","is_active"]], how="left", on=["season","player_id"])

    if fut.columns.duplicated().any():
        dups = fut.columns[fut.columns.duplicated()].tolist()
        logging.warning("Duplicate columns detected; keeping last occurrence: %s", set(dups))
        fut = fut.loc[:, ~fut.columns.duplicated(keep="last")]

    # Feature matrix in exact training order
    def _get_unique_col(df: pd.DataFrame, name: str) -> pd.Series:
        obj = df[name]
        if isinstance(obj, pd.DataFrame):
            return obj[obj.columns[-1]]
        return obj

    X = pd.DataFrame(index=fut.index)
    for c in feat_cols:
        if c in fut.columns:
            X[c] = _get_unique_col(fut, c)
        else:
            X[c] = np.nan

    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    med = X.median(numeric_only=True).fillna(0.0)

    # Predict per-90
    pos_ser = fut["pos"].astype(str).str.upper()
    g_p90_mean = _predict_per_pos("goals", X, pos_ser, args.model_dir)
    a_p90_mean = _predict_per_pos("assists", X, pos_ser, args.model_dir)

    # Optional Poisson heads
    g_p90_pois = _predict_poisson_per_pos("goals", X, pos_ser, args.model_dir, med=med)
    a_p90_pois = _predict_poisson_per_pos("assists", X, pos_ser, args.model_dir, med=med)

    # Scale to per-match using pred_minutes
    scale = pd.to_numeric(fut["pred_minutes"], errors="coerce").fillna(0.0).to_numpy() / 90.0
    pred_goals_mean   = g_p90_mean * scale
    pred_assists_mean = a_p90_mean * scale
    pred_goals_pois   = g_p90_pois * scale
    pred_assists_pois = a_p90_pois * scale

    # Probabilities
    rg90 = np.where(~np.isnan(g_p90_pois), g_p90_pois, g_p90_mean)
    ra90 = np.where(~np.isnan(a_p90_pois), a_p90_pois, a_p90_mean)

    have_mix = all(c in fut.columns for c in ["p_start","p_cameo","pred_start_head","pred_bench_cameo_head"])
    if have_mix:
        ps = fut["p_start"].clip(0,1).to_numpy()
        pc = fut["p_cameo"].clip(0,1).to_numpy()
        ms = np.clip(fut["pred_start_head"].to_numpy(), 0, None)
        mb = np.clip(fut["pred_bench_cameo_head"].to_numpy(), 0, None)

        lam_g_s = rg90 * (ms / 90.0); lam_g_b = rg90 * (mb / 90.0)
        lam_a_s = ra90 * (ms / 90.0); lam_a_b = ra90 * (mb / 90.0)

        p_goal_raw   = ps*(1.0 - np.exp(-lam_g_s)) + (1.0-ps)*pc*(1.0 - np.exp(-lam_g_b))
        p_assist_raw = ps*(1.0 - np.exp(-lam_a_s)) + (1.0-ps)*pc*(1.0 - np.exp(-lam_a_b))
    else:
        lam_g_eff = rg90 * scale
        lam_a_eff = ra90 * scale
        p_goal_raw   = 1.0 - np.exp(-lam_g_eff)
        p_assist_raw = 1.0 - np.exp(-lam_a_eff)

    p_goal = np.clip(p_goal_raw, 0, 1)
    p_assist = np.clip(p_assist_raw, 0, 1)

    # Optional isotonic calibration
    if args.apply_calibration:
        def _load_iso(fn: str) -> Dict[str, object]:
            p = args.model_dir / "artifacts" / fn
            try:
                return joblib.load(p) if p.exists() else {}
            except Exception:
                return {}
        g_iso = _load_iso("p_goal_isotonic_per_pos.joblib")
        a_iso = _load_iso("p_assist_isotonic_per_pos.joblib")
        for tag, iso in g_iso.items():
            m = pos_ser.eq(tag).to_numpy()
            if iso is not None and m.any():
                p_goal[m] = np.clip(iso.transform(p_goal[m]), 0, 1)
        for tag, iso in a_iso.items():
            m = pos_ser.eq(tag).to_numpy()
            if iso is not None and m.any():
                p_assist[m] = np.clip(iso.transform(p_assist[m]), 0, 1)

    p_return_any = 1.0 - (1.0 - p_goal) * (1.0 - p_assist)

    # Zero GK if requested
    if args.skip_gk and pos_ser.eq("GK").any():
        m = pos_ser.eq("GK").to_numpy()
        for arr in (g_p90_mean, a_p90_mean, g_p90_pois, a_p90_pois,
                    pred_goals_mean, pred_assists_mean, pred_goals_pois, pred_assists_pois,
                    p_goal, p_assist, p_return_any):
            arr[m] = 0.0

    # Assemble output
    cols = {
        "season": fut["season"].values,
        "gw_orig": fut[gw_key_m].values if gw_key_m in fut.columns else np.nan,
        "date_sched": fut["date_sched"].values,
        "player_id": fut["player_id"].values,
        "team_id": fut.get("team_id", pd.Series([np.nan]*len(fut))).values,
        "player": fut.get("player", pd.Series([np.nan]*len(fut))).values,
        "pos": fut["pos"].values,
        "pred_minutes": fut["pred_minutes"].values,
        "team_att_z_venue": fut.get("team_att_z_venue", pd.Series([np.nan]*len(fut))).values,
        "opp_def_z_venue": fut.get("opp_def_z_venue", pd.Series([np.nan]*len(fut))).values,
        "pred_goals_p90_mean": g_p90_mean,
        "pred_assists_p90_mean": a_p90_mean,
        "pred_goals_mean": pred_goals_mean,
        "pred_assists_mean": pred_assists_mean,
        "pred_goals_p90_poisson": g_p90_pois,
        "pred_assists_p90_poisson": a_p90_pois,
        "pred_goals_poisson": pred_goals_pois,
        "pred_assists_poisson": pred_assists_pois,
        "p_goal": p_goal,
        "p_assist": p_assist,
        "p_return_any": p_return_any
    }
    out = pd.DataFrame(cols)

    # sort and persist
    sort_keys = [k for k in ["gw_orig","team_id","player_id"] if k in out.columns]
    if sort_keys: out = out.sort_values(sort_keys).reset_index(drop=True)

    season_dir.mkdir(parents=True, exist_ok=True)
    gw_from_eff, gw_to_eff = int(min(target_gws)), int(max(target_gws))
    out_path = season_dir / f"GW{gw_from_eff}_{gw_to_eff}.csv"
    out.to_csv(out_path, index=False)

    diag = {
        "rows": int(len(out)),
        "season": str(args.future_season),
        "requested": {"gw_from": int(gw_from_req), "gw_to": int(gw_to_req), "n_future": int(args.n_future)},
        "available_team_gws": [int(x) for x in avail_gws],
        "scored_gws": [int(x) for x in target_gws],
        "as_of": str(as_of_ts),
        "out": str(out_path)
    }
    print(json.dumps(diag, indent=2))

if __name__ == "__main__":
    main()
