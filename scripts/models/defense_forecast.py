#!/usr/bin/env python3
r"""
defense_forecast.py — leak-free forecaster for CS & DCP using trained defense heads
Now with optional roster gating from master_teams.json.

Inputs
------
• Trained artifacts from defense_model_builder.py:
    - team_cs_lgbm.txt                  (LightGBM Booster; team-level CS classifier)
    - cs_isotonic.joblib                (optional isotonic calibration for team CS)
    - dcp_DEF_lgbm.joblib, dcp_MID_lgbm.joblib, dcp_FWD_lgbm.joblib  (LGBMRegressor per-90)
    - artifacts/cs_features_team.json   (authoritative feature order for CS team head)
    - artifacts/dcp_features.json       (authoritative feature order for DCP per-90 head)

• Minutes forecast CSV (from minutes_forecast.py):
    - Must include: season, player_id, team_id, pos, date_sched, pred_minutes
    - Should include: gw_played or gw_orig or gw     (any one works)
    - Nice-to-have: prob_played60_cal/prob_played60 (else fallback from E[min])
    - Nice-to-have: fdr, is_home (else venue inferred via fixtures)
    - Nice-to-have for mixture DCP: p_start, p_cameo, pred_start_head, pred_bench_cameo_head

• Registry features (history + up to --as-of for current season):
    - players_form.csv  (for last-known *_roll/_ewm DCP features)
    - team_form.csv     (for team_def_xga_venue[_z], team_possession_venue, opp_att_z_venue)

• Team fixtures (for venue map if minutes CSV doesn’t provide it):
    - fixtures/<season>/<filename> (default: fixture_calendar.csv) with is_home and a GW column

Roster gating (optional)
------------------------
• --teams-json points to master_teams.json with per-season rosters.
• --league-filter narrows to a specific league label.
• --require-on-roster forces a hard error if any rows are not on roster (else soft drop + audit CSV).

Output
------
<out-dir>/<SEASON>/GW<from>_<to>.csv with columns:
  season, gw_orig, date_sched, player_id, team_id, player, pos,
  pred_minutes, prob_played60_use,
  team_att_z_venue, team_def_xga_venue, team_def_xga_venue_z, team_possession_venue, opp_att_z_venue, fdr, venue_bin,
  p_teamCS, prob_cs,
  lambda90, expected_dc, prob_dcp
[+ optional: lambda_match if --dump-lambdas]

Notes
-----
• No retraining; strictly uses artifacts.
• No future leakage: player *_roll/_ewm come from last row strictly before --as-of.
• GK: included in CS; excluded from DCP (prob_dcp/expected_dc set NaN) unless --include-gk-dcp is passed.
"""

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib


# ----------------------------- helpers ----------------------------------------

def _load_json(p: Path) -> list | dict:
    if not p.exists():
        raise FileNotFoundError(f"Missing artifact: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def _pick_gw_col(cols: List[str]) -> Optional[str]:
    for k in ("gw_played","gw_orig","gw"):
        if k in cols:
            return k
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
    if miss:
        raise KeyError(f"players_form missing: {miss}")
    df["pos"] = df["pos"].astype(str).str.upper()
    return df

def _load_team_form(features_root: Path, form_version: str, seasons: List[str]) -> Optional[pd.DataFrame]:
    frames = []
    for s in seasons:
        fp = features_root / form_version / s / "team_form.csv"
        if fp.exists():
            t = pd.read_csv(fp, parse_dates=["date_played"])
            t["season"] = s
            frames.append(t)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def _map_team_context(tf: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Build per (season, gw_orig, team_id, is_home) table with:
      team_def_xga_venue, team_def_xga_venue_z, team_possession_venue,
      team_att_z_venue, opp_att_z_venue, is_home
    """
    if tf is None:
        return None

    t = tf.copy()

    # Mark is_home if venue exists
    if "venue" in t.columns:
        t["is_home"] = t["venue"].astype(str).str.lower().eq("home").astype(int)
    elif "was_home" in t.columns:
        t["is_home"] = t["was_home"].astype(str).str.lower().isin(["1","true","yes"]).astype(int)
    else:
        t["is_home"] = np.nan

    # team_def_xga_venue[_z]
    if "venue" in t.columns and {"def_xga_home_roll","def_xga_away_roll"}.issubset(t.columns):
        t["team_def_xga_venue"] = np.where(
            t["venue"].astype(str).str.lower().eq("home"),
            t["def_xga_home_roll"], t["def_xga_away_roll"]
        )
    elif "def_xga_roll" in t.columns:
        t["team_def_xga_venue"] = t["def_xga_roll"]
    else:
        t["team_def_xga_venue"] = np.nan

    if "venue" in t.columns and {"def_xga_home_roll_z","def_xga_away_roll_z"}.issubset(t.columns):
        t["team_def_xga_venue_z"] = np.where(
            t["venue"].astype(str).str.lower().eq("home"),
            t["def_xga_home_roll_z"], t["def_xga_away_roll_z"]
        )
    elif "def_xga_roll_z" in t.columns:
        t["team_def_xga_venue_z"] = t["def_xga_roll_z"]
    else:
        t["team_def_xga_venue_z"] = np.nan

    # possession (venue-aware if available)
    if "venue" in t.columns and {"possession_home_roll","possession_away_roll"}.issubset(t.columns):
        t["team_possession_venue"] = np.where(
            t["venue"].astype(str).str.lower().eq("home"),
            t["possession_home_roll"], t["possession_away_roll"]
        )
    elif "possession_roll" in t.columns:
        t["team_possession_venue"] = t["possession_roll"]
    elif "possession" in t.columns:
        t["team_possession_venue"] = t["possession"]
    else:
        t["team_possession_venue"] = np.nan

    # team_att_z_venue (for opp map and optional output)
    if "venue" in t.columns and {"att_xg_home_roll_z","att_xg_away_roll_z"}.issubset(t.columns):
        t["team_att_z_venue"] = np.where(
            t["venue"].astype(str).str.lower().eq("home"),
            t["att_xg_home_roll_z"], t["att_xg_away_roll_z"]
        )
    elif "att_xg_roll_z" in t.columns:
        t["team_att_z_venue"] = t["att_xg_roll_z"]
    else:
        t["team_att_z_venue"] = np.nan

    # opponent attacking z (from opponent's team_att_z_venue)
    if {"home_id","away_id","team_id"}.issubset(t.columns):
        t["opp_id"] = np.where(t["team_id"]==t["home_id"], t["away_id"], t["home_id"])
    else:
        t["opp_id"] = np.nan
    opp = (t[["season","gw_orig","team_id","team_att_z_venue"]]
           .drop_duplicates()
           .rename(columns={"team_id":"opp_id","team_att_z_venue":"opp_att_z_venue"}))

    out = (t.merge(opp, on=["season","gw_orig","opp_id"], how="left")
             [["season","gw_orig","team_id","is_home",
               "team_def_xga_venue","team_def_xga_venue_z",
               "team_possession_venue","team_att_z_venue","opp_att_z_venue"]])

    # numeric
    for c in ["team_def_xga_venue","team_def_xga_venue_z","team_possession_venue","team_att_z_venue","opp_att_z_venue"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Deduplicate on the full key (important for DGWs)
    dups = out.duplicated(["season","gw_orig","team_id","is_home"]).sum()
    if dups:
        logging.warning("team_ctx has %d duplicate rows across (season, gw_orig, team_id, is_home); keeping last.", dups)
        out = (out.sort_values(["season","gw_orig","team_id","is_home"])
                  .drop_duplicates(subset=["season","gw_orig","team_id","is_home"], keep="last"))

    return out

def _load_team_fixtures(fix_root: Path, season: str, filename: str) -> Optional[pd.DataFrame]:
    path = fix_root / season / filename
    if not path.exists():
        return None
    tf = pd.read_csv(path)
    tf["season"] = season
    if "is_home" not in tf.columns:
        if "was_home" in tf.columns:
            tf["is_home"] = tf["was_home"].astype(str).str.lower().isin(["1","true","yes"]).astype(int)
        elif "venue" in tf.columns:
            tf["is_home"] = tf["venue"].astype(str).str.lower().eq("home").astype(int)
        else:
            tf["is_home"] = 0
    for dc in ("date_sched","date_played"):
        if dc in tf.columns:
            tf[dc] = pd.to_datetime(tf[dc], errors="coerce")
    for c in ("gw_played","gw_orig","gw"):
        if c in tf.columns:
            tf[c] = pd.to_numeric(tf[c], errors="coerce")
    if "team_id" not in tf.columns:
        for alt in ("team","teamId","team_code"):
            if alt in tf.columns:
                tf = tf.rename(columns={alt:"team_id"})
                break
    return tf

def _gw_for_selection(df: pd.DataFrame) -> pd.Series:
    def num(s): return pd.to_numeric(df.get(s), errors="coerce")
    gwp = num("gw_played"); gwo = num("gw_orig"); gwa = num("gw")
    return gwo.where(gwp.isna() | (gwp <= 0), gwp).where(lambda x: x.notna(), gwa)

def _last_snapshot_per_player(df: pd.DataFrame, feature_cols: List[str], as_of_ts: pd.Timestamp, tz: Optional[str]) -> pd.DataFrame:
    du = pd.to_datetime(df["date_played"], errors="coerce")
    if tz:
        if du.dt.tz is None:
            du = du.dt.tz_localize(tz)
        else:
            du = du.dt.tz_convert(tz)
    hist = df[du < as_of_ts].copy()
    if hist.empty:
        return pd.DataFrame(columns=["season","player_id"] + feature_cols)
    hist = hist.sort_values(["player_id","season","date_played","gw_orig"])
    last = hist.groupby(["season","player_id"], as_index=False).tail(1).copy()
    keep = ["season","player_id"] + [c for c in feature_cols if c in last.columns]
    for c in feature_cols:
        if c not in last.columns:
            last[c] = np.nan
    return last[["season","player_id"] + feature_cols].copy()

def _poisson_tail_prob_vec(lam: np.ndarray, k: np.ndarray) -> np.ndarray:
    """P(K >= k) for K ~ Poisson(lam). Handles NaN/negatives gracefully."""
    lam = np.asarray(lam, dtype=float)
    k = np.asarray(k, dtype=int)
    out = np.zeros_like(lam, dtype=float)

    def tail_one(l, kk):
        if not np.isfinite(l) or l <= 0:
            return 0.0 if kk > 0 else 1.0
        if kk <= 0:
            return 1.0
        term = np.exp(-l) * (l ** kk) / np.math.factorial(kk)
        s = term
        i = kk + 1
        for _ in range(200):
            term *= (l / i)
            s += term
            if term < 1e-12:
                break
            i += 1
        return float(min(max(s, 0.0), 1.0))

    for i in range(lam.shape[0]):
        out[i] = tail_one(lam[i], int(k[i]))
    return out

def _parse_k90(s: str) -> Dict[str, int]:
    out = {"DEF":10, "MID":12, "FWD":12}
    if not s:
        return out
    for part in s.split(";"):
        if not part.strip():
            continue
        k, v = part.split(":")
        out[k.strip().upper()] = int(v)
    return out

# ---------- roster gating (same semantics as goals/assists) ----------

def _norm_label(s: str) -> str:
    return str(s or "").lower().replace("-", " ").replace("_", " ").strip()

def _load_roster_pairs(teams_json: Optional[Path],
                       season: str,
                       league_filter: Optional[str]) -> Optional[set[tuple[str, str]]]:
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

    lf = _norm_label(league_filter) if league_filter else ""
    allowed: set[tuple[str, str]] = set()

    for team_id, obj in (data or {}).items():
        season_info = (obj or {}).get("career", {}).get(season)
        if not season_info:
            continue
        if lf:
            if _norm_label(season_info.get("league", "")) != lf:
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
                       allowed_pairs: Optional[set[tuple[str, str]]],
                       season: str,
                       where: str,
                       out_artifacts_dir: Optional[Path] = None,
                       require_on_roster: bool = False) -> pd.DataFrame:
    """
    Keep only rows with (team_id, player_id) inside allowed_pairs.
    If require_on_roster is True and any rows are dropped, raise RuntimeError.
    """
    if allowed_pairs is None or df.empty:
        return df
    tid = df.get("team_id").astype(str)
    pid = df.get("player_id").astype(str)
    pairs = list(zip(tid.to_numpy(), pid.to_numpy()))
    mask_ok = np.fromiter(((a, b) in allowed_pairs for (a, b) in pairs), count=len(pairs), dtype=bool)

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
    # windowing
    ap.add_argument("--history-seasons", required=True, help="Comma list of past seasons")
    ap.add_argument("--future-season", required=True, help="Season to score (contains future GWs)")
    ap.add_argument("--as-of", default="now", help='ISO timestamp or "now"')
    ap.add_argument("--as-of-tz", default="Africa/Lagos")
    ap.add_argument("--as-of-gw", type=int, required=True, help="First unplayed GW at --as-of (next GW)")
    ap.add_argument("--n-future", type=int, default=3)
    ap.add_argument("--gw-from", type=int, default=None)
    ap.add_argument("--gw-to", type=int, default=None)
    ap.add_argument("--strict-n-future", action="store_true")

    # IO
    ap.add_argument("--features-root", type=Path, default=Path("data/processed/registry/features"))
    ap.add_argument("--form-version", required=True)
    ap.add_argument("--fix-root", type=Path, default=Path("data/processed/registry/fixtures"))
    ap.add_argument("--team-fixtures-filename", default="fixture_calendar.csv")
    ap.add_argument("--minutes-csv", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True, help="Folder with trained DEFENSE artifacts (a specific version)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/predictions/defense"))

    # Policy / inference tweaks
    ap.add_argument("--dcp-k90", type=str, default="DEF:10;MID:12;FWD:12")
    ap.add_argument("--require-pred-minutes", action="store_true")
    ap.add_argument("--include-gk-dcp", action="store_true", help="If set, computes DCP for GK (otherwise NaN)")
    ap.add_argument("--dump-lambdas", action="store_true", help="Include lambda_match column")
    ap.add_argument("--log-level", default="INFO")

    # Roster gating
    ap.add_argument("--teams-json", type=Path, help="Path to master_teams.json containing per-season rosters")
    ap.add_argument("--league-filter", type=str, default="", help="Optional league name (e.g., 'ENG-Premier League')")
    ap.add_argument("--require-on-roster", action="store_true",
                    help="If set, error out when any scoring rows are not on the future-season roster")

    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    # seasons & as-of
    history = [s.strip() for s in args.history_seasons.split(",") if s.strip()]
    seasons_all = history + [args.future_season]
    tz = args.as_of_tz
    as_of_ts = (pd.Timestamp.now(tz=tz)
                if str(args.as_of).lower() in ("now","auto","today")
                else pd.Timestamp(args.as_of, tz=tz))

    # Prepare season dirs for artifacts early (for roster audit dumps)
    season_dir = args.out_dir / f"{args.future_season}"
    artifacts_dir = season_dir / "artifacts"

    # --- Load artifacts ---
    cs_feats = _load_json(args.model_dir / "artifacts" / "cs_features_team.json")  # list[str]
    dcp_feats = _load_json(args.model_dir / "artifacts" / "dcp_features.json")     # list[str]

    cs_booster = lgb.Booster(model_file=str(args.model_dir / "team_cs_lgbm.txt"))
    iso_cs = None
    iso_path = args.model_dir / "cs_isotonic.joblib"
    if iso_path.exists():
        try:
            iso_cs = joblib.load(iso_path)
            logging.info("Loaded isotonic calibration for team CS.")
        except Exception:
            logging.warning("Failed loading isotonic calibration; proceeding without.")

    dcp_models: Dict[str, object] = {}
    for tag in ("DEF","MID","FWD"):
        p = args.model_dir / f"dcp_{tag}_lgbm.joblib"
        if p.exists():
            dcp_models[tag] = joblib.load(p)
    if not dcp_models:
        raise FileNotFoundError("No DCP per-pos models found (expected dcp_DEF/MID/FWD_lgbm.joblib).")

    # --- Load registry up to as-of ---
    pf = _load_players_form(args.features_root, args.form_version, seasons_all)
    tf = _load_team_form(args.features_root, args.form_version, seasons_all)
    team_ctx = _map_team_context(tf)

    # --- Minutes target window ---
    minutes = pd.read_csv(args.minutes_csv, parse_dates=["date_sched"])
    if "season" not in minutes.columns:
        minutes["season"] = args.future_season

    gw_sel = _gw_for_selection(minutes)
    avail_gws = sorted(pd.unique(gw_sel.dropna().astype(int)))
    gw_from_req = int(args.gw_from) if args.gw_from is not None else int(args.as_of_gw)
    gw_to_req = int(args.gw_to) if args.gw_to is not None else (gw_from_req + max(1, args.n_future) - 1)
    # Fix: compute n_future safely
    gw_to_req   = int(args.gw_to) if args.gw_to is not None else (gw_from_req + max(1, args.n_future) - 1)
    target_gws  = [g for g in avail_gws if gw_from_req <= g <= gw_to_req]
    if not target_gws:
        raise RuntimeError(f"No target GWs in [{gw_from_req}, {gw_to_req}] from minutes CSV. Available: {avail_gws}")
    if args.strict_n_future and len(target_gws) < args.n_future:
        raise RuntimeError(f"Only {len(target_gws)} GW(s) available; wanted {args.n_future}. Available: {avail_gws}")

    # restrict minutes to target_gws
    minutes = minutes[gw_sel.isin(target_gws)].copy()
    if minutes.empty:
        raise RuntimeError("No rows after GW filtering.")
    # ensure no duplicate columns
    if minutes.columns.duplicated().any():
        dups = minutes.columns[minutes.columns.duplicated()].tolist()
        logging.warning("Minutes had duplicate columns; dropping earlier duplicates: %s", set(dups))
        minutes = minutes.loc[:, ~minutes.columns.duplicated(keep="last")]

    # --- Roster gating (authoritative filter for what we score) ---
    allowed_pairs = _load_roster_pairs(
        teams_json=args.teams_json,
        season=args.future_season,
        league_filter=(args.league_filter.strip() or None)
    )
    minutes = _apply_roster_gate(
        minutes,
        allowed_pairs=allowed_pairs,
        season=args.future_season,
        where="def_scoring",
        out_artifacts_dir=artifacts_dir,
        require_on_roster=args.require_on_roster
    )
    if minutes.empty:
        raise RuntimeError("All rows were dropped by roster gating; nothing to score.")

    # --- Venue / FDR merge ---
    gw_key_m = _pick_gw_col(minutes.columns.tolist()) or "gw_orig"

    if "is_home" not in minutes.columns:
        team_fix = _load_team_fixtures(args.fix_root, args.future_season, args.team_fixtures_filename)
        if team_fix is not None:
            gw_key_t = _pick_gw_col(team_fix.columns.tolist()) or "gw_orig"
            vmap = (team_fix[["season","team_id",gw_key_t,"is_home"]]
                    .dropna(subset=[gw_key_t,"team_id"])
                    .drop_duplicates())
            vmap = vmap.rename(columns={gw_key_t: gw_key_m})
            minutes = minutes.merge(vmap, on=["season","team_id",gw_key_m], how="left", validate="many_to_one")
        else:
            minutes["is_home"] = 0
    minutes["venue_bin"] = minutes["is_home"].fillna(0).astype(int)
    if "fdr" not in minutes.columns:
        minutes["fdr"] = 0.0

    # --- Build team rows for CS features (keep minutes' GW key as-is) ---
    team_rows = (minutes[["season", gw_key_m, "team_id", "is_home", "venue_bin", "fdr", "date_sched"]]
                 .drop_duplicates()
                 .rename(columns={gw_key_m: "gw_orig"}))

    if team_ctx is not None:
        # venue-aware merge on (season, gw_orig, team_id, is_home)
        team_rows = team_rows.merge(
            team_ctx,
            on=["season", "gw_orig", "team_id", "is_home"],
            how="left",
            validate="many_to_one",
        )
    else:
        for c in ["team_def_xga_venue","team_def_xga_venue_z","team_possession_venue","team_att_z_venue","opp_att_z_venue"]:
            team_rows[c] = np.nan

    # Assemble X_cs in artifact order
    Xcs = pd.DataFrame(index=team_rows.index)
    for f in cs_feats:
        if f in team_rows.columns:
            Xcs[f] = pd.to_numeric(team_rows[f], errors="coerce")
        elif f == "venue_bin":
            Xcs[f] = team_rows["venue_bin"]
        elif f == "fdr":
            Xcs[f] = pd.to_numeric(team_rows["fdr"], errors="coerce")
        else:
            Xcs[f] = np.nan
    Xcs = Xcs.fillna(0.0)

    # Predict team CS
    p_team_raw = np.clip(cs_booster.predict(Xcs), 0, 1)
    p_team = np.clip(iso_cs.transform(p_team_raw), 0, 1) if iso_cs is not None else p_team_raw
    team_rows["p_teamCS"] = p_team

    # --- Player last snapshot for DCP *_roll/EWM features (leak-free) ---
    pull_cols = [c for c in dcp_feats if c not in ("venue_bin","fdr",
                                                   "team_possession_venue","opp_att_z_venue",
                                                   "team_def_xga_venue","team_def_xga_venue_z","team_att_z_venue")]
    pf_hist = pf[(pf["season"].isin(history)) |
                 ((pf["season"] == args.future_season) & (_coerce_ts(pf["date_played"], tz) < as_of_ts))].copy()
    last = _last_snapshot_per_player(pf_hist, feature_cols=pull_cols, as_of_ts=as_of_ts, tz=tz)

    # --- Build per-player future frame ---
    fut = minutes.copy()
    # attach team CS probs & context
    team_rows_for_merge = team_rows.rename(columns={"gw_orig": gw_key_m})
    fut = fut.merge(
        team_rows_for_merge[["season", gw_key_m, "team_id", "p_teamCS",
                             "team_def_xga_venue","team_def_xga_venue_z",
                             "team_possession_venue","team_att_z_venue","opp_att_z_venue"]],
        on=["season", gw_key_m, "team_id"], how="left", validate="many_to_one"
    )
    # attach last-known player feature snapshot
    fut = fut.merge(last, on=["season","player_id"], how="left", validate="many_to_one")

    # Deduplicate any accidental column name collisions (keep rightmost)
    if fut.columns.duplicated().any():
        dups = fut.columns[fut.columns.duplicated()].tolist()
        logging.warning("Future frame had duplicate columns; keeping last occurrence: %s", set(dups))
        fut = fut.loc[:, ~fut.columns.duplicated(keep="last")]

    # p60: prefer existing column(s); else fallback from E[min]
    p60_col = None
    for c in ("prob_played60_cal","prob_played60_raw","prob_played60","p60"):
        if c in fut.columns:
            p60_col = c; break
    if p60_col is None:
        fut["prob_played60_use"] = np.clip((pd.to_numeric(fut["pred_minutes"], errors="coerce") - 30.0) / 60.0, 0.0, 1.0)
    else:
        fut["prob_played60_use"] = pd.to_numeric(fut[p60_col], errors="coerce")
        miss = fut["prob_played60_use"].isna()
        if miss.any():
            fut.loc[miss, "prob_played60_use"] = np.clip((pd.to_numeric(fut.loc[miss,"pred_minutes"], errors="coerce") - 30.0) / 60.0, 0.0, 1.0)

    # Player CS probability
    fut["prob_cs"] = pd.to_numeric(fut["p_teamCS"], errors="coerce") * pd.to_numeric(fut["prob_played60_use"], errors="coerce")

    # --- DCP per-90 features in exact artifact order ---
    Xdcp = pd.DataFrame(index=fut.index)
    for f in dcp_feats:
        if f in fut.columns:
            Xdcp[f] = pd.to_numeric(fut[f], errors="coerce")
        elif f == "venue_bin":
            Xdcp[f] = fut["venue_bin"]
        elif f == "fdr":
            Xdcp[f] = pd.to_numeric(fut["fdr"], errors="coerce")
        else:
            Xdcp[f] = np.nan
    Xdcp = Xdcp.fillna(0.0)

    # Predict per-90 λ for DEF/MID/FWD (GK optional)
    pos = fut["pos"].astype(str).str.upper().to_numpy()
    lam90 = np.full(len(fut), np.nan, dtype=float)
    for tag, mdl in dcp_models.items():
        mask = (pos == tag)
        if mask.any():
            lam90[mask] = np.clip(mdl.predict(Xdcp.loc[mask]), 0, None)
    if not args.include_gk_dcp:
        lam90[(pos == "GK")] = np.nan

    # --- DCP scaling & probabilities (mixture-aware if available) ---
    m_pred = np.clip(pd.to_numeric(fut["pred_minutes"], errors="coerce").fillna(0).to_numpy(), 0, None)
    k90_map = _parse_k90(args.dcp_k90)

    def _k_from_minutes(pos_arr, mins_arr):
        return np.array([
            int(np.ceil(k90_map.get(p, 12) * (mm / 90.0))) if p in ("DEF","MID","FWD") else 9_999
            for p, mm in zip(pos_arr, mins_arr)
        ], dtype=int)

    have_mix = all(c in fut.columns for c in ["p_start","p_cameo","pred_start_head","pred_bench_cameo_head"])
    if have_mix:
        ps = pd.to_numeric(fut["p_start"], errors="coerce").fillna(0).clip(0,1).to_numpy()
        pc = pd.to_numeric(fut["p_cameo"], errors="coerce").fillna(0).clip(0,1).to_numpy()
        ms = np.clip(pd.to_numeric(fut["pred_start_head"], errors="coerce").fillna(0).to_numpy(), 0, None)
        mb = np.clip(pd.to_numeric(fut["pred_bench_cameo_head"], errors="coerce").fillna(0).to_numpy(), 0, None)

        lam_s = lam90 * (ms / 90.0)
        lam_b = lam90 * (mb / 90.0)

        k_s = _k_from_minutes(pos, ms)
        k_b = _k_from_minutes(pos, mb)

        p_s = _poisson_tail_prob_vec(np.where(np.isfinite(lam_s), lam_s, 0.0), k_s)
        p_b = _poisson_tail_prob_vec(np.where(np.isfinite(lam_b), lam_b, 0.0), k_b)

        # Mixture probability and mixture mean
        prob_dcp = ps * p_s + (1.0 - ps) * pc * p_b
        expected_dc = ps * lam_s + (1.0 - ps) * pc * lam_b
        lam_match = ps * lam_s + (1.0 - ps) * pc * lam_b

    else:
        lam_match = lam90 * (m_pred / 90.0)
        k_match = _k_from_minutes(pos, m_pred)
        prob_dcp = _poisson_tail_prob_vec(np.where(np.isfinite(lam_match), lam_match, 0.0), k_match)
        expected_dc = lam_match

    # Respect GK policy
    if not args.include_gk_dcp:
        gk_mask = (pos == "GK")
        prob_dcp[gk_mask] = np.nan
        expected_dc[gk_mask] = np.nan
        lam_match[gk_mask] = np.nan

    # --- Assemble output ---
    out_cols = {
        "season": fut["season"].values,
        # Output column must be named gw_orig, but we DO NOT rename in-frame to avoid duplicates
        "gw_orig": pd.to_numeric(fut[gw_key_m], errors="coerce").values,
        "date_sched": fut["date_sched"].values,
        "player_id": fut["player_id"].values,
        "team_id": fut.get("team_id", pd.Series([np.nan]*len(fut))).values,
        "player": fut.get("player", pd.Series([np.nan]*len(fut))).values,
        "pos": fut["pos"].values,
        "pred_minutes": fut["pred_minutes"].values,
        "prob_played60_use": fut["prob_played60_use"].values,
        # context (handy for downstream QA)
        "team_att_z_venue": fut.get("team_att_z_venue", pd.Series([np.nan]*len(fut))).values,
        "team_def_xga_venue": fut.get("team_def_xga_venue", pd.Series([np.nan]*len(fut))).values,
        "team_def_xga_venue_z": fut.get("team_def_xga_venue_z", pd.Series([np.nan]*len(fut))).values,
        "team_possession_venue": fut.get("team_possession_venue", pd.Series([np.nan]*len(fut))).values,
        "opp_att_z_venue": fut.get("opp_att_z_venue", pd.Series([np.nan]*len(fut))).values,
        "fdr": fut.get("fdr", pd.Series([np.nan]*len(fut))).values,
        "venue_bin": fut.get("venue_bin", pd.Series([np.nan]*len(fut))).values,
        # outputs
        "p_teamCS": fut["p_teamCS"].values,
        "prob_cs": fut["prob_cs"].values,
        "lambda90": lam90,
        "expected_dc": expected_dc,
        "prob_dcp": prob_dcp,
    }
    out = pd.DataFrame(out_cols)
    if args.dump_lambdas:
        out["lambda_match"] = lam_match

    sort_keys = [k for k in ["gw_orig","team_id","player_id"] if k in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys).reset_index(drop=True)

    season_dir.mkdir(parents=True, exist_ok=True)
    gw_from_eff, gw_to_eff = int(min(target_gws)), int(max(target_gws))
    out_path = season_dir / f"GW{gw_from_eff}_{gw_to_eff}.csv"
    out.to_csv(out_path, index=False)

    diag = {
        "rows": int(len(out)),
        "season": str(args.future_season),
        "requested": {"gw_from": int(gw_from_req), "gw_to": int(gw_to_req), "n_future": int(args.n_future)},
        "available_gws": [int(x) for x in avail_gws],
        "scored_gws": [int(x) for x in target_gws],
        "as_of": str(as_of_ts),
        "out": str(out_path),
        "has_iso_cs": bool(iso_cs is not None),
        "dcp_models": list(dcp_models.keys()),
        "mixture_used": bool(have_mix),
        "roster_gate": {
            "enabled": bool(args.teams_json),
            "league_filter": args.league_filter or None,
            "require_on_roster": bool(args.require_on_roster),
        }
    }
    print(json.dumps(diag, indent=2))

if __name__ == "__main__":
    main()
