#!/usr/bin/env python3
"""
minutes_forecast.py — score future GWs using fixed v1 gates/calibration + roster gating.

Key updates:
- --form-source supports "player" or "team".
- FDR is a single venue-aware column named 'fdr':
    if is_home==1 -> fdr_home
    else          -> fdr_away
  Fallbacks: use the available side if one missing; else flat 'fdr' if present.
- Output columns reordered: all metadata first, then fdr, then predictions.

Reads fixtures from:
  <fix-root>/<season>/player_fixture_calendar.csv
Fallback for future rows from:
  <fix-root>/<season>/fixture_calendar.csv
Reads form from:
  <form-root>/<version>/<season>/(player_form|players_form|team_form).csv
"""

import argparse, json, logging
from pathlib import Path
from typing import Optional, List, Tuple, Set

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.models.minutes_model_builder import (
    _pick_gw_col, load_minutes, make_features,
    train_regressor, train_cameo_minutes_by_pos,
    parse_pos_thresholds, per_position_bench_cap_from_train,
    predict_with_model, taper_start_minutes
)

# ----------------------------- small utils ------------------------------------

def load_booster(path: Path) -> lgb.Booster:
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return lgb.Booster(model_file=str(path))

def predict_booster_prob(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    Xn = X.select_dtypes(include=[np.number]).fillna(0)
    p = booster.predict(Xn)
    return np.clip(p.astype(float), 0.0, 1.0)

def coerce_ts(x, tz: Optional[str]) -> pd.Timestamp:
    t = pd.to_datetime(x, errors="coerce")
    if t.tzinfo is None:
        return t.tz_localize(tz) if tz else t
    return t.tz_convert(tz) if tz else t

def build_date_used(df: pd.DataFrame, tz: Optional[str]) -> pd.Series:
    dp = pd.to_datetime(df.get("date_played"), errors="coerce")
    ds = pd.to_datetime(df.get("date_sched"), errors="coerce")
    out = dp.where(dp.notna(), ds)
    if tz:
        if out.dt.tz is None: out = out.dt.tz_localize(tz)
        else: out = out.dt.tz_convert(tz)
    return out

def numeric_series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")

def gw_coalesce_for_future(df: pd.DataFrame) -> pd.Series:
    gw_played = numeric_series_or_nan(df, "gw_played")
    gw_orig   = numeric_series_or_nan(df, "gw_orig")
    gw_any    = numeric_series_or_nan(df, "gw")
    gw_played_valid = gw_played.where(gw_played > 0)
    gw_orig_valid   = gw_orig.where(gw_orig > 0)
    gw_any_valid    = gw_any.where(gw_any > 0)
    return gw_played_valid.where(
        gw_played_valid.notna(),
        gw_orig_valid.where(gw_orig_valid.notna(), gw_any_valid)
    )

def _normalize_is_home_inplace(df: pd.DataFrame) -> None:
    if "is_home" in df.columns:
        df["is_home"] = pd.to_numeric(df["is_home"], errors="coerce").fillna(0).astype(int); return
    if "was_home" in df.columns:
        df["is_home"] = df["was_home"].astype(str).str.lower().isin(["1","true","yes"]).astype(int); return
    if "venue" in df.columns:
        df["is_home"] = df["venue"].astype(str).str.lower().eq("home").astype(int); return
    df["is_home"] = 0

def _tolerant_id_normalize_inplace(df: pd.DataFrame) -> None:
    # Accept fpl_id -> player_id
    if "player_id" not in df.columns:
        for alt in ["fpl_id", "playerId", "playerid"]:
            if alt in df.columns: df.rename(columns={alt:"player_id"}, inplace=True); break
    # Accept team|team_code -> team_id
    if "team_id" not in df.columns:
        for alt in ["team", "team_code", "teamId", "teamid"]:
            if alt in df.columns: df.rename(columns={alt:"team_id"}, inplace=True); break

# ----------------------------- form loading / FDR -----------------------------

def _read_form_csv(form_root: Path, form_version: str, season: str, source: str) -> Optional[pd.DataFrame]:
    """
    Load <form-root>/<version>/<season>/<source>_form.csv.
    For player source, try both 'player_form.csv' and 'players_form.csv'.
    """
    base = form_root / form_version / season
    candidates = [f"{source}_form.csv"]
    if source == "player":
        candidates = ["player_form.csv", "players_form.csv"]
    for name in candidates:
        p = base / name
        if p.exists():
            try:
                df = pd.read_csv(p)
                df["season"] = season
                _tolerant_id_normalize_inplace(df)
                return df
            except Exception as e:
                logging.warning("Failed to read %s: %s", p, e)
    logging.warning("No %s form CSV found under %s (tried: %s)", source, base, candidates)
    return None

def attach_fdr_unified(df: pd.DataFrame,
                       seasons: List[str],
                       form_root: Path,
                       form_version: str,
                       source: str) -> pd.DataFrame:
    """
    Unifies FDR attach:
      source == "player": join by (season, player_id, GW*)
      source == "team":   join by (season, team_id,   GW*)
    If form has fdr_home/fdr_away, compute venue-aware 'fdr'.
    Else if only 'fdr' exists, use it directly.
    """
    if df.empty:
        return df

    df = df.copy()
    _tolerant_id_normalize_inplace(df)
    _normalize_is_home_inplace(df)
    df["_gw_join"] = gw_coalesce_for_future(df)
    if source == "team":
        df["team_id"] = df["team_id"].astype(str)
    else:
        df["player_id"] = df["player_id"].astype(str)

    frames = []
    for s in seasons:
        form = _read_form_csv(form_root, form_version, s, source=source)
        if form is None or form.empty: continue

        _tolerant_id_normalize_inplace(form)
        # coalesce GW on form too
        gw_col = _pick_gw_col(form.columns.tolist()) or ("gw_orig" if "gw_orig" in form.columns else ("gw_played" if "gw_played" in form.columns else "gw"))
        for c in ("gw_played", "gw_orig", "gw"):
            if c in form.columns: form[c] = pd.to_numeric(form[c], errors="coerce")
        form["_gw_join"] = numeric_series_or_nan(form, gw_col)

        # keep relevant cols
        keep_keys = ["season", "_gw_join", "player_id"] if source == "player" else ["season", "_gw_join", "team_id"]
        _tolerant_id_normalize_inplace(form)
        cols = keep_keys + [c for c in ["fdr_home", "fdr_away", "fdr"] if c in form.columns]
        form_small = form[cols].dropna(subset=keep_keys).copy()

        # aggregate duplicate keys by mean to stabilize
        agg = {c: "mean" for c in ["fdr_home", "fdr_away", "fdr"] if c in form_small.columns}
        if agg:
            form_small = form_small.groupby(keep_keys, as_index=False).agg(agg)

        frames.append(form_small)

    if not frames:
        logging.warning("FDR attach: no form frames loaded for source=%s. 'fdr' will be NaN.", source)
        df.drop(columns=["_gw_join"], inplace=True, errors="ignore")
        return df

    form_union = pd.concat(frames, ignore_index=True)

    # Join
    on_keys = ["season", "_gw_join", "player_id"] if source == "player" else ["season", "_gw_join", "team_id"]
    # normalize dtypes for join stability
    for k in on_keys:
        if k in ["player_id", "team_id"]:
            df[k] = df[k].astype(str)
            form_union[k] = form_union[k].astype(str)

    merged = df.merge(form_union, how="left", on=on_keys, suffixes=("", "_form"))

    # Compute final fdr
    fdr_home = pd.to_numeric(merged.get("fdr_home"), errors="coerce") if "fdr_home" in merged.columns else None
    fdr_away = pd.to_numeric(merged.get("fdr_away"), errors="coerce") if "fdr_away" in merged.columns else None
    fdr_flat = pd.to_numeric(merged.get("fdr"), errors="coerce") if "fdr" in merged.columns else None
    is_home = pd.to_numeric(merged.get("is_home"), errors="coerce").fillna(0).astype(int)

    if (fdr_home is not None) or (fdr_away is not None):
        # venue-aware
        left = fdr_home if fdr_home is not None else pd.Series(np.nan, index=merged.index)
        right = fdr_away if fdr_away is not None else pd.Series(np.nan, index=merged.index)
        by_venue = np.where(is_home == 1, left, right)  # if away -> right
        both_missing = left.isna() & right.isna()
        # if venue choice is NaN but one side exists, use the available side (nanmean of the two)
        avg_lr = np.nanmean(np.vstack([left.values, right.values]), axis=0)
        fdr_final = np.where(np.isnan(by_venue) & (~both_missing), avg_lr, by_venue)
        # if still missing, fallback to flat fdr if present
        if fdr_flat is not None:
            fdr_final = np.where(both_missing, fdr_flat.values, fdr_final)
        merged["fdr"] = fdr_final
        src = "home/away"
    elif fdr_flat is not None:
        merged["fdr"] = fdr_flat.values
        src = "flat"
    else:
        merged["fdr"] = np.nan
        src = "none"

    # Diagnostics
    miss_rate = float(pd.isna(merged["fdr"]).mean())
    logging.info("FDR attached from %s form (%s). Missing rate: %.1f%%", source, src, 100.0 * miss_rate)

    merged.drop(columns=["_gw_join"], inplace=True, errors="ignore")
    return merged

# ----------------------------- fixtures (team calendar) -----------------------

def load_team_fixtures(fix_root: Path, season: str, filename: str) -> pd.DataFrame:
    path = fix_root / season / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing team fixtures: {path}")
    tf = pd.read_csv(path)

    for dc in ("date_sched","date_played"):
        if dc in tf.columns: tf[dc] = pd.to_datetime(tf[dc], errors="coerce")

    _normalize_is_home_inplace(tf)

    for c in ("gw_played","gw_orig","gw"):
        if c in tf.columns: tf[c] = pd.to_numeric(tf[c], errors="coerce")

    if "team_id" not in tf.columns:
        for alt in ["team","teamId","team_code"]:
            if alt in tf.columns: tf.rename(columns={alt:"team_id"}, inplace=True); break

    tf["season"] = season
    return tf

def attach_fixture_extras(df: pd.DataFrame, team_fix: pd.DataFrame) -> pd.DataFrame:
    """
    Attach: fbref_id, team, opponent_id, opponent (derive from home/away if needed).
    Join keys: (season, team_id, GW*).
    """
    if df.empty or team_fix.empty: return df
    df2 = df.copy(); _normalize_is_home_inplace(df2)
    df2["_gw_join"] = gw_coalesce_for_future(df2); df2["team_id"] = df2["team_id"].astype(str)

    tf = team_fix.copy(); _normalize_is_home_inplace(tf)
    tf["_gw_join"] = gw_coalesce_for_future(tf); tf["team_id"] = tf["team_id"].astype(str)

    tf_keep = ["season","team_id","_gw_join"] + [c for c in ["fbref_id","team","opponent_id","opponent","home","away"] if c in tf.columns]
    extras = tf[tf_keep].dropna(subset=["_gw_join"]).copy()

    merged = df2.merge(extras, how="left", on=["season","team_id","_gw_join"], suffixes=("", "_fx"))

    for col in ["fbref_id","team","opponent_id","opponent","home","away"]:
        fx = f"{col}_fx"
        if fx in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[fx])
            else:
                merged[col] = merged[fx]

    if "opponent" not in merged.columns or merged["opponent"].isna().any():
        if "home" in merged.columns and "away" in merged.columns:
            need = merged["opponent"].isna() if "opponent" in merged.columns else np.ones(len(merged), dtype=bool)
            opp = np.where(merged["is_home"] == 1, merged["away"], merged["home"])
            if "opponent" not in merged.columns: merged["opponent"] = np.nan
            merged.loc[need, "opponent"] = opp[need]

    merged.drop(columns=[c for c in merged.columns if c.endswith("_fx")] + ["_gw_join"],
                inplace=True, errors="ignore")
    return merged

# ----------------------------- roster gating ----------------------------------

def _load_roster_pairs(teams_json: Optional[Path],
                       season: str,
                       league_filter: Optional[str]) -> Optional[Set[Tuple[str, str]]]:
    if not teams_json: return None
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
        season_info = (obj or {}).get("career", {}).get(season)
        if not season_info: continue
        if league_filter:
            if str(season_info.get("league","")).strip().lower() != str(league_filter).strip().lower():
                continue
        for pl in season_info.get("players", []) or []:
            pid = str(pl.get("id","")).strip()
            if pid: allowed.add((str(team_id), pid))
    if not allowed:
        logging.warning("Roster map for %s produced 0 allowed pairs (league=%r).", season, league_filter)
    return allowed or None

def _apply_roster_gate(df: pd.DataFrame,
                       allowed_pairs: Optional[Set[Tuple[str, str]]],
                       season: str,
                       where: str,
                       out_artifacts_dir: Optional[Path] = None,
                       require_on_roster: bool = False) -> pd.DataFrame:
    if allowed_pairs is None or df.empty: return df
    tid = df.get("team_id").astype(str); pid = df.get("player_id").astype(str)
    mask_ok = tid.combine(pid, lambda a,b: (a,b) in allowed_pairs)

    dropped = int((~mask_ok).sum())
    if dropped:
        logging.info("Roster gate dropped %d %s row(s) not present on the %s roster.", dropped, where, season)
        if out_artifacts_dir is not None:
            out_artifacts_dir.mkdir(parents=True, exist_ok=True)
            df.loc[~mask_ok].to_csv(out_artifacts_dir / f"roster_dropped_{where}.csv", index=False)
        if require_on_roster:
            raise RuntimeError(f"--require-on-roster set: {dropped} {where} rows are not on the {season} roster.")
    return df.loc[mask_ok].copy()

# ----------------------------- build squad/synth ------------------------------

def build_asof_squad(hist: pd.DataFrame, as_of_ts: pd.Timestamp, tz: Optional[str]) -> pd.DataFrame:
    h = hist.copy()
    du = build_date_used(h, tz)
    h = h[du < as_of_ts].copy()
    if h.empty:
        return pd.DataFrame(columns=["player_id","player","pos","team_id","is_active"])
    gw_key = _pick_gw_col(h.columns.tolist()) or "gw_orig"
    h = h.sort_values(["player_id","season",du.name,gw_key])
    last = h.groupby("player_id", as_index=False).tail(1)
    for c in ["player","pos","team_id","is_active"]:
        if c not in last.columns: last[c] = 1 if c=="is_active" else np.nan
    return last[["player_id","player","pos","team_id","is_active"]].dropna(subset=["team_id"])

def synthesize_future_player_rows(team_fix: pd.DataFrame,
                                  squad: pd.DataFrame,
                                  target_gws: List[int]) -> pd.DataFrame:
    tf = team_fix[team_fix["season"].notna()].copy()
    gwn_team = gw_coalesce_for_future(tf)
    tf = tf[gwn_team.isin(target_gws)].copy()
    if tf.empty or squad.empty:
        return pd.DataFrame(columns=["season","gw_orig","gw_played","gw","date_sched","player_id","player","team_id","pos","is_home","minutes","is_starter","is_active","_is_synth"])
    keep_cols = [c for c in ["season","gw_orig","gw_played","gw","date_sched","team_id","is_home"] if c in tf.columns]
    tf_small = tf[keep_cols].copy()
    synth = tf_small.merge(squad, how="left", on="team_id")
    synth["minutes"] = np.nan; synth["is_starter"] = np.nan
    if "is_active" not in synth.columns: synth["is_active"] = 1
    synth["_is_synth"] = 1
    for c in ["gw_orig","gw_played","gw"]:
        if c in synth.columns: synth[c] = pd.to_numeric(synth[c], errors="coerce")
    return synth

# ----------------------------- main -------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-seasons", required=True)
    ap.add_argument("--future-season", required=True)
    ap.add_argument("--as-of", default="now")
    ap.add_argument("--as-of-tz", default="Africa/Lagos")
    ap.add_argument("--as-of-gw", type=int, required=True)
    ap.add_argument("--n-future", type=int, default=3)
    ap.add_argument("--gw-from", type=int, default=None)
    ap.add_argument("--gw-to", type=int, default=None)
    ap.add_argument("--strict-n-future", action="store_true")

    # IO
    ap.add_argument("--fix-root", default="data/processed/registry/fixtures")
    ap.add_argument("--team-fixtures-filename", default="fixture_calendar.csv")
    ap.add_argument("--out-dir", default="data/predictions/minutes_v1")
    ap.add_argument("--model-dir", default="data/models/minutes/versions/v1")

    # Features
    ap.add_argument("--use-fdr", action="store_true")
    ap.add_argument("--form-root", default="data/processed/registry/features")
    ap.add_argument("--form-version", default="v2")
    ap.add_argument("--form-source", choices=["team","player"], default="team")

    # Routing
    ap.add_argument("--gate-blend", type=float, default=0.25)
    ap.add_argument("--t-lo", type=float, default=0.25)
    ap.add_argument("--t-hi", type=float, default=0.65)
    ap.add_argument("--pos-thresholds", type=str, default="")
    ap.add_argument("--no-mix-gk", action="store_true")

    # Taper + caps
    ap.add_argument("--use-taper", action="store_true")
    ap.add_argument("--taper-lo", type=float, default=0.40)
    ap.add_argument("--taper-hi", type=float, default=0.70)
    ap.add_argument("--taper-min-scale", type=float, default=0.80)
    ap.add_argument("--use-pos-bench-caps", action="store_true")
    ap.add_argument("--bench-cap", type=float, default=45.0)

    # Roster gating
    ap.add_argument("--teams-json", type=Path)
    ap.add_argument("--league-filter", type=str, default="")
    ap.add_argument("--require-on-roster", action="store_true")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    fix_root = Path(args.fix_root)

    # GW window
    gw_from_req = args.gw_from if args.gw_from is not None else args.as_of_gw
    gw_to_req = args.gw_to if args.gw_to is not None else (gw_from_req + max(1, args.n_future) - 1)

    # Data
    history = [s.strip() for s in args.history_seasons.split(",") if s.strip()]
    seasons_all = history + [args.future_season]
    df = load_minutes(seasons_all, fix_root)

    as_of_ts = (pd.Timestamp.now(tz=args.as_of_tz)
                if str(args.as_of).lower() in ("now","auto","today")
                else coerce_ts(args.as_of, args.as_of_tz))

    team_fix = load_team_fixtures(fix_root, args.future_season, args.team_fixtures_filename)
    gwn_team = gw_coalesce_for_future(team_fix)
    avail_gws = sorted(pd.unique(gwn_team.dropna().astype(int)))
    target_gws = [g for g in avail_gws if g >= gw_from_req][:args.n_future]
    if not target_gws:
        raise RuntimeError(f"No future GWs available. requested [{gw_from_req},{gw_to_req}], available={avail_gws}")
    if args.strict_n_future and len(target_gws) < args.n_future:
        raise RuntimeError(f"Only {len(target_gws)} GW(s) available ≥ {gw_from_req}, but --n-future={args.n_future}")

    # Build as-of squad + synth
    date_used_full = build_date_used(df, args.as_of_tz)
    hist_mask = (df["season"].isin(history)) | ((df["season"]==args.future_season) & (date_used_full < as_of_ts))
    squad = build_asof_squad(df.loc[hist_mask], as_of_ts, args.as_of_tz)

    # Optional roster gate for synthesis
    league = args.league_filter.strip() or None
    allowed_pairs = _load_roster_pairs(args.teams_json, args.future_season, league)
    if allowed_pairs is not None and not squad.empty:
        mask_squad = squad["team_id"].astype(str).combine(squad["player_id"].astype(str), lambda a,b: (a,b) in allowed_pairs)
        dropped_squad = int((~mask_squad).sum())
        if dropped_squad:
            logging.info("Roster gate dropped %d as-of squad rows.", dropped_squad)
        squad = squad.loc[mask_squad].copy()

    synth = synthesize_future_player_rows(team_fix, squad, target_gws)
    if not synth.empty:
        gw_key = _pick_gw_col(df.columns.tolist()) or "gw_orig"
        if gw_key not in synth.columns:
            if "gw_orig" in synth.columns: gw_key = "gw_orig"
            elif "gw_played" in synth.columns: gw_key = "gw_played"
        existing_keys = set(tuple(x) for x in df[["season","player_id",gw_key,"team_id"]].dropna(subset=[gw_key]).to_records(index=False))
        keep_mask = ~synth.apply(lambda r: (r["season"], r["player_id"], r.get(gw_key, np.nan), r["team_id"]) in existing_keys, axis=1)
        synth = synth.loc[keep_mask].copy()
        df = pd.concat([df, synth], ignore_index=True)

    # Attach FDR (unified) and fixture extras
    if args.use_fdr:
        df = attach_fdr_unified(df, seasons_all, Path(args.form_root), args.form_version, args.form_source)
    df = attach_fixture_extras(df, team_fix)

    # Leakage-safe history mask
    date_used = build_date_used(df, args.as_of_tz)
    pre_asof_mask = (df["season"].isin(history)) | ((df["season"]==args.future_season) & (date_used < as_of_ts))
    pre_asof_mask &= df["minutes"].notna()

    # Features
    try:
        df = make_features(df, halflife_min=2.0, halflife_start=3.0, days_cap=14,
                           use_log_days=False, use_fdr=args.use_fdr, add_team_rotation=True,
                           taper_lo=args.taper_lo, taper_hi=args.taper_hi, taper_min_scale=args.taper_min_scale)
    except TypeError:
        df = make_features(df, halflife_min=2.0, halflife_start=3.0, days_cap=14,
                           use_log_days=False, use_fdr=args.use_fdr, add_team_rotation=True)

    # Neutralize rotation leakage on synth rows
    if "_is_synth" in df.columns and df["_is_synth"].fillna(0).eq(1).any():
        gw_key = _pick_gw_col(df.columns.tolist()) or "gw_orig"
        idx_pre = df.index[pre_asof_mask]
        tmp = df.loc[idx_pre, ["team_id","team_rot3","season",gw_key]].copy()
        tmp["_du"] = date_used.loc[idx_pre]
        tmp = tmp.sort_values(["team_id","_du",gw_key])
        last_rot = tmp.groupby("team_id")["team_rot3"].last()
        mask_synth = df["_is_synth"].fillna(0).eq(1)
        df.loc[mask_synth, "team_rot3"] = df.loc[mask_synth, "team_id"].map(last_rot).fillna(0.0)

    # Prediction slice
    gwn_pred = gw_coalesce_for_future(df)
    future_mask = (df["season"] == args.future_season)
    df_pred = df.loc[future_mask & gwn_pred.isin(target_gws)].copy()
    if df_pred.empty:
        diag = {"target_gws": target_gws, "future_unique_gws_in_df": sorted(pd.unique(gwn_pred[future_mask].dropna().astype(int)))}
        raise RuntimeError(f"No rows to score after synthesis. Diagnostics: {diag}")

    # Apply roster gate on scoring slice
    season_dir = (out_dir / f"{args.future_season}")
    artifacts_dir = season_dir / "artifacts"
    df_pred = _apply_roster_gate(df_pred, allowed_pairs, args.future_season, "minutes_scoring", artifacts_dir, args.require_on_roster)
    if df_pred.empty: raise RuntimeError("All rows were dropped by roster gating; nothing to score.")

    # Bench caps from history
    hist = df.loc[pre_asof_mask].copy()
    cameo_hist = hist[(hist["is_starter"]==0) & (hist["minutes"]>0)]
    bench_caps = ({"GK":5.0,"DEF":20.0,"MID":30.0,"FWD":30.0}
                  if cameo_hist.empty else cameo_hist.groupby("pos")["minutes"].quantile(0.95).to_dict())

    # Load models
    start_glob = load_booster(model_dir / "start_classifier_global.txt")
    start_pos = {p: load_booster(model_dir / f"start_classifier_{p}.txt")
                 for p in ["GK","DEF","MID","FWD"] if (model_dir / f"start_classifier_{p}.txt").exists()}
    iso_glob = joblib.load(model_dir / "start_classifier_global_iso.joblib") if (model_dir / "start_classifier_global_iso.joblib").exists() else None
    iso_pos = {p: joblib.load(model_dir / f"start_classifier_{p}_iso.joblib")
               for p in ["GK","DEF","MID","FWD"] if (model_dir / f"start_classifier_{p}_iso.joblib").exists()}
    p60_boost = load_booster(model_dir / "p60_direct.txt")
    p60_iso = joblib.load(model_dir / "p60_direct_calib.joblib") if (model_dir / "p60_direct_calib.joblib").exists() else None
    cameo_pos_boost = {p: load_booster(model_dir / f"cameo_given_bench_{p}.txt")
                       for p in ["GK","DEF","MID","FWD"] if (model_dir / f"cameo_given_bench_{p}.txt").exists()}
    cameo_glob_boost = load_booster(model_dir / "cameo_given_bench_global.txt") if (model_dir / "cameo_given_bench_global.txt").exists() else None
    cameo_iso_pos = {p: joblib.load(model_dir / f"cameo_given_bench_{p}_iso.joblib")
                     for p in ["GK","DEF","MID","FWD"] if (model_dir / f"cameo_given_bench_{p}_iso.joblib").exists()}
    cameo_iso_glob = joblib.load(model_dir / "cameo_given_bench_global_iso.joblib") if (model_dir / "cameo_given_bench_global_iso.joblib").exists() else None

    # Feature sets (use 'fdr' directly)
    feat_start = ["start_lag1","start_rate_hl3","min_lag1","min_ewm_hl2",
                  "played_last","days_feat","long_gap14","start_streak","bench_streak",
                  "pos_enc","team_rot3","fdr"]
    feat_reg = ["min_lag1","min_ewm_hl2","played_last","days_feat","long_gap14",
                "start_lag1","start_rate_hl3","pos_enc","team_rot3","fdr"]
    feat_p60 = feat_start
    feat_cameo = ["min_lag1","min_ewm_hl2","played_last","days_feat","long_gap14",
                  "start_rate_hl3","bench_streak","pos_enc","team_rot3","fdr"]
    feat_cameo_min = ["min_lag1","min_ewm_hl2","played_last","days_feat","long_gap14",
                      "start_rate_hl3","bench_streak","pos_enc","team_rot3","fdr"]

    # Gates
    def predict_p_start(df_in: pd.DataFrame) -> np.ndarray:
        X = df_in[feat_start].fillna(0)
        n = len(df_in); p_pos = np.full(n, np.nan)
        for pos in ["GK","DEF","MID","FWD"]:
            m = start_pos.get(pos)
            if m is None: continue
            rows = (df_in["pos"].values == pos)
            if not rows.any(): continue
            pv = predict_booster_prob(m, X.loc[rows])
            iso = iso_pos.get(pos)
            if iso is not None: pv = np.clip(iso.transform(pv), 0, 1)
            p_pos[rows] = pv
        p_glob = predict_booster_prob(start_glob, X)
        if iso_glob is not None: p_glob = np.clip(iso_glob.transform(p_glob), 0, 1)
        blend = float(np.clip(args.gate_blend, 0, 1))
        return np.clip(np.where(~np.isnan(p_pos), blend*p_pos + (1-blend)*p_glob, p_glob), 0, 1)

    def predict_p60(df_in: pd.DataFrame) -> np.ndarray:
        p = predict_booster_prob(p60_boost, df_in[feat_p60].fillna(0))
        if p60_iso is not None: p = np.clip(p60_iso.transform(p), 0, 1)
        return p

    def predict_p_cameo(df_in: pd.DataFrame) -> np.ndarray:
        X = df_in[feat_cameo].fillna(0)
        n = len(df_in); p = np.zeros(n, dtype=float); used = np.zeros(n, dtype=bool)
        for pos in ["GK","DEF","MID","FWD"]:
            m = cameo_pos_boost.get(pos)
            if m is None: continue
            rows = (df_in["pos"].values == pos)
            if not rows.any(): continue
            pv = predict_booster_prob(m, X.loc[rows])
            iso = cameo_iso_pos.get(pos)
            if iso is not None: pv = np.clip(iso.transform(pv), 0, 1)
            p[rows] = pv; used[rows] = True
        if (~used).any() and cameo_glob_boost is not None:
            pv = predict_booster_prob(cameo_glob_boost, X.loc[~used])
            if cameo_iso_glob is not None: pv = np.clip(cameo_iso_glob.transform(pv), 0, 1)
            p[~used] = pv
        return np.clip(p, 0, 1)

    # Refit minute regressors on leakage-safe history
    hist = df.loc[pre_asof_mask].copy()
    reg_start, _ = train_regressor(hist[hist["is_starter"]==1], feat_reg, objective="regression_l2")
    cameo_min_by_pos, _, cameo_min_global = train_cameo_minutes_by_pos(hist, feat_cameo_min, min_rows=150)

    def predict_cameo_minutes(df_in: pd.DataFrame) -> np.ndarray:
        Xc = df_in[feat_cameo_min].fillna(0)
        mu = np.zeros(len(df_in), dtype=float); used = np.zeros(len(df_in), dtype=bool)
        for pos in ["GK","DEF","MID","FWD"]:
            model = cameo_min_by_pos.get(pos)
            if model is None: continue
            rows = (df_in["pos"].values == pos)
            if not rows.any(): continue
            mu[rows] = predict_with_model(model, Xc.loc[rows], default_val=15.0, clip_hi=None)
            used[rows] = True
        if (~used).any() and cameo_min_global is not None:
            mu[~used] = predict_with_model(cameo_min_global, Xc.loc[~used], default_val=15.0, clip_hi=None)
        return np.clip(mu, 0, 60)

    # Score
    p_start = predict_p_start(df_pred)
    p60 = predict_p60(df_pred)
    p_cameo = predict_p_cameo(df_pred)
    mu_cameo = predict_cameo_minutes(df_pred)

    pred_start = predict_with_model(reg_start, df_pred[feat_reg], default_val=60.0, clip_hi=120)
    if args.use_taper:
        try:
            pred_start = taper_start_minutes(pred_start, p_start, df_pred["pos"], args.taper_lo, args.taper_hi, args.taper_min_scale)
        except TypeError:
            pred_start = taper_start_minutes(pred_start, p_start, df_pred["pos"])

    pred_bench_cameo = mu_cameo
    pred_bench = p_cameo * pred_bench_cameo
    pred_bench = (np.minimum(pred_bench, per_position_bench_cap_from_train(df_pred["pos"], bench_caps))
                  if args.use_pos_bench_caps else np.clip(pred_bench, 0, args.bench_cap))

    # Threshold mix
    tlo = np.full(len(df_pred), args.t_lo, float)
    thi = np.full(len(df_pred), args.t_hi, float)
    for pos, (a, b) in parse_pos_thresholds(args.pos_thresholds).items():
        rows = (df_pred["pos"].values == pos)
        if rows.any(): tlo[rows], thi[rows] = a, b

    mix = p_start * pred_start + (1.0 - p_start) * pred_bench
    is_gk = (df_pred["pos"].values == "GK")
    minutes_pred = np.empty(len(df_pred), dtype=float)
    if args.no_mix_gk and is_gk.any():
        minutes_pred[is_gk] = np.where(p_start[is_gk] >= thi[is_gk], pred_start[is_gk],
                                 np.where(p_start[is_gk] <= tlo[is_gk], pred_bench[is_gk], pred_start[is_gk]))
    if (~is_gk).any():
        idx = ~is_gk
        minutes_pred[idx] = np.where(p_start[idx] >= thi[idx], pred_start[idx],
                              np.where(p_start[idx] <= tlo[idx], pred_bench[idx], mix[idx]))
    minutes_pred = np.clip(minutes_pred, 0, 120)

    p_play = np.clip(p_start + (1.0 - p_start) * p_cameo, 0, 1)
    exp_minutes_points = np.clip(p_play + p60, 0, 2)

    # ---------------- Output: metadata first, then fdr, then predictions ----------------
    meta_cols_order = ["season","gw_played","gw_orig","gw","date_sched",
                       "fbref_id","team_id","team","opponent_id","opponent","is_home",
                       "player_id","player","pos"]
    meta_cols = [c for c in meta_cols_order if c in df_pred.columns]

    out = pd.DataFrame(index=df_pred.index)
    for c in meta_cols:
        out[c] = df_pred[c].values

    # FDR (venue-aware already applied in attach_fdr_unified)
    out["fdr"] = df_pred["fdr"].values if "fdr" in df_pred.columns else np.nan
    out["fdr"] = out["fdr"].astype("Int8")

    # Predictions
    out["p_start"] = p_start
    out["p60"] = p60
    out["p_cameo"] = p_cameo
    out["p_play"] = p_play
    out["pred_start_head"] = pred_start
    out["pred_bench_cameo_head"] = pred_bench_cameo
    out["pred_bench_head"] = pred_bench
    out["pred_minutes"] = minutes_pred
    out["exp_minutes_points"] = exp_minutes_points
    if "_is_synth" in df_pred.columns:
        out["_is_synth"] = df_pred["_is_synth"].fillna(0).astype(int).values

    # Stable sort
    sort_keys = [k for k in ["gw_played","gw_orig","gw","team_id","player_id"] if k in out.columns]
    if sort_keys: out = out.sort_values(sort_keys).reset_index(drop=True)

    # Write
    gw_from_eff, gw_to_eff = int(min(target_gws)), int(max(target_gws))
    season_dir = (out_dir / f"{args.future_season}"); season_dir.mkdir(parents=True, exist_ok=True)
    out_path = season_dir / f"GW{gw_from_eff}_{gw_to_eff}.csv"
    out.to_csv(out_path, index=False)

    print(json.dumps({
        "rows": int(len(out)),
        "season": args.future_season,
        "requested": {"gw_from": int(gw_from_req), "gw_to": int(gw_to_req), "n_future": int(args.n_future)},
        "available_team_gws": [int(g) for g in avail_gws],
        "scored_gws": [int(g) for g in target_gws],
        "as_of": str(as_of_ts),
        "out": str(out_path)
    }, indent=2))

if __name__ == "__main__":
    main()
