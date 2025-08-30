#!/usr/bin/env python3
"""
minutes_forecast.py — score future GWs using fixed v1 gates/calibration.

• Trains minutes regressors on: all history + current-season rows strictly before an as-of timestamp.
• Predicts for a GW window (default next 3 GWs).
• If player future rows are missing, synthesizes them from the team fixture calendar + an as-of squad snapshot.
• Robust GW selection:
    - Prefer gw_played only if >0, else fall back to gw_orig, else gw.
    - Soft-selects the next available GWs ≥ --as-of-gw (up to --n-future).

Reads fixtures from:
  <fix-root>/<season>/player_fixture_calendar.csv
Fallback for future rows from:
  <fix-root>/<season>/fixture_calendar.csv
Reads FDR (optional) from:
  <form-root>/<version>/<season>/<team|player>_form.csv
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from scripts.models.minutes_model_builder import (
    _pick_gw_col, load_minutes, attach_fdr, make_features,
    train_regressor, train_cameo_minutes_by_pos,
    parse_pos_thresholds, per_position_bench_cap_from_train,
    predict_with_model, taper_start_minutes
)

# ----------------------------- utils -----------------------------------------

def load_booster(path: Path) -> lgb.Booster:
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return lgb.Booster(model_file=str(path))

def predict_booster_prob(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    Xn = X.select_dtypes(include=[np.number]).fillna(0)
    p = booster.predict(Xn)  # binary → prob
    return np.clip(p.astype(float), 0.0, 1.0)

def coerce_ts(x, tz: str | None) -> pd.Timestamp:
    t = pd.to_datetime(x, errors="coerce")
    if t.tzinfo is None:
        return t.tz_localize(tz) if tz else t
    return t.tz_convert(tz) if tz else t

def build_date_used(df: pd.DataFrame, tz: str | None) -> pd.Series:
    """Prefer date_played; fallback to date_sched; localize/convert to tz."""
    dp = pd.to_datetime(df.get("date_played"), errors="coerce")
    ds = pd.to_datetime(df.get("date_sched"), errors="coerce")
    out = dp.where(dp.notna(), ds)
    if tz:
        if out.dt.tz is None:
            out = out.dt.tz_localize(tz)
        else:
            out = out.dt.tz_convert(tz)
    return out

def numeric_series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric Series aligned to df.index if column missing."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")

def gw_coalesce_for_future(df: pd.DataFrame) -> pd.Series:
    """GW key for selection: gw_played>0 → gw_orig>0 → gw>0."""
    gw_played = numeric_series_or_nan(df, "gw_played")
    gw_orig   = numeric_series_or_nan(df, "gw_orig")
    gw_any    = numeric_series_or_nan(df, "gw")
    gw_played_valid = gw_played.where(gw_played > 0)
    gw_orig_valid   = gw_orig.where(gw_orig > 0)
    gw_any_valid    = gw_any.where(gw_any > 0)
    gwn_pred = gw_played_valid.where(
        gw_played_valid.notna(),
        gw_orig_valid.where(gw_orig_valid.notna(), gw_any_valid)
    )
    return gwn_pred

def load_team_fixtures(fix_root: Path, season: str, filename: str) -> pd.DataFrame:
    """Load team fixtures (scheduled + played) for a season."""
    path = fix_root / season / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing team fixtures: {path}")
    tf = pd.read_csv(path)

    # tolerant date parsing
    for dc in ("date_sched", "date_played"):
        if dc in tf.columns:
            tf[dc] = pd.to_datetime(tf[dc], errors="coerce")

    # normalize is_home
    if "is_home" not in tf.columns:
        if "was_home" in tf.columns:
            tf["is_home"] = tf["was_home"].astype(str).str.lower().isin(["1","true","yes"]).astype(int)
        elif "venue" in tf.columns:
            tf["is_home"] = tf["venue"].astype(str).str.lower().eq("home").astype(int)
        else:
            tf["is_home"] = 0

    # Ensure numeric GWs
    for c in ("gw_played","gw_orig","gw"):
        if c in tf.columns:
            tf[c] = pd.to_numeric(tf[c], errors="coerce")

    # Team id column naming tolerance
    if "team_id" not in tf.columns:
        # try some common variants
        for alt in ["team", "teamId", "team_code"]:
            if alt in tf.columns:
                tf = tf.rename(columns={alt: "team_id"})
                break

    tf["season"] = season
    return tf

def build_asof_squad(hist: pd.DataFrame, as_of_ts: pd.Timestamp, tz: str | None) -> pd.DataFrame:
    """
    From historical minutes rows strictly before as_of_ts, build the latest-known
    (player_id → player, pos, team_id, is_active) snapshot.
    """
    # Work with a copy to avoid surprises
    h = hist.copy()
    # Reuse date selection in local tz
    du = build_date_used(h, tz)
    h = h[du < as_of_ts].copy()
    if h.empty:
        return pd.DataFrame(columns=["player_id","player","pos","team_id","is_active"])

    # sort so tail(1) yields last known state per player
    gw_key = _pick_gw_col(h.columns.tolist()) or "gw_orig"
    h = h.sort_values(["player_id", "season", du.name, gw_key])
    last = h.groupby("player_id", as_index=False).tail(1)

    # Ensure required columns exist
    for c in ["player","pos","team_id","is_active"]:
        if c not in last.columns:
            if c == "is_active":
                last[c] = 1
            else:
                last[c] = np.nan

    return last[["player_id","player","pos","team_id","is_active"]].dropna(subset=["team_id"])

def synthesize_future_player_rows(team_fix: pd.DataFrame,
                                  squad: pd.DataFrame,
                                  target_gws: list[int]) -> pd.DataFrame:
    """
    Cross-join each target team fixture with as-of squad rows of the same team_id
    to create player-level future rows (minutes/is_starter unknown).
    """
    tf = team_fix[team_fix["season"].notna()].copy()
    gwn_team = gw_coalesce_for_future(tf)
    tf = tf[gwn_team.isin(target_gws)].copy()
    if tf.empty or squad.empty:
        return pd.DataFrame(columns=[
            "season","gw_orig","gw_played","date_sched","player_id","player","team_id","pos","is_home",
            "minutes","is_starter","is_active","_is_synth"
        ])

    # Prepare small fixtures frame for join
    keep_cols = [c for c in ["season","gw_orig","gw_played","gw","date_sched","team_id","is_home"] if c in tf.columns]
    tf_small = tf[keep_cols].copy()

    # Left join squad per team_id (cartesian via merge)
    synth = tf_small.merge(squad, how="left", on="team_id", suffixes=("", ""))
    synth["minutes"] = np.nan
    synth["is_starter"] = np.nan
    if "is_active" not in synth.columns:
        synth["is_active"] = 1
    synth["_is_synth"] = 1

    # prefer gw_orig/gw_played columns
    for c in ["gw_orig","gw_played","gw"]:
        if c in synth.columns:
            synth[c] = pd.to_numeric(synth[c], errors="coerce")

    return synth

# ----------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # Windows
    ap.add_argument("--history-seasons", required=True, help="Comma list of past seasons")
    ap.add_argument("--future-season", required=True, help="Season to score (contains future GWs)")
    ap.add_argument("--as-of", default="now", help='ISO timestamp or "now" (default)')
    ap.add_argument("--as-of-tz", default="Africa/Lagos", help="Timezone to localize naive dates")
    ap.add_argument("--as-of-gw", type=int, required=True, help="First GW not yet played at --as-of (the next GW)")
    # Range control
    ap.add_argument("--n-future", type=int, default=3, help="Number of future GWs to score (default 3)")
    ap.add_argument("--gw-from", type=int, default=None, help="Override start GW; default = --as-of-gw")
    ap.add_argument("--gw-to", type=int, default=None, help="Override end GW; default = gw-from + n-future - 1")
    ap.add_argument("--strict-n-future", action="store_true",
                    help="If set, error out when fewer than --n-future GWs are available (default: proceed with what's available)")

    # IO
    ap.add_argument("--fix-root", default="data/processed/registry/fixtures")
    ap.add_argument("--team-fixtures-filename", default="fixture_calendar.csv",
                    help="Filename of team fixtures under <fix-root>/<season>/ (default: fixture_calendar.csv)")
    ap.add_argument("--out-dir", default="data/predictions/minutes_v1")
    ap.add_argument("--model-dir", default="data/models/minutes/versions/v1")

    # Features (mirror v1)
    ap.add_argument("--use-fdr", action="store_true")
    ap.add_argument("--form-root", default="data/processed/registry/features")
    ap.add_argument("--form-version", default="v2")
    ap.add_argument("--form-source", choices=["team","player"], default="team")

    # Routing config (v1)
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

    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    fix_root = Path(args.fix_root)

    # Resolve GW range requested
    gw_from_req = args.gw_from if args.gw_from is not None else args.as_of_gw
    gw_to_req = args.gw_to if args.gw_to is not None else (gw_from_req + max(1, args.n_future) - 1)

    # Load fixtures for history + (whatever player rows exist) future
    history = [s.strip() for s in args.history_seasons.split(",") if s.strip()]
    seasons_all = history + [args.future_season]
    df = load_minutes(seasons_all, fix_root)

    # Determine as-of
    as_of_ts = (pd.Timestamp.now(tz=args.as_of_tz)
                if str(args.as_of).lower() in ("now", "auto", "today")
                else coerce_ts(args.as_of, args.as_of_tz))

    # If player future rows are missing for future season, synthesize from team fixtures
    # 1) Find available GWs in team fixture file
    team_fix = load_team_fixtures(fix_root, args.future_season, args.team_fixtures_filename)
    gwn_team = gw_coalesce_for_future(team_fix)
    avail_gws = sorted(pd.unique(gwn_team.dropna().astype(int)))
    target_gws = [g for g in avail_gws if g >= gw_from_req][:args.n_future]
    if not target_gws:
        diag = {
            "requested_from_to": [int(gw_from_req), int(gw_to_req)],
            "available_team_gws": avail_gws
        }
        raise RuntimeError(f"No future GWs available in team fixtures for selection. Diagnostics: {diag}")
    if args.strict_n_future and len(target_gws) < args.n_future:
        raise RuntimeError(
            f"Only {len(target_gws)} GW(s) available ≥ {gw_from_req}, "
            f"but --n-future={args.n_future} and --strict-n-future was set. "
            f"Available: {avail_gws}"
        )

    # 2) Build as-of squad from history (strictly before as_of_ts)
    date_used_full = build_date_used(df, args.as_of_tz)
    hist_mask = (df["season"].isin(history)) | (
        (df["season"] == args.future_season) & (date_used_full < as_of_ts)
    )
    squad = build_asof_squad(df.loc[hist_mask], as_of_ts, args.as_of_tz)

    # 3) Synthesize player rows for target_gws that aren't already present
    synth = synthesize_future_player_rows(team_fix, squad, target_gws)

    # Avoid duplicating real rows: drop any synth rows that already exist in df
    if not synth.empty:
        gw_key = _pick_gw_col(df.columns.tolist()) or "gw_orig"
        if gw_key not in synth.columns:
            # prefer gw_orig if present
            if "gw_orig" in synth.columns:
                gw_key = "gw_orig"
            elif "gw_played" in synth.columns:
                gw_key = "gw_played"
        existing_keys = set(
            tuple(x) for x in df[["season", "player_id", gw_key, "team_id"]]
            .dropna(subset=[gw_key]).to_records(index=False)
        )
        keep_mask = ~synth.apply(
            lambda r: (r["season"], r["player_id"], r.get(gw_key, np.nan), r["team_id"]) in existing_keys,
            axis=1
        )
        synth = synth.loc[keep_mask].copy()

    # 4) Extend df with synthesized rows (if any)
    if not synth.empty:
        df = pd.concat([df, synth], ignore_index=True)

    # Attach FDR **after** synthesis so future rows get it
    if args.use_fdr:
        df = attach_fdr(df, seasons_all, Path(args.form_root), args.form_version, args.form_source)

    # Leakage control mask computed on the extended df
    date_used = build_date_used(df, args.as_of_tz)
    pre_asof_mask = (df["season"].isin(history)) | (
        (df["season"] == args.future_season) & (date_used < as_of_ts)
    )
    pre_asof_mask &= df["minutes"].notna()

    # Feature engineering (compat with old/new signatures)
    try:
        df = make_features(
            df,
            halflife_min=2.0, halflife_start=3.0,
            days_cap=14, use_log_days=False,
            use_fdr=args.use_fdr, add_team_rotation=True,
            taper_lo=args.taper_lo, taper_hi=args.taper_hi, taper_min_scale=args.taper_min_scale
        )
    except TypeError:
        df = make_features(
            df,
            halflife_min=2.0, halflife_start=3.0,
            days_cap=14, use_log_days=False,
            use_fdr=args.use_fdr, add_team_rotation=True
        )

    # Neutralize team_rot3 leakage on synthesized rows: set to last known per team pre-as-of
    if "_is_synth" in df.columns and df["_is_synth"].fillna(0).eq(1).any():
        # compute last known team_rot3 per team using only pre-as-of rows
        gw_key = _pick_gw_col(df.columns.tolist()) or "gw_orig"
        idx_pre = df.index[pre_asof_mask]
        tmp = df.loc[idx_pre, ["team_id", "team_rot3", "season", gw_key]].copy()
        # order by chronological keys; use date_used for stable ordering
        tmp["_du"] = date_used.loc[idx_pre]
        tmp = tmp.sort_values(["team_id", "_du", gw_key])
        last_rot = tmp.groupby("team_id")["team_rot3"].last()
        # apply to synth rows
        mask_synth = df["_is_synth"].fillna(0).eq(1)
        df.loc[mask_synth, "team_rot3"] = df.loc[mask_synth, "team_id"].map(last_rot).fillna(0.0)

    # ----- Build prediction slice from extended df -----
    gwn_pred = gw_coalesce_for_future(df)
    future_mask = (df["season"] == args.future_season)
    df_pred = df.loc[future_mask & gwn_pred.isin(target_gws)].copy()
    if df_pred.empty:
        diag = {
            "target_gws": target_gws,
            "future_unique_gws_in_df": sorted(pd.unique(gwn_pred[future_mask].dropna().astype(int)))
        }
        raise RuntimeError(f"No rows to score after synthesis. Diagnostics: {diag}")

    # Bench caps from historical bench cameos
    hist = df.loc[pre_asof_mask].copy()
    cameo_hist = hist[(hist["is_starter"] == 0) & (hist["minutes"] > 0)]
    if cameo_hist.empty:
        bench_caps = {"GK": 5.0, "DEF": 20.0, "MID": 30.0, "FWD": 30.0}
    else:
        bench_caps = cameo_hist.groupby("pos")["minutes"].quantile(0.95).to_dict()

    # Load v1 gates + calibrations
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

    # Feature sets
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

    # Gates (v1)
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
        p = np.where(~np.isnan(p_pos), blend*p_pos + (1-blend)*p_glob, p_glob)
        return np.clip(p, 0, 1)

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
    reg_start, _ = train_regressor(hist[hist["is_starter"] == 1], feat_reg, objective="regression_l2")
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

    # Score future rows
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
    if args.use_pos_bench_caps:
        pred_bench = np.minimum(pred_bench, per_position_bench_cap_from_train(df_pred["pos"], bench_caps))
    else:
        pred_bench = np.clip(pred_bench, 0, args.bench_cap)

    # thresholds
    tlo = np.full(len(df_pred), args.t_lo, float)
    thi = np.full(len(df_pred), args.t_hi, float)
    for pos, (a, b) in parse_pos_thresholds(args.pos_thresholds).items():
        rows = (df_pred["pos"].values == pos)
        if rows.any(): tlo[rows], thi[rows] = a, b

    mix = p_start * pred_start + (1.0 - p_start) * pred_bench
    is_gk = (df_pred["pos"].values == "GK")
    minutes_pred = np.empty(len(df_pred), dtype=float)
    if args.no_mix_gk and is_gk.any():
        minutes_pred[is_gk] = np.where(
            p_start[is_gk] >= thi[is_gk], pred_start[is_gk],
            np.where(p_start[is_gk] <= tlo[is_gk], pred_bench[is_gk], pred_start[is_gk])
        )
    if (~is_gk).any():
        idx = ~is_gk
        minutes_pred[idx] = np.where(
            p_start[idx] >= thi[idx], pred_start[idx],
            np.where(p_start[idx] <= tlo[idx], pred_bench[idx], mix[idx])
        )
    minutes_pred = np.clip(minutes_pred, 0, 120)

    p_play = np.clip(p_start + (1.0 - p_start) * p_cameo, 0, 1)
    exp_minutes_points = np.clip(p_play + p60, 0, 2)

    # Build output (include both GW columns if available)
    out_cols = {
        "season": df_pred["season"].values,
        "player_id": df_pred["player_id"].values,
        "player": df_pred.get("player", np.nan).values,
        "team_id": df_pred.get("team_id", np.nan).values,
        "pos": df_pred["pos"].values,
        "date_sched": df_pred.get("date_sched", pd.NaT).values,
        "p_start": p_start, "p60": p60, "p_cameo": p_cameo, "p_play": p_play,
        "pred_start_head": pred_start,
        "pred_bench_cameo_head": pred_bench_cameo,
        "pred_bench_head": pred_bench,
        "pred_minutes": minutes_pred,
        "exp_minutes_points": exp_minutes_points,
        "fdr": df_pred.get("fdr", 0.0).values
    }
    if "gw_played" in df_pred.columns: out_cols["gw_played"] = df_pred["gw_played"].values
    if "gw_orig"   in df_pred.columns: out_cols["gw_orig"]   = df_pred["gw_orig"].values
    if "gw"        in df_pred.columns: out_cols["gw"]        = df_pred["gw"].values
    if "_is_synth" in df_pred.columns: out_cols["_is_synth"] = df_pred["_is_synth"].fillna(0).astype(int).values
    out = pd.DataFrame(out_cols)

    # Optional: sort for readability
    sort_keys = [k for k in ["gw_played","gw_orig","gw","team_id","player_id"] if k in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys).reset_index(drop=True)

    # Use the actual target_gws for naming and logging
    gw_from_eff, gw_to_eff = int(min(target_gws)), int(max(target_gws))
    out_path = out_dir / f"minutes_{args.future_season}_GW{gw_from_eff}_{gw_to_eff}_v1.csv"
    out.to_csv(out_path, index=False)
    print(json.dumps({
        "rows": int(len(out)),
        "season": args.future_season,
        "requested": {"gw_from": int(gw_from_req), "gw_to": int(gw_to_req), "n_future": int(args.n_future)},
        "available_team_gws": [int(g) for g in avail_gws],
        "scored_gws": [int(g) for g in target_gws],
        "as_of": str(as_of_ts),
        "out": str(out_path),
        "synthesized_rows": int(out.get("_is_synth", pd.Series([0]*len(out))).sum()) if "_is_synth" in out.columns else 0
    }, indent=2))

if __name__ == "__main__":
    main()
