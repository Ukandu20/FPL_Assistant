#!/usr/bin/env python3
"""
Defense CLI Template (Eval-only, Probabilities-only)

This is a **tailored** CLI template for the DEFENSE model that mirrors your GA/Assist CLI style
but sticks strictly to the defense spec:
  • Outputs **prob_cs** and **prob_dcp** only (no expected points, no forecasting).
  • CS is trained at **team–match** level (label=1 if GA==0); per-player prob_cs = p_teamCS × p60_i.
  • DCP uses **minutes-fair binary**: hit if observed contributions ≥ ceil(K90_pos · minutes/90).
    We predict λ90 (per 90 rate) per position, then use Poisson tail to turn into a probability.

Train/Eval split:
  • --seasons: comma list; the **last** is the TEST season
  • --first-test-gw: first GW (inclusive) in TEST season that belongs to the EVAL tail.

Outputs:
  models/defense/<version_name>/
    ├─ team_cs_lgbm.txt, cs_isotonic.joblib (optional)
    ├─ dcp_DEF_lgbm.joblib (and MID/FWD if trained)
    ├─ artifacts/
    │    cs_features_team.json, dcp_features.json,
    │    cs_feature_importances.csv, dcp_*_feature_importances.csv,
    │    cs_calibration_bins.csv (team-level reliability; if labels available)
    └─ predictions/
         defense_probabilities_eval__<test_season>__GW<from>_<to>.csv
         metrics_cs.json, metrics_dcp.json
         missing_minutes_join.csv

Notes:
  • Uses **true minutes** for DCP training (per-90 outcome + min-minutes filter + sample weights).
  • CS training does not use minutes (team label), minutes only enter at combination time via p60.
  • No predict-only mode; a separate **forecaster** will consume the saved models.
"""
from __future__ import annotations
import argparse, logging, re, math, json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, mean_absolute_error
import joblib

RANDOM_STATE = 42

# ─────────────────────────────── Helpers ───────────────────────────────

def _ensure_version_dirs(base: Path, bump: bool = False, tag: Optional[str] = None) -> tuple[Path, str]:
    base.mkdir(parents=True, exist_ok=True)
    if bump:
        # create a new incremental version directory
        existing = sorted([d.name for d in base.iterdir() if d.is_dir() and re.match(r"^v\d+$", d.name)])
        n = 1 + (int(existing[-1][1:]) if existing else 0)
        name = f"v{n}" + (f"_{tag}" if tag else "")
    else:
        name = (tag or "latest")
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out, name


def _parse_k90(s: str) -> Dict[str, int]:
    out: Dict[str,int] = {"DEF":10, "MID":12, "FWD":12}
    if not s:
        return out
    for part in s.split(";"):
        if not part.strip():
            continue
        k, v = part.split(":")
        out[k.strip().upper()] = int(v)
    return out


def _load_players(features_root: Path, form_version: str, seasons: List[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        fp = features_root / form_version / s / "players_form.csv"
        if not fp.is_file():
            logging.warning("Missing %s – skipped", fp)
            continue
        df = pd.read_csv(fp, parse_dates=["date_played"])
        df["season"] = s
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No players_form.csv files loaded")
    df = pd.concat(frames, ignore_index=True)

    need = {"season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"players_form missing required columns: {miss}")

    if "ga" not in df.columns:
        df["ga"] = np.nan

    # Defensive event columns (optional; filled if missing)
    if "clr" not in df.columns:
        logging.warning("`clr` missing; filling 0 for DCP counts.")
        df["clr"] = 0
    for c in ["blocks","tkl","int","recoveries","fdr_home","fdr_away"]:
        if c not in df.columns:
            df[c] = np.nan

    df = df.sort_values(["season","date_played","gw_orig","player_id","team_id"]).reset_index(drop=True)
    df["rid"] = df.index.astype(int)
    df["pos"] = df["pos"].astype(str).str.strip().str.upper()
    return df


def _merge_team_features(df_players: pd.DataFrame, features_root: Path, form_version: str) -> pd.DataFrame:
    seasons = sorted(df_players["season"].unique())
    t_frames = []
    for s in seasons:
        fp = features_root / form_version / s / "team_form.csv"
        if not fp.is_file():
            logging.warning("team_form missing for %s", s)
            continue
        tf = pd.read_csv(fp, parse_dates=["date_played"])
        tf["season"] = s
        t_frames.append(tf)
    if not t_frames:
        logging.warning("No team_form available – filling team features with NaN")
        for c in ["team_att_z_venue","opp_att_z_venue","team_def_xga_venue","team_def_xga_venue_z","team_possession_venue"]:
            df_players[c] = np.nan
        return df_players

    tf = pd.concat(t_frames, ignore_index=True)

    if "venue" in tf.columns and {"att_xg_home_roll_z","att_xg_away_roll_z"}.issubset(tf.columns):
        tf["team_att_z_venue"] = np.where(tf["venue"]=="Home", tf["att_xg_home_roll_z"], tf["att_xg_away_roll_z"])
    elif "att_xg_roll_z" in tf.columns:
        tf["team_att_z_venue"] = tf["att_xg_roll_z"]
    else:
        tf["team_att_z_venue"] = np.nan

    if "venue" in tf.columns and {"def_xga_home_roll","def_xga_away_roll"}.issubset(tf.columns):
        tf["team_def_xga_venue"] = np.where(tf["venue"]=="Home", tf["def_xga_home_roll"], tf["def_xga_away_roll"])
    elif "def_xga_roll" in tf.columns:
        tf["team_def_xga_venue"] = tf["def_xga_roll"]
    else:
        tf["team_def_xga_venue"] = np.nan

    if "venue" in tf.columns and {"def_xga_home_roll_z","def_xga_away_roll_z"}.issubset(tf.columns):
        tf["team_def_xga_venue_z"] = np.where(tf["venue"]=="Home", tf["def_xga_home_roll_z"], tf["def_xga_away_roll_z"])
    elif "def_xga_roll_z" in tf.columns:
        tf["team_def_xga_venue_z"] = tf["def_xga_roll_z"]
    else:
        tf["team_def_xga_venue_z"] = np.nan

    if "venue" in tf.columns and {"possession_home_roll","possession_away_roll"}.issubset(tf.columns):
        tf["team_possession_venue"] = np.where(tf["venue"]=="Home", tf["possession_home_roll"], tf["possession_away_roll"])
    elif "possession_roll" in tf.columns:
        tf["team_possession_venue"] = tf["possession_roll"]
    elif "possession" in tf.columns:
        tf["team_possession_venue"] = tf["possession"]
    else:
        tf["team_possession_venue"] = np.nan

    if {"home_id","away_id","team_id"}.issubset(tf.columns):
        tf["opp_id"] = np.where(tf["team_id"]==tf["home_id"], tf["away_id"], tf["home_id"])
    else:
        tf["opp_id"] = np.nan

    use_team = ["season","gw_orig","team_id","venue","date_played","team_att_z_venue","team_def_xga_venue","team_def_xga_venue_z","team_possession_venue","opp_id"]
    use_team = [c for c in use_team if c in tf.columns or c in ["season","gw_orig","team_id","venue","date_played"]]
    T = tf[use_team].drop_duplicates()
    df = df_players.merge(T, on=["season","gw_orig","team_id","venue","date_played"], how="left")

    if {"season","gw_orig","opp_id"}.issubset(df.columns):
        opp_map = tf[["season","gw_orig","team_id","team_att_z_venue"]].drop_duplicates().rename(
            columns={"team_id":"opp_id","team_att_z_venue":"opp_att_z_venue"}
        )
        df = df.merge(opp_map, on=["season","gw_orig","opp_id"], how="left")
    else:
        df["opp_att_z_venue"] = np.nan

    for c in ["team_att_z_venue","opp_att_z_venue","team_def_xga_venue","team_def_xga_venue_z","team_possession_venue"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_minutes_predictions(path: Path) -> pd.DataFrame:
    mp = pd.read_csv(path, parse_dates=["date_played"])
    need = {"season","gw_orig","date_played","player_id","pred_minutes"}
    if not need.issubset(set(mp.columns)):
        missing = need - set(mp.columns)
        raise KeyError(f"minutes_preds missing columns: {missing}")

    p60_cols = [c for c in ["prob_played60_cal","prob_played60_raw","prob_played60"] if c in mp.columns]
    if p60_cols:
        mp = mp.rename(columns={p60_cols[0]: "prob_played60_use"})
    else:
        mp["prob_played60_use"] = np.nan

    key = ["season","gw_orig","date_played","player_id"]
    mp = mp.sort_values(key).drop_duplicates(subset=key, keep="last")
    return mp[key + ["pred_minutes","prob_played60_use"]].copy()


def _build_cs_features_team(df_team: pd.DataFrame, use_z: bool) -> tuple[pd.DataFrame, list[str]]:
    df = df_team.copy()
    df["venue_bin"] = (df["venue"]=="Home").astype(int)
    df["fdr"] = np.where(df["venue"]=="Home", df.get("fdr_home", np.nan), df.get("fdr_away", np.nan)).astype(float)
    feats = ["venue_bin","fdr"]
    choose = ["opp_att_z_venue", "team_possession_venue"]
    choose.append("team_def_xga_venue_z" if use_z and "team_def_xga_venue_z" in df.columns else "team_def_xga_venue")
    feats += [c for c in choose if c in df.columns]
    return df[feats].copy(), feats


def _build_dcp_features(df: pd.DataFrame, na_thresh: float, use_z: bool) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df["venue_bin"] = (df["venue"]=="Home").astype(int)
    df["fdr"] = np.where(df["venue"]=="Home", df.get("fdr_home", np.nan), df.get("fdr_away", np.nan)).astype(float)
    base = ["venue_bin","fdr","team_possession_venue","opp_att_z_venue"]
    base.append("team_def_xga_venue_z" if use_z and "team_def_xga_venue_z" in df.columns else "team_def_xga_venue")
    roll = [c for c in df.columns if c.startswith("def_") and c.endswith(("_p90_roll","_p90_home_roll","_p90_away_roll","_p90_roll_z","_p90_home_roll_z","_p90_away_roll_z"))]
    roll = [c for c in roll if df[c].notna().mean() >= na_thresh]
    feats = [c for c in base if c in df.columns] + sorted(roll)
    return df[feats].copy(), feats


def _lgbm_cls(monotone: Optional[list[int]] = None) -> lgb.LGBMClassifier:
    params = dict(objective="binary", n_estimators=2000, learning_rate=0.03,
                  num_leaves=127, min_data_in_leaf=20, subsample=0.9, colsample_bytree=0.9,
                  reg_lambda=1.0, n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1)
    if monotone is not None:
        params["monotone_constraints"] = monotone
    return lgb.LGBMClassifier(**params)


def _lgbm_reg() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="regression", n_estimators=2000, learning_rate=0.03,
                             num_leaves=127, min_data_in_leaf=20, subsample=0.9, colsample_bytree=0.9,
                             reg_lambda=1.0, n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1)


def _poisson_tail_prob_vec(lam: np.ndarray, k: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    k = np.asarray(k, dtype=int)
    out = np.zeros_like(lam, dtype=float)

    def tail_one(l, kk):
        if not np.isfinite(l) or l <= 0:
            return 0.0 if kk > 0 else 1.0
        if kk <= 0:
            return 1.0
        term = math.exp(-l) * (l ** kk) / math.factorial(kk)
        s = term
        i = kk + 1
        for _ in range(200):
            term *= l / i
            s += term
            if term < 1e-12:
                break
            i += 1
        return float(min(max(s, 0.0), 1.0))

    for i in range(lam.shape[0]):
        out[i] = tail_one(lam[i], int(k[i]))
    return out


# ─────────────────────────────── Main ───────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True,
                    help="Comma-separated seasons; last is TEST season (e.g. 2021-2022,2022-2023,2023-2024)")
    ap.add_argument("--first-test-gw", type=int, default=26)

    ap.add_argument("--features-root", type=Path, default=Path("data/processed/registry/features"))
    ap.add_argument("--form-version", required=True)

    ap.add_argument("--use-z", action="store_true")
    ap.add_argument("--na-thresh", type=float, default=0.70)

    ap.add_argument("--minutes-preds", type=Path, required=True,
                    help="expected_minutes.csv from minutes model (must include pred_minutes and p60 or fallback)")
    ap.add_argument("--require-pred-minutes", action="store_true",
                    help="If set, drop target rows lacking p60 even if fallback is available")
    ap.add_argument("--fallback-p60-from-minutes", action="store_true",
                    help="If p60 missing, approximate from expected minutes as clip((E[min]-30)/60,0,1)")

    # Calibration & diagnostics
    ap.add_argument("--calibrate-team-cs", action="store_true")
    ap.add_argument("--reliability-bins", type=int, default=10)

    # DCP config
    ap.add_argument("--min-dcp-minutes", type=int, default=30,
                    help="Minimum true minutes to include a row in DCP training")
    ap.add_argument("--dcp-k90", type=str, default="DEF:10;MID:12;FWD:12",
                    help="Per-position per-90 threshold, e.g. 'DEF:10;MID:12;FWD:12'")
    ap.add_argument("--skip-gk", action="store_true",
                    help="Set GK prob_dcp to NaN and exclude GK from DCP training/metrics")

    # Monotone constraints (team CS)
    ap.add_argument("--monotone-cs", action="store_true")

    # Output versioning
    ap.add_argument("--model-out", type=Path, default=Path("data/models/defense"))
    ap.add_argument("--bump-version", action="store_true")
    ap.add_argument("--version-tag", type=str, default="")
    ap.add_argument("--log-level", default="INFO")

    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    test_season = seasons[-1]

    version_dir, version_name = _ensure_version_dirs(args.model_out, bump=args.bump_version, tag=(args.version_tag or None))
    latest_dir = args.model_out
    logging.info("Writing artifacts to %s (versioned: %s) and %s (latest).", version_dir, version_name, latest_dir)

    # Load data
    df = _load_players(args.features_root, args.form_version, seasons)
    df = _merge_team_features(df, args.features_root, args.form_version)

    # Team table for CS training
    team_keys = ["season","gw_orig","team_id","venue","date_played"]
    keep_cols = team_keys + [c for c in [
        "ga","fdr_home","fdr_away","team_att_z_venue","team_def_xga_venue","team_def_xga_venue_z","team_possession_venue","opp_att_z_venue"
    ] if c in df.columns]
    df_team = df[keep_cols].drop_duplicates(team_keys).reset_index(drop=True)

    # Features
    Xcs_team, cs_feats = _build_cs_features_team(df_team, use_z=args.use_z)
    Xdcp_all, dcp_feats = _build_dcp_features(df, na_thresh=args.na_thresh, use_z=args.use_z)

    # Train/Eval split using first-test-gw in the TEST season
    # Train = all rows outside the eval window; Eval = TEST season with gw >= first-test-gw
    gw_all_test = sorted(df.loc[df["season"]==test_season, "gw_orig"].dropna().unique())
    if not gw_all_test:
        raise ValueError(f"No gw_orig for season {test_season}")
    gw_from, gw_to = int(args.first_test_gw), int(gw_all_test[-1])

    is_eval_player = (df["season"]==test_season) & (df["gw_orig"]>=args.first_test_gw)
    df_eval = df.loc[is_eval_player].copy()
    df_train = df.loc[~is_eval_player].copy()

    # Map to team rows for team CS
    tgt_team_keys = df_eval[team_keys].drop_duplicates()
    is_tgt = pd.merge(df_team.reset_index(), tgt_team_keys, on=team_keys, how="inner")["index"].to_numpy()
    is_trn = np.setdiff1d(np.arange(len(df_team)), is_tgt)
    df_team_train = df_team.iloc[is_trn].copy()

    # Train team CS
    mono_map = {"venue_bin": +1, "fdr": -1, "opp_att_z_venue": -1,
                "team_def_xga_venue": -1, "team_def_xga_venue_z": -1,
                "team_possession_venue": +1}
    monotone = [mono_map.get(f, 0) for f in cs_feats] if args.monotone_cs else None
    cs_model = _lgbm_cls(monotone=monotone)
    y_cs_tr = (df_team_train["ga"]==0).astype(int).to_numpy()
    Xcs_tr = Xcs_team.loc[df_team_train.index]
    cs_model.fit(Xcs_tr, y_cs_tr)

    # Optional isotonic calibration on a temporal tail of train
    iso_cs = None
    if args.calibrate_team_cs:
        tr_sorted = df_team_train.sort_values(["season","date_played","gw_orig"]).copy()
        k = max(1, int(0.2 * len(tr_sorted)))
        fit_idx = tr_sorted.index[:-k]
        cal_idx = tr_sorted.index[-k:]
        X_fit = Xcs_team.loc[fit_idx]
        y_fit = (df_team_train.loc[fit_idx, "ga"]==0).astype(int).to_numpy()
        X_cal = Xcs_team.loc[cal_idx]
        y_cal = (df_team_train.loc[cal_idx, "ga"]==0).astype(int).to_numpy()
        if len(np.unique(y_fit))>=2 and len(np.unique(y_cal))>=2:
            cs_model = _lgbm_cls(monotone=monotone)
            cs_model.fit(X_fit, y_fit)
            p_cal_raw = cs_model.predict_proba(X_cal)[:,1]
            iso_cs = IsotonicRegression(out_of_bounds="clip").fit(p_cal_raw, y_cal)
            logging.info("Fitted isotonic calibration for team CS.")

    # Minutes join for EVAL players
    mp = _load_minutes_predictions(args.minutes_preds)
    key = ["season","gw_orig","date_played","player_id"]
    target_keys = df_eval[key + ["rid","team_id","player","pos","venue","minutes","ga"]].copy()
    join = target_keys.merge(mp, on=key, how="left", validate="many_to_one")

    if args.fallback_p60_from_minutes:
        pm = join["pred_minutes"].astype(float)
        proxy = (pm - 30.0) / 60.0
        join["prob_played60_use"] = join["prob_played60_use"].fillna(proxy.clip(0.0,1.0))

    # >>> FIX 1: strict vs lenient minutes mask (DGW-safe logic later uses 'kept')
    if args.require_pred_minutes:
        # strict: require both E[min] and p60
        miss_mask = join["pred_minutes"].isna() | join["prob_played60_use"].isna()
    else:
        # lenient: require E[min]; p60 may be NaN
        miss_mask = join["pred_minutes"].isna()

    n_drop = int(miss_mask.sum())
    cov = 100.0 * (1 - n_drop / max(1, len(join)))
    if n_drop:
        logging.warning("Dropping %d/%d EVAL rows with missing mins/p60 (coverage %.1f%%).", n_drop, len(join), cov)
    kept = join.loc[~miss_mask].reset_index(drop=True)

    # Team CS predictions for EVAL teams
    Xcs_team_full = Xcs_team.join(df_team[team_keys])
    tgt_team = kept[["season","gw_orig","team_id","venue","date_played"]].drop_duplicates()
    tgt_team_full = tgt_team.merge(Xcs_team_full.join(df_team[["ga"]]), on=team_keys, how="left")
    p_team_raw = cs_model.predict_proba(tgt_team_full[cs_feats])[:,1]
    p_team = iso_cs.predict(p_team_raw) if iso_cs is not None else p_team_raw
    tgt_team_full["p_teamCS"] = p_team

    # >>> FIX 2: DGW-safe merge on full team–match key
    team_match_key = ["season","gw_orig","team_id","venue","date_played"]
    dups = tgt_team_full.duplicated(team_match_key).sum()
    if dups:
        logging.warning("tgt_team_full has %d duplicate team–match rows; dropping duplicates on %s", dups, team_match_key)
    tt = tgt_team_full[team_match_key + ["p_teamCS"]].drop_duplicates(team_match_key)

    kept = kept.merge(tt, on=team_match_key, how="left", validate="many_to_one")
    prob_cs = kept["p_teamCS"].to_numpy() * kept["prob_played60_use"].to_numpy()

    # DCP training (λ90) on TRAIN players using **true minutes**
    Xdcp_tr_full, _ = _build_dcp_features(df_train, args.na_thresh, args.use_z)
    m_tr = df_train["minutes"].fillna(0).clip(lower=0)
    m90_tr = (m_tr / 90.0)
    c_clr = df_train["clr"].fillna(0)
    c_blk = df_train["blocks"].fillna(0)
    c_tkl = df_train["tkl"].fillna(0)
    c_int = df_train["int"].fillna(0)
    c_rec = df_train["recoveries"].fillna(0)
    pos_tr = df_train["pos"].astype(str).values

    with np.errstate(divide='ignore', invalid='ignore'):
        d_def = ((c_clr+c_blk+c_tkl+c_int) / m90_tr).replace([np.inf, -np.inf], np.nan)
        d_out = ((c_clr+c_blk+c_tkl+c_int+c_rec) / m90_tr).replace([np.inf, -np.inf], np.nan)
    y_dcp_p90 = np.zeros(len(df_train), dtype=float)
    y_dcp_p90[(pos_tr=="DEF")] = d_def.fillna(0).to_numpy()[(pos_tr=="DEF")]
    y_dcp_p90[(pos_tr=="MID")] = d_out.fillna(0).to_numpy()[(pos_tr=="MID")]
    y_dcp_p90[(pos_tr=="FWD")] = d_out.fillna(0).to_numpy()[(pos_tr=="FWD")]

    keep_dcp = (m_tr >= args.min_dcp_minutes).to_numpy()
    Xdcp_tr = Xdcp_tr_full.loc[keep_dcp]
    y_dcp_tr = y_dcp_p90[keep_dcp]
    pos_dcp = pos_tr[keep_dcp]
    sw = m90_tr[keep_dcp].clip(lower=1e-6).to_numpy()


    # Train ONLY DEF & MID
    dcp_models: Dict[str, lgb.LGBMRegressor] = {}
    for p in ["DEF", "MID", "FWD"]:
        mask = (pos_dcp == p)
        if mask.sum() == 0:
            continue
        model = _lgbm_reg()
        model.fit(Xdcp_tr.loc[mask], y_dcp_tr[mask], sample_weight=sw[mask])
        dcp_models[p] = model


    # DCP probability on EVAL players
    Xdcp_te, _ = _build_dcp_features(df.loc[kept["rid"].to_numpy()], args.na_thresh, args.use_z)
    pos_te = df.loc[kept["rid"].to_numpy(), "pos"].astype(str).to_numpy()

    # Only DEF, MID, FWD get λ90; GK remains NaN
    valid_dcp_pos = np.isin(pos_te, ["DEF", "MID", "FWD"])
    lam90 = np.full(len(kept), np.nan, dtype=float)
    for p, model in dcp_models.items():
        mask = (pos_te == p)
        if mask.any():
            lam90[mask] = np.clip(model.predict(Xdcp_te.loc[mask]), 0, None)

    m_exp = kept["pred_minutes"].to_numpy()
    lam_match = np.where(np.isfinite(lam90), lam90 * (m_exp / 90.0), np.nan)

    k90_map = _parse_k90(args.dcp_k90)
    k_match = np.array([
        int(np.ceil((k90_map.get(pos, 12)) * (mm/90.0))) if (np.isfinite(mm) and pos in ("DEF","MID","FWD")) else 9999
        for pos, mm in zip(pos_te, m_exp)
    ], dtype=int)

    # Compute tail prob safely, then mask to outfield only
    prob_dcp_all = _poisson_tail_prob_vec(
        np.where(np.isfinite(lam_match), lam_match, 0.0),
        k_match
    )
    prob_dcp = np.where(valid_dcp_pos, prob_dcp_all, np.nan)
    # Safety: force GK -> NaN (even if upstream mislabels sneak through)
    prob_dcp = np.where(kept["pos"].astype(str).to_numpy() == "GK", np.nan, prob_dcp)

    # Expected DC = λ_match (match-level mean). Only for DEF/MID/FWD; GK -> NaN
    exp_dc = np.where(valid_dcp_pos, lam_match, np.nan)
    exp_dc = np.where(kept["pos"].astype(str).to_numpy() == "GK", np.nan, exp_dc)





    # ── Save artifacts & predictions ──
    outdir = version_dir
    pred_dir = outdir / "predictions"
    art_dir = outdir / "artifacts"
    pred_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    # Missing minutes join (use the *same* mask we dropped on)
    if n_drop > 0:
        join.loc[miss_mask].to_csv(pred_dir / "missing_minutes_join.csv", index=False)

    # Save models for forecaster
    cs_model.booster_.save_model(str(outdir / "team_cs_lgbm.txt"))
    if iso_cs is not None:
        joblib.dump(iso_cs, outdir / "cs_isotonic.joblib")
    for p, mreg in dcp_models.items():
        joblib.dump(mreg, outdir / f"dcp_{p}_lgbm.joblib")

    # Feature artifacts
    pd.DataFrame({"feature": cs_feats, "importance": cs_model.feature_importances_}).to_csv(art_dir / "cs_feature_importances.csv", index=False)
    (art_dir / "cs_features_team.json").write_text(json.dumps(cs_feats, indent=2))
    (art_dir / "dcp_features.json").write_text(json.dumps(dcp_feats, indent=2))

    # --- True DC from actual events, aligned to kept order ---
    comp_true = df.loc[kept["rid"].to_numpy(), ["clr","blocks","tkl","int","recoveries"]].copy().fillna(0)
    pos_kept = kept["pos"].astype(str).to_numpy()

    base_true = comp_true["clr"] + comp_true["blocks"] + comp_true["tkl"] + comp_true["int"]
    true_dc = base_true + np.where(np.isin(pos_kept, ["MID","FWD"]), comp_true["recoveries"], 0)

    # GK not part of DCP schema → NaN
    true_dc = np.where(pos_kept == "GK", np.nan, true_dc)


    # Predictions (EVAL tail only)
    dump_cols = ["season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes",
                 "pred_minutes","prob_played60_use"]
    out = kept[dump_cols].copy()
    out["prob_cs"] = prob_cs
    out["prob_dcp"] = prob_dcp
    out["p_teamCS"] = kept["p_teamCS"]
    out["exp_dc"] = exp_dc
    out["true_dc"] = true_dc 


    fp = pred_dir / f"expected_defense.csv"
    out.to_csv(fp, index=False)
    logging.info("Wrote EVAL probabilities to %s", fp.resolve())

    # ── Metrics (optional) ──
    metrics_cs = {}
    # Player-level CS label: earned CS points (≥60 & GA==0 for non-FWD)
    y_cs = ((kept["minutes"] >= 60) & (kept["ga"] == 0) & (~kept["pos"].isin(["FWD"])) ).astype(int).to_numpy()
    p_cs_eval = out["prob_cs"].to_numpy()
    if len(np.unique(y_cs)) > 1:
        try:
            metrics_cs["auc_player"] = float(roc_auc_score(y_cs, p_cs_eval))
        except Exception:
            pass
    try:
        metrics_cs["brier_player"] = float(brier_score_loss(y_cs, p_cs_eval))
        metrics_cs["logloss_player"] = float(log_loss(y_cs, np.c_[1-p_cs_eval, p_cs_eval]))
    except Exception:
        pass

    # Team-level calibration table (build directly from team table to avoid DGW duplication)
    rel_bins = max(2, int(args.reliability_bins))
    team_eval = tgt_team_full.dropna(subset=["ga","p_teamCS"]).copy()
    if len(team_eval):
        lab = (team_eval["ga"]==0).astype(int).to_numpy()
        pte = team_eval["p_teamCS"].to_numpy()
        dfb = pd.DataFrame({"y":lab,"p":pte})
        dfb["bin"] = pd.qcut(dfb["p"], q=rel_bins, duplicates="drop")
        tab = dfb.groupby("bin", observed=False).agg(
            p_mean=("p","mean"), y_rate=("y","mean"), count=("y","size")
        ).reset_index()
        tab["abs_gap"] = (tab["p_mean"]-tab["y_rate"]).abs()
        tab.to_csv(art_dir / "cs_calibration_bins.csv", index=False)
        w = tab["count"].to_numpy(); w = w/w.sum()
        metrics_cs["ece_team"] = float(np.sum(w*tab["abs_gap"].to_numpy()))

    metrics_dcp = {}

    # Build once, aligned to 'kept' order
    pos_kept = kept["pos"].astype(str).to_numpy()
    outfield_mask = np.isin(pos_kept, ["DEF", "MID", "FWD"])

    # Observed counts from df (aligned via rid)
    comp = df.loc[kept["rid"].to_numpy(), ["clr","blocks","tkl","int","recoveries","minutes"]].copy().fillna(0)

    base_counts = comp["clr"] + comp["blocks"] + comp["tkl"] + comp["int"]
    # DEF excludes recoveries; MID/FWD include recoveries
    obs_counts_all = base_counts + np.where(np.isin(pos_kept, ["MID","FWD"]), comp["recoveries"], 0)

    # Minutes-fair thresholds by position
    k90_map = _parse_k90(args.dcp_k90)
    kmatch_all = np.ceil([
        k90_map.get(p, 12) * (m/90.0) if p in ("DEF","MID","FWD") else 9_999
        for p, m in zip(pos_kept, comp["minutes"])
    ]).astype(int)

    # Labels & preds using the SAME mask
    y_dcp = (obs_counts_all.to_numpy()[outfield_mask] >= kmatch_all[outfield_mask]).astype(int)
    p_dcp_eval = out["prob_dcp"].to_numpy()[outfield_mask]

    # Sanity check
    if y_dcp.shape[0] != p_dcp_eval.shape[0]:
        logging.warning("DCP metric shapes differ AFTER fix: labels=%d, preds=%d", y_dcp.shape[0], p_dcp_eval.shape[0])

    if len(np.unique(y_dcp)) > 1 and y_dcp.shape[0] == p_dcp_eval.shape[0]:
        try:
            metrics_dcp["brier"] = float(brier_score_loss(y_dcp, p_dcp_eval))
            metrics_dcp["logloss"] = float(log_loss(y_dcp, np.c_[1 - p_dcp_eval, p_dcp_eval]))
        except Exception:
            pass
        try:
            # Diagnostic MAE: observed counts vs λ_match (restrict to outfield)
            metrics_dcp["mae_lambda_proxy"] = float(mean_absolute_error(
                obs_counts_all.to_numpy()[outfield_mask],
                lam_match[outfield_mask]
            ))
        except Exception:
            pass




    (pred_dir / "metrics_cs.json").write_text(json.dumps(metrics_cs, indent=2))
    (pred_dir / "metrics_dcp.json").write_text(json.dumps(metrics_dcp, indent=2))

    logging.info("Saved models, artifacts, and eval probabilities to %s", outdir.resolve())


if __name__ == "__main__":
    main()
