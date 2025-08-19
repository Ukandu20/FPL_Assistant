#!/usr/bin/env python3
r"""
discipline_model_builder.py – v2.0

Negative FPL components:
  • Yellow (YC):  -1
  • Red (RC):     -3
  • Own goal (OG):-2
  • Missed pen:   -2
  • GC penalty:   -1 per 2 conceded (GK/DEF only)

What’s new (v2.0):
  • --predict-season [--predict-gws 1,2,3] for future-season inference (no labels required).
  • TARGET slice decoupled from TRAIN; train-only baselines used for minutes imputation.
  • Minutes-join coverage CSV (missing_minutes_join.csv).
  • Metrics only in eval mode.

Kept:
  • Robust minutes merge; no KeyErrors on optional prob_*.
  • Correct opp feature: opp_att_z_venue = opponent’s attack at their venue.
  • Team z features built from team_form if needed.
  • LGBM per-90 mean heads + optional Poisson GLM heads.
  • Minutes-aware GC penalty: for Poisson N~Pois(λ_on), E[floor(N/2)] = λ_on/2 − (1 − e^(−2λ_on))/4,
    λ_on = λ * (exp_minutes/90); applied to GK/DEF.
  • Accept defense preds as pred_ga_lambda / pred_ga_mean / cs_prob_* (λ = -log P(CS)).

Outputs:
  data/models/discipline/<model_version>/
    ├─ yc_lgbm.txt, rc_lgbm.txt, og_lgbm.txt, mpen_lgbm.txt
    ├─ *.joblib (Poisson heads, if used)
    ├─ features_used.txt
    └─ predictions/discipline_predictions.csv (+ missing_minutes_join.csv)
"""

from __future__ import annotations
import argparse, logging, re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# ------------------ Constants ------------------
KEY = ["season","gw_orig","date_played","player_id","team_id"]

# ------------------ IO helpers ------------------
def _find_seasons(features_dir: Path, version: str) -> List[str]:
    base = features_dir / version
    if not base.exists():
        raise FileNotFoundError(f"{base} does not exist")
    seasons = [d.name for d in base.iterdir() if d.is_dir() and re.match(r"^\d{4}-\d{4}$", d.name)]
    if not seasons:
        seasons = [d.name for d in base.iterdir() if (base / d.name / "players_form.csv").is_file()]
    if not seasons:
        raise FileNotFoundError(f"No season folders under {base}")
    return sorted(seasons)

def _load_players(features_dir: Path, version: str, seasons: List[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        fp = features_dir / version / s / "players_form.csv"
        if not fp.is_file():
            logging.warning("Missing %s – skipped", fp); continue
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
    # Normalize join keys
    for c in ("season","player_id","team_id"):
        df[c] = df[c].astype(str).str.lower().str.strip()
    df["date_played"] = pd.to_datetime(df["date_played"], errors="coerce").dt.date
    return df

def _load_minutes_predictions(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        logging.warning("No --minutes-preds provided; cannot scale per-90 to per-match.")
        return None
    if not path.is_file():
        logging.warning("minutes_preds not found at %s; skipping", path)
        return None
    mp = pd.read_csv(path, parse_dates=["date_played"])
    for c in ("season","player_id","team_id"):
        if c in mp.columns:
            mp[c] = mp[c].astype(str).str.lower().str.strip()
    if "date_played" in mp.columns:
        mp["date_played"] = pd.to_datetime(mp["date_played"], errors="coerce").dt.date
    need = {"season","gw_orig","date_played","player_id","team_id","pred_exp_minutes"}
    missing = need - set(mp.columns)
    if missing:
        logging.warning("minutes_preds missing columns %s – proceeding with partial merge", missing)
    for c in ("prob_played1_cal","prob_played1_raw"):
        if c not in mp.columns:
            mp[c] = np.nan
    mp = mp.sort_values(KEY).drop_duplicates(subset=KEY, keep="last")
    return mp

def _load_defense_predictions(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Return KEY + pred_ga_lambda (team GA rate λ). Accepts multiple input schemas."""
    if path is None:
        return None
    if not path.is_file():
        logging.warning("defense_preds not found at %s; GC penalty will default to 0", path)
        return None

    df = pd.read_csv(path, parse_dates=["date_played"])
    for c in ("season","player_id","team_id"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()
    if "date_played" in df.columns:
        df["date_played"] = pd.to_datetime(df["date_played"], errors="coerce").dt.date

    have = set(df.columns)
    lam = None
    if "pred_ga_lambda" in have:
        lam = pd.to_numeric(df["pred_ga_lambda"], errors="coerce")
    elif "pred_ga_mean" in have:
        lam = pd.to_numeric(df["pred_ga_mean"], errors="coerce")
    else:
        p_cs = None
        if "cs_prob_cal" in have:
            p_cs = pd.to_numeric(df["cs_prob_cal"], errors="coerce")
        elif "cs_prob_raw" in have:
            p_cs = pd.to_numeric(df["cs_prob_raw"], errors="coerce")
        if p_cs is not None:
            p = np.clip(p_cs.fillna(0.0).to_numpy(), 1e-6, 1 - 1e-6)
            lam = pd.Series(-np.log(p), index=df.index)
        else:
            logging.warning("defense_preds lacks GA fields (pred_ga_* or cs_prob_*); setting λ=0")
            lam = pd.Series(0.0, index=df.index)

    out = df[KEY].copy()
    out["pred_ga_lambda"] = pd.to_numeric(lam, errors="coerce").fillna(0.0)
    return out.drop_duplicates(subset=KEY, keep="last")

# ------------------ team z merge ------------------
def _merge_team_z(df_players: pd.DataFrame, team_form_dir: Path, version: str) -> pd.DataFrame:
    seasons = sorted(df_players["season"].unique())
    frames = []
    for s in seasons:
        fp = team_form_dir / version / s / "team_form.csv"
        if not fp.is_file():
            logging.warning("team_form missing for %s", s); continue
        tf = pd.read_csv(fp, parse_dates=["date_played"])
        tf["season"] = s
        frames.append(tf)
    if not frames:
        logging.warning("No team_form files loaded – skipping team z merge")
        df_players["team_def_z_venue"] = np.nan
        df_players["opp_att_z_venue"] = np.nan
        return df_players

    tf = pd.concat(frames, ignore_index=True)
    if {"team_def_z_venue","opp_att_z_venue"}.issubset(tf.columns):
        t = tf[["season","gw_orig","team_id","team_def_z_venue","opp_att_z_venue"]].drop_duplicates()
    else:
        needed = ["season","gw_orig","team_id","venue",
                  "def_xga_home_roll_z","def_xga_away_roll_z",
                  "att_xg_home_roll_z","att_xg_away_roll_z"]
        if not set(needed).issubset(tf.columns):
            logging.warning("team_form lacks z columns; proceeding without")
            df_players["team_def_z_venue"] = np.nan
            df_players["opp_att_z_venue"] = np.nan
            return df_players
        t = tf[needed].copy()
        t["team_def_z_venue"] = np.where(t["venue"]=="Home", t["def_xga_home_roll_z"], t["def_xga_away_roll_z"])
        t["opp_att_z_venue"] = np.where(t["venue"]=="Home", t["att_xg_away_roll_z"], t["att_xg_home_roll_z"])
        t = t.drop(columns=["venue","def_xga_home_roll_z","def_xga_away_roll_z",
                            "att_xg_home_roll_z","att_xg_away_roll_z"]).drop_duplicates()

    for c in ("team_id",):
        if c in t.columns:
            t[c] = t[c].astype(str).str.lower().str.strip()
    out = df_players.merge(t, on=["season","gw_orig","team_id"], how="left")
    return out

# ------------------ feature build ------------------
def _pick_col(df: pd.DataFrame, candidates: List[str], default: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    df[default] = 0
    return default

def _build_features(df: pd.DataFrame, use_z: bool, na_thresh: float) -> Tuple[pd.DataFrame, List[str]]:
    feats: List[str] = []
    df["venue_bin"] = (df["venue"] == "Home").astype(int); feats.append("venue_bin")
    if {"fdr_home","fdr_away"}.issubset(df.columns):
        df["fdr"] = np.where(df["venue"]=="Home", df["fdr_home"], df["fdr_away"]).astype(float); feats.append("fdr")
    if "minutes" in df.columns:
        df["prev_minutes"] = df.groupby(["player_id","season"], sort=False)["minutes"].shift(1); feats.append("prev_minutes")
    for c in ["team_def_z_venue","opp_att_z_venue"]:
        if c in df.columns: feats.append(c)
    for c in ["days_since_last","is_active"]:
        if c in df.columns: feats.append(c)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df["pos_enc"] = enc.fit_transform(df[["pos"]]); feats.append("pos_enc")
    roll_candidates = [c for c in df.columns if c.endswith(("_roll","_roll_z"))
                       and any(tok in c.lower() for tok in ["yc","rc","card","og","pen_miss","miss"])]
    keep = [c for c in roll_candidates if df[c].notna().mean() >= na_thresh]
    feats.extend(sorted(keep))
    X = df[feats].copy()
    return X, feats

# ------------------ models ------------------
def _lgbm_reg():
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=127,
        min_data_in_leaf=15,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )

def _train_poisson(Xtr: np.ndarray, ytr: np.ndarray) -> TweedieRegressor:
    m = TweedieRegressor(power=1.0, link="log", alpha=0.0005, max_iter=2000, tol=1e-6)
    m.fit(Xtr, ytr)
    return m

# ------------------ split / inference index ------------------
def _time_split_last_n(df: pd.DataFrame, test_season: str, last_n_gws: int) -> Tuple[pd.Index, pd.Index]:
    gws = sorted(df.loc[df["season"] == test_season, "gw_orig"].dropna().unique())
    if not gws:
        raise ValueError(f"No gw_orig found for season {test_season}")
    test_gws = set(gws[-last_n_gws:])
    test_idx = df.index[(df["season"] == test_season) & (df["gw_orig"].isin(test_gws))]
    train_idx = df.index.difference(test_idx)
    if len(test_idx) == 0:
        raise ValueError("Empty test split; check --test-season/--test-last-n")
    return train_idx, test_idx

def _build_inference_index(df: pd.DataFrame, season: str, gws: Optional[List[int]]) -> pd.Index:
    mask = (df["season"] == season)
    if gws is not None:
        mask &= df["gw_orig"].isin(gws)
    idx = df.index[mask]
    if len(idx) == 0:
        raise ValueError(f"No rows available to predict for season={season} gws={gws}")
    return idx

# ------------------ misc ------------------
def _write_missing_join_csv(out_pred_dir: Path, frame: pd.DataFrame, miss_mask: pd.Series) -> None:
    """Write rows (from `frame`) where minutes were missing, using positional boolean mask."""
    if isinstance(miss_mask, pd.Series):
        mask_np = miss_mask.to_numpy()
    else:
        mask_np = np.asarray(miss_mask, dtype=bool)
    if mask_np.any():
        miss = frame.iloc[mask_np].copy()
        (out_pred_dir / "missing_minutes_join.csv").parent.mkdir(parents=True, exist_ok=True)
        miss.to_csv(out_pred_dir / "missing_minutes_join.csv", index=False)

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--version", required=True)
    ap.add_argument("--team-form-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--test-season", required=True)
    ap.add_argument("--test-last-n", type=int, default=10)

    ap.add_argument("--predict-season", type=str, default=None,
                    help="If set, run inference for this season (all GWs or --predict-gws).")
    ap.add_argument("--predict-gws", type=str, default=None,
                    help="Comma-separated GW list for inference, e.g., '1,2,3'.")

    ap.add_argument("--use-z", action="store_true")
    ap.add_argument("--na-thresh", type=float, default=0.70)
    ap.add_argument("--minutes-preds", type=Path, required=True)
    ap.add_argument("--defense-preds", type=Path, help="optional; to compute GC penalty via GA λ or CS probability")
    ap.add_argument("--models-out", type=Path, default=Path("data/models/discipline"))
    ap.add_argument("--model-version", default="v2")
    ap.add_argument("--poisson-heads", action="store_true")
    ap.add_argument("--drop-missing-minutes", action="store_true",
                    help="Drop TARGET rows without pred_exp_minutes (default False; we impute instead)",
                    default=False)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    seasons = _find_seasons(args.features_dir, args.version)
    logging.info("Found seasons: %s", ", ".join(seasons))

    # Load & merge
    df = _load_players(args.features_dir, args.version, seasons)
    df = _merge_team_z(df, args.team_form_dir, args.version)

    # Targets (create zero columns if missing)
    col_yc  = _pick_col(df, ["yc","yellow_cards","yellow"], "yc")
    col_rc  = _pick_col(df, ["rc","red_cards","red"], "rc")
    col_og  = _pick_col(df, ["og","own_goals"], "og")
    col_mp  = _pick_col(df, ["pen_miss","pens_missed","missed_pens","pens_miss"], "pen_miss")

    # Compute per-90 targets from counts
    m = df["minutes"].fillna(0).clip(lower=0)
    m90 = (m / 90.0).replace(0, np.nan)
    df["y_yc_p90"]  = (pd.to_numeric(df[col_yc], errors="coerce").fillna(0) / m90).fillna(0)
    df["y_rc_p90"]  = (pd.to_numeric(df[col_rc], errors="coerce").fillna(0) / m90).fillna(0)
    df["y_og_p90"]  = (pd.to_numeric(df[col_og], errors="coerce").fillna(0) / m90).fillna(0)
    df["y_mp_p90"]  = (pd.to_numeric(df[col_mp], errors="coerce").fillna(0) / m90).fillna(0)

    # Build features
    df = df.sort_values(["season","date_played","gw_orig","player_id"]).reset_index(drop=True)
    X, feat_cols = _build_features(df, use_z=args.use_z, na_thresh=args.na_thresh)

    # Choose TRAIN & TARGET
    if args.predict_season:
        train_idx, _ = _time_split_last_n(df, args.test_season, args.test_last_n)  # consistent with other heads
        pred_gws = None if not args.predict_gws else [int(x) for x in args.predict_gws.split(",")]
        target_idx = _build_inference_index(df, args.predict_season, pred_gws)
        logging.info("Mode: PREDICT-ONLY for season %s%s",
                     args.predict_season,
                     f" GWs {args.predict_gws}" if args.predict_gws else " (all GWs)")
    else:
        train_idx, target_idx = _time_split_last_n(df, args.test_season, args.test_last_n)
        logging.info("Mode: EVAL (last %d GWs of %s)", args.test_last_n, args.test_season)

    # TRAIN (minutes>0)
    m_tr = df.loc[train_idx,"minutes"].fillna(0).clip(lower=0)
    tr_mask = (m_tr > 0)
    if tr_mask.sum() == 0:
        raise ValueError("No valid training rows (minutes>0).")
    Xtr = X.loc[train_idx].iloc[tr_mask.values].copy()
    y_yc_tr = df.loc[train_idx,"y_yc_p90"].iloc[tr_mask.values].to_numpy()
    y_rc_tr = df.loc[train_idx,"y_rc_p90"].iloc[tr_mask.values].to_numpy()
    y_og_tr = df.loc[train_idx,"y_og_p90"].iloc[tr_mask.values].to_numpy()
    y_mp_tr = df.loc[train_idx,"y_mp_p90"].iloc[tr_mask.values].to_numpy()

    # TARGET base keys
    base = df.loc[target_idx, KEY + ["player","pos","venue","prev_minutes"]].reset_index(drop=True).copy()

    # Minutes merge + hierarchical imputation (train-only baselines)
    mp = _load_minutes_predictions(args.minutes_preds)
    if mp is None:
        raise ValueError("--minutes-preds is required; cannot scale per-90 to per-match")

    mp_key = mp[KEY + ["pred_exp_minutes","prob_played1_cal","prob_played1_raw"]].drop_duplicates(subset=KEY, keep="last").copy()
    merged = base.merge(mp_key, on=KEY, how="left")

    # Train-only baselines
    train_df = df.loc[train_idx, ["season","team_id","pos","minutes","prev_minutes"]].copy()
    train_df["played1"] = (train_df["minutes"].fillna(0) > 0).astype(float)

    med_team_pos   = train_df.groupby(["season","team_id","pos"])["minutes"].median().rename("med_min_team_pos")
    med_pos_season = train_df.groupby(["season","pos"])["minutes"].median().rename("med_min_pos_season")
    med_pos_global = train_df.groupby(["pos"])["minutes"].median().rename("med_min_pos_global")
    med_global     = float(train_df["minutes"].median())

    rate_team_pos   = train_df.groupby(["season","team_id","pos"])["played1"].mean().rename("p1_rate_team_pos")
    rate_pos_season = train_df.groupby(["season","pos"])["played1"].mean().rename("p1_rate_pos_season")
    rate_pos_global = train_df.groupby(["pos"])["played1"].mean().rename("p1_rate_pos_global")
    rate_global     = float(train_df["played1"].mean())

    aux = base[["season","team_id","pos"]].copy()
    aux = aux.merge(med_team_pos.reset_index(), on=["season","team_id","pos"], how="left")
    aux = aux.merge(med_pos_season.reset_index(), on=["season","pos"], how="left")
    aux = aux.merge(med_pos_global.reset_index(), on=["pos"], how="left")
    aux = aux.merge(rate_team_pos.reset_index(), on=["season","team_id","pos"], how="left")
    aux = aux.merge(rate_pos_season.reset_index(), on=["season","pos"], how="left")
    aux = aux.merge(rate_pos_global.reset_index(), on=["pos"], how="left")

    prev_minutes_target = base["prev_minutes"].copy()

    p1_est = merged["prob_played1_cal"].copy()
    p1_est = p1_est.fillna(merged["prob_played1_raw"])
    p1_est = p1_est.fillna(aux["p1_rate_team_pos"])
    p1_est = p1_est.fillna(aux["p1_rate_pos_season"])
    p1_est = p1_est.fillna(aux["p1_rate_pos_global"])
    p1_est = p1_est.fillna((prev_minutes_target.fillna(0) > 0).astype(float))
    p1_est = p1_est.fillna(rate_global)

    med_base = aux["med_min_team_pos"].copy()
    med_base = med_base.fillna(aux["med_min_pos_season"])
    med_base = med_base.fillna(aux["med_min_pos_global"])
    med_base = med_base.fillna(med_global)

    miss_minutes = merged["pred_exp_minutes"].isna()
    imputed_exp_minutes = (p1_est * med_base).astype(float)
    merged.loc[miss_minutes, "pred_exp_minutes"] = imputed_exp_minutes[miss_minutes]
    if int(miss_minutes.sum()):
        logging.info("Imputed exp_minutes for %d/%d TARGET rows (train-only baselines).",
                     int(miss_minutes.sum()), len(merged))

    # Final p1 for dump (diagnostic)
    p1 = merged["prob_played1_cal"].copy()
    p1 = p1.fillna(merged["prob_played1_raw"])
    p1 = p1.fillna(aux["p1_rate_team_pos"])
    p1 = p1.fillna(aux["p1_rate_pos_season"])
    p1 = p1.fillna(aux["p1_rate_pos_global"])
    p1 = p1.fillna((prev_minutes_target.fillna(0) > 0).astype(float))
    p1 = p1.fillna((merged["pred_exp_minutes"].fillna(0) > 0).astype(float))
    p1 = p1.fillna(rate_global)

    exp_mins = merged["pred_exp_minutes"].fillna(0.0).to_numpy()
    scale = exp_mins / 90.0

    # Train models
    mdl_yc = _lgbm_reg().fit(Xtr, y_yc_tr)
    mdl_rc = _lgbm_reg().fit(Xtr, y_rc_tr)
    mdl_og = _lgbm_reg().fit(Xtr, y_og_tr)
    mdl_mp = _lgbm_reg().fit(Xtr, y_mp_tr)

    Xte = X.loc[target_idx].copy()
    yc_p90_mean = np.clip(mdl_yc.predict(Xte), 0, None)
    rc_p90_mean = np.clip(mdl_rc.predict(Xte), 0, None)
    og_p90_mean = np.clip(mdl_og.predict(Xte), 0, None)
    mp_p90_mean = np.clip(mdl_mp.predict(Xte), 0, None)

    yc_mean = yc_p90_mean * scale
    rc_mean = rc_p90_mean * scale
    og_mean = og_p90_mean * scale
    mp_mean = mp_p90_mean * scale

    # Optional Poisson heads
    if args.poisson_heads:
        pois_yc = _train_poisson(Xtr.to_numpy(), y_yc_tr)
        pois_rc = _train_poisson(Xtr.to_numpy(), y_rc_tr)
        pois_og = _train_poisson(Xtr.to_numpy(), y_og_tr)
        pois_mp = _train_poisson(Xtr.to_numpy(), y_mp_tr)

        yc_p90_pois = np.clip(pois_yc.predict(Xte.to_numpy()), 0, None)
        rc_p90_pois = np.clip(pois_rc.predict(Xte.to_numpy()), 0, None)
        og_p90_pois = np.clip(pois_og.predict(Xte.to_numpy()), 0, None)
        mp_p90_pois = np.clip(pois_mp.predict(Xte.to_numpy()), 0, None)

        yc_pois = yc_p90_pois * scale
        rc_pois = rc_p90_pois * scale
        og_pois = og_p90_pois * scale
        mp_pois = mp_p90_pois * scale
    else:
        pois_yc = pois_rc = pois_og = pois_mp = None
        yc_p90_pois = rc_p90_pois = og_p90_pois = mp_p90_pois = None
        yc_pois = rc_pois = og_pois = mp_pois = None

    # Defense preds for GA (optional) → derive λ
    dfx = _load_defense_predictions(args.defense_preds)
    if dfx is not None:
        merged = merged.merge(dfx, on=KEY, how="left")
    else:
        merged["pred_ga_lambda"] = 0.0

    # GC penalty (minutes-aware; GK/DEF only)
    ga_lambda = pd.to_numeric(merged["pred_ga_lambda"], errors="coerce").fillna(0.0).to_numpy()
    minutes_share = np.clip(exp_mins / 90.0, 0.0, 1.0)
    lam_on = ga_lambda * minutes_share
    e_floor_half = 0.5 * lam_on - 0.25 * (1.0 - np.exp(-2.0 * lam_on))

    pos = base["pos"].reset_index(drop=True)
    is_def_like = pos.isin(["GK","DEF"]).to_numpy()
    gc_pen_points = np.where(is_def_like, -e_floor_half, 0.0)

    # Component points
    pts_yc  = -1.0 * yc_mean
    pts_rc  = -3.0 * rc_mean
    pts_og  = -2.0 * og_mean
    pts_mp  = -2.0 * mp_mean
    total_neg_points = pts_yc + pts_rc + pts_og + pts_mp + gc_pen_points

    # Eval metrics (counts) — only if eval mode and ground truth present
    if not args.predict_season:
        try:
            y_true_yc = pd.to_numeric(df.loc[target_idx, col_yc], errors="coerce").fillna(0).to_numpy()
            y_true_rc = pd.to_numeric(df.loc[target_idx, col_rc], errors="coerce").fillna(0).to_numpy()
            y_true_og = pd.to_numeric(df.loc[target_idx, col_og], errors="coerce").fillna(0).to_numpy()
            y_true_mp = pd.to_numeric(df.loc[target_idx, col_mp], errors="coerce").fillna(0).to_numpy()
            logging.info("Test MAE (YC count): %.4f", mean_absolute_error(y_true_yc, yc_mean))
            logging.info("Test MAE (RC count): %.4f", mean_absolute_error(y_true_rc, rc_mean))
            logging.info("Test MAE (OG count): %.4f", mean_absolute_error(y_true_og, og_mean))
            logging.info("Test MAE (MP count): %.4f", mean_absolute_error(y_true_mp, mp_mean))
        except Exception:
            pass

    # Dump predictions
    dump = base.drop(columns=["prev_minutes"]).copy()
    dump["pred_exp_minutes"] = exp_mins
    dump["p1"] = p1.to_numpy()
    dump["pred_ga_lambda"] = ga_lambda

    dump["pred_yc_p90_mean"] = yc_p90_mean
    dump["pred_rc_p90_mean"] = rc_p90_mean
    dump["pred_og_p90_mean"] = og_p90_mean
    dump["pred_mpen_p90_mean"] = mp_p90_mean

    dump["pred_yc_mean"] = yc_mean
    dump["pred_rc_mean"] = rc_mean
    dump["pred_og_mean"] = og_mean
    dump["pred_mpen_mean"] = mp_mean

    if args.poisson_heads:
        dump["pred_yc_p90_poisson"] = yc_p90_pois
        dump["pred_rc_p90_poisson"] = rc_p90_pois
        dump["pred_og_p90_poisson"] = og_p90_pois
        dump["pred_mpen_p90_poisson"] = mp_p90_pois
        dump["pred_yc_poisson"] = yc_pois
        dump["pred_rc_poisson"] = rc_pois
        dump["pred_og_poisson"] = og_pois
        dump["pred_mpen_poisson"] = mp_pois

    dump["pts_yc"]   = pts_yc
    dump["pts_rc"]   = pts_rc
    dump["pts_og"]   = pts_og
    dump["pts_mpen"] = pts_mp
    dump["pts_gc_pen"] = gc_pen_points
    dump["neg_points_total"] = total_neg_points

    # Save models + predictions
    outdir = args.models_out / args.model_version
    (outdir / "predictions").mkdir(parents=True, exist_ok=True)

    mdl_yc.booster_.save_model(outdir / "yc_lgbm.txt")
    mdl_rc.booster_.save_model(outdir / "rc_lgbm.txt")
    mdl_og.booster_.save_model(outdir / "og_lgbm.txt")
    mdl_mp.booster_.save_model(outdir / "mpen_lgbm.txt")

    if args.poisson_heads:
        joblib.dump(pois_yc, outdir / "yc_poisson.joblib")
        joblib.dump(pois_rc, outdir / "rc_poisson.joblib")
        joblib.dump(pois_og, outdir / "og_poisson.joblib")
        joblib.dump(pois_mp, outdir / "mpen_poisson.joblib")

    # Minutes-join audit (only if we actually merged)
    if "pred_exp_minutes" in merged.columns:
        _write_missing_join_csv(outdir / "predictions", merged, merged["pred_exp_minutes"].isna())

    fp = outdir / "predictions" / "discipline_predictions.csv"
    dump.to_csv(fp, index=False)
    logging.info("Wrote predictions to %s", fp.resolve())
    (outdir / "features_used.txt").write_text("\n".join(feat_cols), encoding="utf-8")
    logging.info("Models & predictions saved to %s", outdir.resolve())

if __name__ == "__main__":
    main()
