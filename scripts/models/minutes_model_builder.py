#minutes_model_builder.py

from __future__ import annotations
import argparse, logging, re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss
try:
    from sklearn.metrics import mean_pinball_loss  # sklearn >=0.24
except Exception:
    mean_pinball_loss = None
from sklearn.preprocessing import OrdinalEncoder
from sklearn.isotonic import IsotonicRegression
import joblib

# -------------- IO helpers --------------
def _find_seasons(features_dir: Path, version: str) -> List[str]:
    base = features_dir / version
    if not base.exists():
        raise FileNotFoundError(f"{base} does not exist")
    dirs = [d for d in base.iterdir() if d.is_dir()]
    seasons = [d.name for d in dirs if re.match(r"^\d{4}-\d{4}$", d.name)]
    if not seasons:
        seasons = [d.name for d in dirs if (d / "players_form.csv").is_file()]
    if not seasons:
        found = ", ".join(d.name for d in dirs) or "(none)"
        raise FileNotFoundError(f"No season folders under {base}. Found subfolders: {found}")
    return sorted(seasons)

def _prep_calendar_for_join(cal: pd.DataFrame) -> pd.DataFrame:
    if "date_played" in cal.columns:
        cal["date_played"] = pd.to_datetime(cal["date_played"], errors="coerce")
    if "venue" not in cal.columns:
        for c in ["home", "was_home", "is_home"]:
            if c in cal.columns:
                cal["venue"] = np.where(cal[c].astype(bool), "Home", "Away")
                break
    if "venue" not in cal.columns:
        cal["venue"] = pd.NA
    if "is_starter" in cal.columns:
        cal["is_starter"] = cal["is_starter"].astype(float)
        cal.loc[cal["is_starter"].notna(), "is_starter"] = (cal.loc[cal["is_starter"].notna(), "is_starter"] > 0).astype(int)
        cal["is_starter"] = cal["is_starter"].astype("Int64")
    return cal

def _attach_is_starter(form_df: pd.DataFrame, fixtures_dir: Path, season: str) -> pd.DataFrame:
    cal_path = fixtures_dir / season / "player_minutes_calendar.csv"
    if not cal_path.is_file():
        logging.warning("No calendar file found for %s at %s; proceeding without is_starter.", season, cal_path)
        form_df["is_starter"] = pd.Series(pd.NA, index=form_df.index, dtype="Int64")
        return form_df
    try:
        cal = pd.read_csv(cal_path, parse_dates=["date_played"], low_memory=False)
    except Exception:
        cal = pd.read_csv(cal_path, low_memory=False)
    cal = _prep_calendar_for_join(cal)

    join_cols_candidates = []
    if all(c in form_df.columns for c in ["player_id", "date_played", "gw_orig"]):
        join_cols_candidates.append(["player_id", "date_played", "gw_orig"])
    if all(c in form_df.columns for c in ["player_id", "date_played", "venue"]):
        join_cols_candidates.append(["player_id", "date_played", "venue"])

    merged = None
    for keys in join_cols_candidates:
        if not all(k in cal.columns for k in keys):
            continue
        cal_dedup = cal.sort_values(keys).drop_duplicates(subset=keys, keep="first")
        use_cols = [c for c in keys if c in cal_dedup.columns] + [c for c in ["is_starter"] if c in cal_dedup.columns]
        cal_small = cal_dedup[use_cols].copy()
        if "date_played" in keys:
            form_df["date_played"] = pd.to_datetime(form_df["date_played"], errors="coerce")
            cal_small["date_played"] = pd.to_datetime(cal_small["date_played"], errors="coerce")
        tmp = form_df.merge(cal_small, how="left", on=keys, validate="m:1")
        merged = tmp
        break

    if merged is None:
        logging.warning("Could not find a reliable key to merge is_starter for %s; proceeding without.", season)
        form_df["is_starter"] = pd.Series(pd.NA, index=form_df.index, dtype="Int64")
        return form_df

    if "is_starter" in merged.columns:
        merged["is_starter"] = merged["is_starter"].astype("Int64")
    else:
        merged["is_starter"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    return merged

def _load_all(features_dir: Path, version: str, seasons: List[str], fixtures_dir: Path) -> pd.DataFrame:
    frames = []
    for s in seasons:
        fp = features_dir / version / s / "players_form.csv"
        if not fp.is_file():
            logging.warning("Missing %s – skipped", fp)
            continue
        df = pd.read_csv(fp, parse_dates=["date_played"])
        df["season"] = s
        df = _attach_is_starter(df, fixtures_dir=fixtures_dir, season=s)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No season files found")
    df = pd.concat(frames, ignore_index=True)
    need = {
        "player_id","season","gw_orig","date_played",
        "minutes","venue","pos","days_since_last","is_active",
        "fdr_home","fdr_away","team_id","player"
    }
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"Features missing required columns: {miss}")
    return df

# -------------- features --------------
def _build_dataset(df: pd.DataFrame, use_z: bool, na_thresh: float):
    """
    Returns:
      X (DataFrame), y_min (Series), y_p1 (Series), y_p60 (Series),
      feat_cols (List[str]), df_feat (DataFrame)
    """
    df = df.sort_values(["player_id","season","date_played","gw_orig"]).copy()

    # venue-aware FDR
    df["fdr"] = np.where(df["venue"] == "Home", df["fdr_home"], df["fdr_away"]).astype(float)

    # ----- lags -----
    # within-season lag (strict, used for TRAIN)
    df["prev_minutes"] = df.groupby(["player_id","season"], sort=False)["minutes"].shift(1)

    if "is_starter" in df.columns:
        df["prev_is_starter"] = (
            df.groupby(["player_id","season"], sort=False)["is_starter"]
              .shift(1)
              .astype("Float64")
        )
    else:
        df["prev_is_starter"] = np.nan

    # cross-season global lag (for PREDICT cold-start fallback)
    df = df.sort_values(["player_id","date_played","gw_orig"]).copy()
    df["prev_minutes_global"] = df.groupby("player_id", sort=False)["minutes"].shift(1)
    if "is_starter" in df.columns:
        df["prev_is_starter_global"] = (
            df.groupby("player_id", sort=False)["is_starter"].shift(1).astype("Float64")
        )
    else:
        df["prev_is_starter_global"] = np.nan

    # targets
    df["y_minutes"]  = df["minutes"].astype(float)
    df["y_played1"]  = (df["minutes"] >= 1).astype(int)
    df["y_played60"] = (df["minutes"] >= 60).astype(int)

    # base numeric features
    feats = ["prev_minutes", "prev_is_starter", "days_since_last", "is_active", "fdr"]

    # categorical encodings
    df["venue_bin"] = df["venue"].astype(str).str.lower().eq("home").astype(int); feats.append("venue_bin")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df["pos_enc"] = enc.fit_transform(df[["pos"]]); feats.append("pos_enc")

    # rolling features
    roll_cols = [c for c in df.columns if c.endswith(("_roll", "_home_roll", "_away_roll")) and not c.endswith("_roll_z")]
    if use_z:
        roll_cols += [c for c in df.columns if c.endswith(("_roll_z", "_home_roll_z", "_away_roll_z"))]
    keep = [c for c in roll_cols if df[c].notna().mean() >= na_thresh]
    feats.extend(sorted(keep))

    # TRAIN will require prev_minutes later; keep all rows for now (needed for predict cold-start)
    X = df[feats].copy()
    y_min  = df["y_minutes"].copy()
    y_p1   = df["y_played1"].copy()
    y_p60  = df["y_played60"].copy()
    return X, y_min, y_p1, y_p60, feats, df

# -------------- time split / inference index --------------
def _time_split_last_n(df_feat: pd.DataFrame, test_season: str, last_n_gws: int) -> Tuple[pd.Index, pd.Index]:
    gws = sorted(df_feat.loc[df_feat["season"] == test_season, "gw_orig"].dropna().unique())
    if not gws:
        raise ValueError(f"No gw_orig found for season {test_season}")
    test_gws = set(gws[-last_n_gws:])
    test_idx = df_feat.index[(df_feat["season"] == test_season) & (df_feat["gw_orig"].isin(test_gws))]
    train_idx = df_feat.index.difference(test_idx)
    if len(test_idx) == 0:
        raise ValueError("Test split produced 0 rows. Check --test-season and --test-last-n.")
    return train_idx, test_idx

def _build_inference_index(df_feat: pd.DataFrame, season: str, gws_csv: Optional[str]) -> pd.Index:
    mask = (df_feat["season"] == season)
    if gws_csv:
        want = [int(x) for x in gws_csv.split(",") if x.strip() != ""]
        mask &= df_feat["gw_orig"].isin(want)
    idx = df_feat.index[mask]
    if len(idx) == 0:
        raise ValueError(f"No rows available to predict for season={season}, gws={gws_csv or 'ALL'}")
    return idx

# -------------- recency weights --------------
def _season_recency_weights(seasons_sorted: List[str], max_weight: float) -> Dict[str, float]:
    n = len(seasons_sorted)
    if n == 1:
        return {seasons_sorted[0]: 1.0}
    return {s: 1.0 + (max_weight - 1.0) * (i / (n - 1)) for i, s in enumerate(seasons_sorted)}

# -------------- monotone constraints --------------
def _monotone_vector(feat_cols: List[str], enable: bool) -> Optional[List[int]]:
    if not enable:
        return None
    vec = []
    for c in feat_cols:
        if c in ("prev_minutes", "is_active", "prev_is_starter"):
            vec.append(1)
        else:
            vec.append(0)
    return vec

# -------------- models --------------
def _mk_regressor(monotone: Optional[List[int]]) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=4000,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        min_data_in_leaf=10,
        min_sum_hessian_in_leaf=1e-3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=-1,
        verbosity=-1,
        random_state=42,
        monotone_constraints=monotone if monotone is not None else None,
    )

def _mk_classifier(monotone: Optional[List[int]]) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=4000,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        min_data_in_leaf=10,
        min_sum_hessian_in_leaf=1e-3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=-1,
        verbosity=-1,
        random_state=42,
        monotone_constraints=monotone if monotone is not None else None,
    )

def _mk_quantile_regressor(alpha: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=4000,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        min_data_in_leaf=10,
        min_sum_hessian_in_leaf=1e-3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=-1,
        verbosity=-1,
        random_state=42,
    )

def _fit_isotonic(y_true: np.ndarray, proba_raw: np.ndarray, sample_weight: Optional[np.ndarray]) -> Optional[IsotonicRegression]:
    if len(np.unique(y_true)) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_raw, y_true, sample_weight=sample_weight)
    return iso

def _safe_auc(y, p) -> Optional[float]:
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else None

def _pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    if mean_pinball_loss is not None:
        return float(mean_pinball_loss(y_true, y_pred, alpha=q))
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))

# -------------- main --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--fixtures-dir", type=Path, default=Path("data/processed/fixtures"),
                    help="Root dir containing <SEASON>/player_minutes_calendar.csv for is_starter merge")
    ap.add_argument("--version", required=True, help="features version folder, e.g., v7")

    ap.add_argument("--test-season", required=True, help="e.g., 2024-2025")
    ap.add_argument("--test-last-n", type=int, default=10, help="number of last GWs in test-season")

    # NEW: predict (future-season) mode
    ap.add_argument("--predict-season", type=str, default=None,
                    help="If set, produce predictions for this season (no labels needed).")
    ap.add_argument("--predict-gws", type=str, default=None,
                    help="Comma-separated GW list for inference, e.g., '1,2,3'. Omit for all.")

    ap.add_argument("--use-z", action="store_true", help="include *_roll_z features as well")
    ap.add_argument("--na-thresh", type=float, default=0.70, help="drop feature columns with < coverage")
    ap.add_argument("--calibrate", action="store_true", help="fit isotonic calibration on played1/played60")
    ap.add_argument("--cal-frac", type=float, default=0.2, help="fraction of TRAIN rows as chrono tail for calibration")
    ap.add_argument("--monotone", action="store_true", help="enable monotone constraints on prev_minutes/is_active/prev_is_starter")
    ap.add_argument("--recency-weight-max", type=float, default=1.5, help="newest season weight (oldest fixed at 1.0)")
    ap.add_argument("--quantiles", default="0.1,0.5,0.9", help="comma list, e.g. 0.1,0.5,0.9")
    ap.add_argument("--models-out", type=Path, default=Path("data/models/expected_minutes"))
    ap.add_argument("--model-version", default="v2", help="models subdir name, e.g., v3")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    quants = [float(x) for x in args.quantiles.split(",") if x.strip() != ""]
    quants = sorted([q for q in quants if 0.0 < q < 1.0])
    if not quants:
        raise ValueError("No valid quantiles provided via --quantiles")

    seasons = _find_seasons(args.features_dir, args.version)
    logging.info("Found seasons: %s", ", ".join(seasons))

    df = _load_all(args.features_dir, args.version, seasons, fixtures_dir=args.fixtures_dir)
    X, y_min, y_p1, y_p60, feat_cols, df_feat = _build_dataset(df, use_z=args.use_z, na_thresh=args.na_thresh)

    # ------- choose TRAIN vs TARGET -------
    if args.predict_season:
        train_idx, _ = _time_split_last_n(df_feat, args.test_season, args.test_last_n)  # train on history
        target_idx = _build_inference_index(df_feat, args.predict_season, args.predict_gws)
        logging.info("Mode: PREDICT-ONLY for %s%s", args.predict_season,
                     f" (GWs {args.predict_gws})" if args.predict_gws else " (all GWs)")
    else:
        train_idx, target_idx = _time_split_last_n(df_feat, args.test_season, args.test_last_n)
        logging.info("Mode: EVAL (last %d GWs of %s)", args.test_last_n, args.test_season)

    # ------- TRAIN set hygiene (require prev_minutes) -------
    # drop rows with missing prev_minutes on TRAIN only (original behavior)
    tr_req = df_feat.loc[train_idx].copy()
    tr_req = tr_req[tr_req["prev_minutes"].notna()]
    if tr_req.empty:
        raise ValueError("No rows left after requiring prev_minutes on TRAIN.")
    Xtr = X.loc[tr_req.index]
    ytr_min = y_min.loc[tr_req.index]
    ytr_p1  = y_p1.loc[tr_req.index]
    ytr_p60 = y_p60.loc[tr_req.index]

    # ------- TARGET slice (keep all; cold-start impute later) -------
    Xte = X.loc[target_idx].copy()

    # class labels for EVAL only
    yte_min = y_min.loc[target_idx] if not args.predict_season else None
    yte_p1  = y_p1.loc[target_idx]  if not args.predict_season else None
    yte_p60 = y_p60.loc[target_idx] if not args.predict_season else None

    # ------- recency weights -------
    season_weights = _season_recency_weights(seasons, args.recency_weight_max)
    df_feat["recency_w"] = df_feat["season"].map(season_weights).astype(float)
    w_train = df_feat.loc[Xtr.index, "recency_w"].to_numpy()

    # ------- monotone vector -------
    mono_vec = _monotone_vector(feat_cols, enable=args.monotone)

    # ------- mean models -------
    logging.info("Training expected minutes regressor (GLOBAL) …")
    reg_global = _mk_regressor(mono_vec)
    reg_global.fit(Xtr, ytr_min, sample_weight=w_train)

    logging.info("Training per-position minutes regressors …")
    regs_by_pos: Dict[str, lgb.LGBMRegressor] = {}
    for pos in ["GK","DEF","MID","FWD"]:
        pos_mask_tr = (df_feat.loc[Xtr.index, "pos"] == pos).to_numpy()
        if pos_mask_tr.sum() < 100:
            continue
        reg_pos = _mk_regressor(mono_vec)
        reg_pos.fit(Xtr[pos_mask_tr], ytr_min[pos_mask_tr], sample_weight=w_train[pos_mask_tr])
        regs_by_pos[pos] = reg_pos

    # ------- quantile models -------
    logging.info("Training quantile minutes regressors …")
    q_global: Dict[float, lgb.LGBMRegressor] = {}
    q_by_pos: Dict[float, Dict[str, lgb.LGBMRegressor]] = {}
    for q in quants:
        m = _mk_quantile_regressor(q)
        m.fit(Xtr, ytr_min, sample_weight=w_train)
        q_global[q] = m
        q_by_pos[q] = {}
        for pos in ["GK","DEF","MID","FWD"]:
            pos_mask_tr = (df_feat.loc[Xtr.index, "pos"] == pos).to_numpy()
            if pos_mask_tr.sum() < 100:
                continue
            mpos = _mk_quantile_regressor(q)
            mpos.fit(Xtr[pos_mask_tr], ytr_min[pos_mask_tr], sample_weight=w_train[pos_mask_tr])
            q_by_pos[q][pos] = mpos

    # ------- classifiers -------
    p1_single = (len(np.unique(ytr_p1)) < 2)
    if p1_single:
        logging.info("played1 has a single class -> skipping training; setting p1=1.0 on TARGET")
        clf_p1 = None
    else:
        logging.info("Training played1 classifier …")
        clf_p1 = _mk_classifier(mono_vec)
        clf_p1.fit(Xtr, ytr_p1, sample_weight=w_train)

    logging.info("Training played60 classifier …")
    clf_p60 = _mk_classifier(mono_vec)
    clf_p60.fit(Xtr, ytr_p60, sample_weight=w_train)

    # ------- cold-start imputations on TARGET (predict & eval) -------
    # prev_minutes: fill NaN with cross-season last, then train pos median, then global median
    pm = df_feat.loc[target_idx, "prev_minutes"].copy()
    pm_glob = df_feat.loc[target_idx, "prev_minutes_global"].copy()

    train_pos_median = df_feat.loc[Xtr.index].groupby("pos")["prev_minutes"].median()
    train_global_median = float(df_feat.loc[Xtr.index, "prev_minutes"].median())

    fill_prev = pm.copy()
    na = fill_prev.isna()
    fill_prev.loc[na] = pm_glob.loc[na]
    na = fill_prev.isna()
    if na.any():
        pos_series = df_feat.loc[target_idx, "pos"]
        fill_prev.loc[na] = pos_series.loc[na].map(train_pos_median).astype(float)
    na = fill_prev.isna()
    if na.any():
        fill_prev.loc[na] = train_global_median

    Xte.loc[:, "prev_minutes"] = Xte["prev_minutes"].fillna(fill_prev)

    # prev_is_starter: fill NaN with train pos mean, else 0
    pis = df_feat.loc[target_idx, "prev_is_starter"].copy()
    train_pos_mean_pis = df_feat.loc[Xtr.index, "prev_is_starter"].groupby(df_feat.loc[Xtr.index, "pos"]).mean()
    pis = pis.fillna(df_feat.loc[target_idx, "pos"].map(train_pos_mean_pis))
    pis = pis.fillna(0.0)
    Xte.loc[:, "prev_is_starter"] = Xte["prev_is_starter"].fillna(pis)

    # ------- predictions (TARGET) -------
    pos_te = df_feat.loc[target_idx, "pos"]
    pred_min_mean = np.empty(len(target_idx), dtype=float)
    for i, (ix, p) in enumerate(zip(target_idx, pos_te)):
        model = regs_by_pos.get(p, reg_global)
        pred_min_mean[i] = model.predict(X.loc[[ix]].assign(
            prev_minutes=Xte.loc[ix, "prev_minutes"],
            prev_is_starter=Xte.loc[ix, "prev_is_starter"]
        ))[0]
    pred_min_mean = np.clip(pred_min_mean, 0, 120)

    pred_quants: Dict[float, np.ndarray] = {q: np.empty(len(target_idx), dtype=float) for q in quants}
    for q in quants:
        for i, (ix, p) in enumerate(zip(target_idx, pos_te)):
            model = q_by_pos[q].get(p, q_global[q])
            pred_quants[q][i] = model.predict(X.loc[[ix]].assign(
                prev_minutes=Xte.loc[ix, "prev_minutes"],
                prev_is_starter=Xte.loc[ix, "prev_is_starter"]
            ))[0]
        pred_quants[q] = np.clip(pred_quants[q], 0, 120)

    if p1_single:
        proba_p1 = np.ones(len(target_idx), dtype=float)
    else:
        proba_p1 = clf_p1.predict_proba(Xte)[:, 1]
    proba_p60 = clf_p60.predict_proba(Xte)[:, 1]

    # -------- isotonic calibration (optional) --------
    iso_p1: Optional[IsotonicRegression] = None
    iso_p60: Optional[IsotonicRegression] = None
    proba_p1_cal = None
    proba_p60_cal = None

    if args.calibrate:
        logging.info("Fitting isotonic calibration (chronological tail of training) …")
        tr_ordered = df_feat.loc[Xtr.index].sort_values(["season","date_played","gw_orig"])
        fit_idx, cal_idx = _select_calibration_tail(tr_ordered, args.cal_frac, args.test_season)

        w_fit = df_feat.loc[fit_idx, "recency_w"].to_numpy()
        w_cal = df_feat.loc[cal_idx, "recency_w"].to_numpy()

        if not p1_single:
            clf_p1 = _mk_classifier(mono_vec); clf_p1.fit(X.loc[fit_idx], y_p1.loc[fit_idx], sample_weight=w_fit)
        clf_p60 = _mk_classifier(mono_vec);   clf_p60.fit(X.loc[fit_idx], y_p60.loc[fit_idx], sample_weight=w_fit)

        if not p1_single:
            p1_raw_cal = clf_p1.predict_proba(X.loc[cal_idx])[:, 1]
            iso_p1 = _fit_isotonic(y_p1.loc[cal_idx].to_numpy(), p1_raw_cal, w_cal)

        p60_raw_cal = clf_p60.predict_proba(X.loc[cal_idx])[:, 1]
        iso_p60 = _fit_isotonic(y_p60.loc[cal_idx].to_numpy(), p60_raw_cal, w_cal)

        if not p1_single:
            proba_p1 = clf_p1.predict_proba(Xte)[:, 1]
            proba_p1_cal = iso_p1.predict(proba_p1) if iso_p1 is not None else None
        proba_p60 = clf_p60.predict_proba(Xte)[:, 1]
        proba_p60_cal = iso_p60.predict(proba_p60) if iso_p60 is not None else None

    # -------- metrics (eval only) --------
    if not args.predict_season:
        mae_mean = mean_absolute_error(yte_min, pred_min_mean)
        logging.info("Test MAE (minutes, mean regressor): %.3f", mae_mean)
        for q in quants:
            loss = _pinball(yte_min.to_numpy(), pred_quants[q], q)
            logging.info("Pinball loss q=%.2f: %.3f", q, loss)
        q_low, q_high = quants[0], quants[-1]
        ql, qh = pred_quants[q_low], pred_quants[q_high]
        inside = ((yte_min.to_numpy() >= ql) & (yte_min.to_numpy() <= qh)).mean()
        width = float(np.mean(qh - ql))
        logging.info("Interval q=%.2f–%.2f coverage: %.3f  avg width: %.3f", q_low, q_high, inside, width)
        auc1  = None if p1_single else (_safe_auc(yte_p1,  proba_p1))
        auc60 = _safe_auc(yte_p60, proba_p60)
        logging.info("Test AUC (played1): %s", "n/a (single class)" if auc1 is None else f"{auc1:.3f}")
        logging.info("Test AUC (played60): %s", "n/a (single class)" if auc60 is None else f"{auc60:.3f}")
        if p1_single:
            logging.info("Brier played1: skipped (single class)")
        else:
            try:
                logging.info("Brier played1 raw: %.4f", brier_score_loss(yte_p1, proba_p1))
            except Exception as e:
                logging.info("Brier played1 raw: n/a (%s)", e)
            if proba_p1_cal is not None:
                logging.info("Brier played1 cal: %.4f", brier_score_loss(yte_p1, proba_p1_cal))
            else:
                logging.info("Brier played1 cal: skipped (no calibrator)")
        try:
            logging.info("Brier played60 raw: %.4f", brier_score_loss(yte_p60, proba_p60))
        except Exception as e:
            logging.info("Brier played60 raw: n/a (%s)", e)
        if proba_p60_cal is not None:
            logging.info("Brier played60 cal: %.4f", brier_score_loss(yte_p60, proba_p60_cal))
        else:
            logging.info("Brier played60 cal: skipped (no calibrator)")
        logging.info("MAE by position (TEST, mean regressor):")
        for pos in ["GK","DEF","MID","FWD"]:
            mask = (df_feat.loc[target_idx, "pos"] == pos).to_numpy()
            if mask.sum() == 0: continue
            mae_pos = mean_absolute_error(yte_min[mask], pred_min_mean[mask])
            logging.info("  • %-3s: %.3f  (n=%d)", pos, mae_pos, mask.sum())

    # -------- save --------
    outdir = args.models_out / args.model_version
    outdir.mkdir(parents=True, exist_ok=True)

    # mean models
    reg_global.booster_.save_model(outdir / "exp_minutes_global_lgbm.txt")
    for pos, model in regs_by_pos.items():
        model.booster_.save_model(outdir / f"exp_minutes_{pos}_lgbm.txt")

    # quantile models
    for q, model in q_global.items():
        model.booster_.save_model(outdir / f"exp_minutes_q{str(q).replace('.','_')}_global_lgbm.txt")
    for q, posmap in q_by_pos.items():
        for pos, model in posmap.items():
            model.booster_.save_model(outdir / f"exp_minutes_q{str(q).replace('.','_')}_{pos}_lgbm.txt")

    # classifiers & calibrators
    if (not p1_single) and ('clf_p1' in locals()) and (clf_p1 is not None):
        clf_p1.booster_.save_model(outdir / "played1_lgbm.txt")
    clf_p60.booster_.save_model(outdir / "played60_lgbm.txt")
    if args.calibrate:
        if (not p1_single) and (proba_p1_cal is not None):
            joblib.dump(iso_p1,  outdir / "played1_isotonic.joblib")
        if proba_p60_cal is not None:
            joblib.dump(iso_p60, outdir / "played60_isotonic.joblib")

    # dump TARGET predictions
    dump = df_feat.loc[target_idx, ["season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes","days_since_last","fdr"]].copy()
    dump["pred_exp_minutes"] = pred_min_mean
    if 0.5 in pred_quants:
        dump["pred_exp_minutes_med"] = pred_quants[0.5]
    for q in quants:
        col = f"pred_minutes_q{str(q).replace('.','_')}"
        dump[col] = pred_quants[q]

    dump["prob_played60_raw"] = proba_p60
    if not p1_single:
        dump["prob_played1_raw"] = proba_p1
    if proba_p60_cal is not None:
        dump["prob_played60_cal"] = proba_p60_cal
    if (not p1_single) and (proba_p1_cal is not None):
        dump["prob_played1_cal"] = proba_p1_cal

    if p1_single:
        p60_use = dump.get("prob_played60_cal", dump["prob_played60_raw"])
        dump["exp_appearance_points"] = 1.0 + p60_use
    else:
        p1_use  = dump.get("prob_played1_cal",  dump["prob_played1_raw"])
        p60_use = dump.get("prob_played60_cal", dump["prob_played60_raw"])
        dump["exp_appearance_points"] = p1_use + p60_use

    dump_fp = (outdir / "minutes_predictions.csv")
    dump.to_csv(dump_fp, index=False)
    logging.info("Wrote predictions to %s", dump_fp.resolve())
    if not dump_fp.exists():
        raise IOError(f"Failed to create predictions file at {dump_fp.resolve()}")

    (outdir / "features_used.txt").write_text("\n".join(feat_cols), encoding="utf-8")
    logging.info("Models & predictions saved to %s", outdir.resolve())

if __name__ == "__main__":
    main()
