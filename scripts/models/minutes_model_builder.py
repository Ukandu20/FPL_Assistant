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


def _load_all(features_dir: Path, version: str, seasons: List[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        fp = features_dir / version / s / "players_form.csv"
        if not fp.is_file():
            logging.warning("Missing %s – skipped", fp)
            continue
        df = pd.read_csv(fp, parse_dates=["date_played"])
        df["season"] = s
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

    # leak-safe within-season lag
    df["prev_minutes"] = df.groupby(["player_id","season"], sort=False)["minutes"].shift(1)

    # targets
    df["y_minutes"]  = df["minutes"].astype(float)
    df["y_played1"]  = (df["minutes"] >= 1).astype(int)
    df["y_played60"] = (df["minutes"] >= 60).astype(int)

    # base numeric features
    feats = ["prev_minutes", "days_since_last", "is_active", "fdr"]

    # categorical encodings
    df["venue_bin"] = (df["venue"] == "Home").astype(int); feats.append("venue_bin")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df["pos_enc"] = enc.fit_transform(df[["pos"]]); feats.append("pos_enc")

    # rolling features from players_form.csv
    roll_cols = [c for c in df.columns if c.endswith(("_roll", "_home_roll", "_away_roll")) and not c.endswith("_roll_z")]
    if use_z:
        roll_cols += [c for c in df.columns if c.endswith(("_roll_z", "_home_roll_z", "_away_roll_z"))]

    # drop sparse columns
    keep = [c for c in roll_cols if df[c].notna().mean() >= na_thresh]
    feats.extend(sorted(keep))

    # require previous minutes
    before = len(df)
    df = df.dropna(subset=["prev_minutes"]).reset_index(drop=True)
    after = len(df)
    if after == 0:
        raise ValueError("No rows left after requiring prev_minutes.")
    logging.info("Dropped %d rows with missing prev_minutes; using %d rows.", before - after, after)

    X = df[feats].copy()
    y_min  = df["y_minutes"].copy()
    y_p1   = df["y_played1"].copy()
    y_p60  = df["y_played60"].copy()
    return X, y_min, y_p1, y_p60, feats, df


# -------------- time split --------------
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


def _select_calibration_tail(train_df: pd.DataFrame, frac: float, test_season: str) -> Tuple[pd.Index, pd.Index]:
    """
    Return (fit_idx, cal_idx) as **index labels**, not boolean masks.
    Prefer tail of training rows from test_season distribution; else overall tail.
    """
    tr = train_df.sort_values(["season","date_played","gw_orig"]).copy()
    n = len(tr)
    k = max(1, int(n * frac))

    in_test_season = tr["season"] == test_season
    idx_ts = tr.index[in_test_season]
    if len(idx_ts) >= k:
        cal_idx = idx_ts[-k:]
    else:
        cal_idx = tr.index[-k:]

    fit_idx = tr.index.difference(cal_idx)
    return fit_idx, cal_idx


# -------------- recency weights --------------
def _season_recency_weights(seasons_sorted: List[str], max_weight: float) -> Dict[str, float]:
    """
    Oldest season gets 1.0, newest gets max_weight, linear spacing between.
    """
    n = len(seasons_sorted)
    if n == 1:
        return {seasons_sorted[0]: 1.0}
    w = {}
    for i, s in enumerate(seasons_sorted):
        w[s] = 1.0 + (max_weight - 1.0) * (i / (n - 1))
    return w


# -------------- monotone constraints --------------
def _monotone_vector(feat_cols: List[str], enable: bool) -> Optional[List[int]]:
    """
    +1 on features we *know* should be monotone increasing with minutes:
      - prev_minutes
      - is_active
    0 on others.
    """
    if not enable:
        return None
    vec = []
    for c in feat_cols:
        if c == "prev_minutes":
            vec.append(1)
        elif c == "is_active":
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
    # only fit if both classes exist
    u = np.unique(y_true)
    if len(u) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_raw, y_true, sample_weight=sample_weight)
    return iso


def _safe_auc(y, p) -> Optional[float]:
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else None


def _pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    if mean_pinball_loss is not None:
        return float(mean_pinball_loss(y_true, y_pred, alpha=q))
    # manual pinball loss
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


# -------------- main --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--version", required=True, help="features version folder, e.g., v7")
    ap.add_argument("--test-season", required=True, help="e.g., 2024-2025")
    ap.add_argument("--test-last-n", type=int, default=10, help="number of last GWs in test-season")
    ap.add_argument("--use-z", action="store_true", help="include *_roll_z features as well")
    ap.add_argument("--na-thresh", type=float, default=0.70, help="drop feature columns with < coverage")
    ap.add_argument("--calibrate", action="store_true", help="fit isotonic calibration on played1/played60")
    ap.add_argument("--cal-frac", type=float, default=0.2, help="fraction of TRAIN rows as chrono tail for calibration")
    ap.add_argument("--monotone", action="store_true", help="enable monotone constraints on prev_minutes and is_active")
    ap.add_argument("--recency-weight-max", type=float, default=1.5, help="newest season weight (oldest fixed at 1.0)")
    ap.add_argument("--quantiles", default="0.1,0.5,0.9", help="comma list, e.g. 0.1,0.5,0.9")
    ap.add_argument("--models-out", type=Path, default=Path("data/models/expected_minutes"))
    ap.add_argument("--model-version", default="v1", help="models subdir name, e.g., v3")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    # parse quantiles
    quants = [float(x) for x in args.quantiles.split(",") if x.strip() != ""]
    quants = sorted([q for q in quants if 0.0 < q < 1.0])
    if not quants:
        raise ValueError("No valid quantiles provided via --quantiles")

    seasons = _find_seasons(args.features_dir, args.version)
    logging.info("Found seasons: %s", ", ".join(seasons))

    df = _load_all(args.features_dir, args.version, seasons)
    X, y_min, y_p1, y_p60, feat_cols, df_feat = _build_dataset(df, use_z=args.use_z, na_thresh=args.na_thresh)

    # time split (by index labels)
    train_idx, test_idx = _time_split_last_n(df_feat, args.test_season, args.test_last_n)
    Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
    ytr_min, yte_min   = y_min.loc[train_idx],  y_min.loc[test_idx]
    ytr_p1,  yte_p1    = y_p1.loc[train_idx],   y_p1.loc[test_idx]
    ytr_p60, yte_p60   = y_p60.loc[train_idx],  y_p60.loc[test_idx]

    logging.info("Train rows: %d • Test rows: %d", len(train_idx), len(test_idx))

    # class balance diagnostics
    def _counts(y):
        v = y.value_counts().to_dict()
        return f"{v.get(0,0)}/{v.get(1,0)} (0/1)"
    logging.info("Class balance (TRAIN) played1:  %s", _counts(ytr_p1))
    logging.info("Class balance (TRAIN) played60: %s", _counts(ytr_p60))
    logging.info("Class balance (TEST)  played1:  %s", _counts(yte_p1))
    logging.info("Class balance (TEST)  played60: %s", _counts(yte_p60))

    # -------- recency weights --------
    season_weights = _season_recency_weights(seasons, args.recency_weight_max)
    df_feat["recency_w"] = df_feat["season"].map(season_weights).astype(float)
    w_train = df_feat.loc[train_idx, "recency_w"].to_numpy()

    # -------- monotone vector --------
    mono_vec = _monotone_vector(feat_cols, enable=args.monotone)

    # -------- mean models --------
    logging.info("Training expected minutes regressor (GLOBAL) …")
    reg_global = _mk_regressor(mono_vec)
    reg_global.fit(Xtr, ytr_min, sample_weight=w_train)

    # per-position mean regressors
    logging.info("Training per-position minutes regressors …")
    regs_by_pos: Dict[str, lgb.LGBMRegressor] = {}
    for pos in ["GK","DEF","MID","FWD"]:
        pos_mask_tr = (df_feat.loc[train_idx, "pos"] == pos).to_numpy()
        if pos_mask_tr.sum() < 100:
            continue
        reg_pos = _mk_regressor(mono_vec)
        reg_pos.fit(Xtr[pos_mask_tr], ytr_min[pos_mask_tr], sample_weight=w_train[pos_mask_tr])
        regs_by_pos[pos] = reg_pos

    # -------- quantile models --------
    logging.info("Training quantile minutes regressors …")
    q_global: Dict[float, lgb.LGBMRegressor] = {}
    q_by_pos: Dict[float, Dict[str, lgb.LGBMRegressor]] = {}
    for q in quants:
        # global
        m = _mk_quantile_regressor(q)
        m.fit(Xtr, ytr_min, sample_weight=w_train)
        q_global[q] = m
        # per pos
        q_by_pos[q] = {}
        for pos in ["GK","DEF","MID","FWD"]:
            pos_mask_tr = (df_feat.loc[train_idx, "pos"] == pos).to_numpy()
            if pos_mask_tr.sum() < 100:
                continue
            mpos = _mk_quantile_regressor(q)
            mpos.fit(Xtr[pos_mask_tr], ytr_min[pos_mask_tr], sample_weight=w_train[pos_mask_tr])
            q_by_pos[q][pos] = mpos

    # played1 (handle single class)
    p1_single = (len(np.unique(ytr_p1)) < 2)
    if p1_single:
        logging.info("played1 has a single class -> skipping training; setting p1=1.0 on TEST")
        clf_p1 = None
    else:
        logging.info("Training played1 classifier …")
        clf_p1 = _mk_classifier(mono_vec)
        clf_p1.fit(Xtr, ytr_p1, sample_weight=w_train)

    logging.info("Training played60 classifier …")
    clf_p60 = _mk_classifier(mono_vec)
    clf_p60.fit(Xtr, ytr_p60, sample_weight=w_train)

    # -------- predictions (TEST) --------
    pos_te = df_feat.loc[test_idx, "pos"]
    # mean prediction: try pos-specific, else global
    pred_min_mean = np.empty(len(test_idx), dtype=float)
    for i, (ix, p) in enumerate(zip(test_idx, pos_te)):
        model = regs_by_pos.get(p, reg_global)
        pred_min_mean[i] = model.predict(X.loc[[ix]])[0]
    pred_min_mean = np.clip(pred_min_mean, 0, 120)

    # quantile predictions
    pred_quants: Dict[float, np.ndarray] = {q: np.empty(len(test_idx), dtype=float) for q in quants}
    for q in quants:
        for i, (ix, p) in enumerate(zip(test_idx, pos_te)):
            model = q_by_pos[q].get(p, q_global[q])
            pred_quants[q][i] = model.predict(X.loc[[ix]])[0]
        pred_quants[q] = np.clip(pred_quants[q], 0, 120)

    # classifiers
    if p1_single:
        proba_p1 = np.ones(len(test_idx), dtype=float)
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
        train_df = df_feat.loc[train_idx]
        fit_idx, cal_idx = _select_calibration_tail(train_df, args.cal_frac, args.test_season)

        # sample weights for these splits
        w_fit = df_feat.loc[fit_idx, "recency_w"].to_numpy()
        w_cal = df_feat.loc[cal_idx, "recency_w"].to_numpy()

        # refit classifiers on FIT subset only (with weights)
        if not p1_single:
            clf_p1 = _mk_classifier(mono_vec)
            clf_p1.fit(X.loc[fit_idx], y_p1.loc[fit_idx], sample_weight=w_fit)

        clf_p60 = _mk_classifier(mono_vec)
        clf_p60.fit(X.loc[fit_idx], y_p60.loc[fit_idx], sample_weight=w_fit)

        # raw probs on CAL subset
        if not p1_single:
            p1_raw_cal = clf_p1.predict_proba(X.loc[cal_idx])[:, 1]
            iso_p1 = _fit_isotonic(y_p1.loc[cal_idx].to_numpy(), p1_raw_cal, w_cal)

        p60_raw_cal = clf_p60.predict_proba(X.loc[cal_idx])[:, 1]
        iso_p60 = _fit_isotonic(y_p60.loc[cal_idx].to_numpy(), p60_raw_cal, w_cal)

        # apply calibrated mapping on TEST if available
        if not p1_single:
            proba_p1 = clf_p1.predict_proba(Xte)[:, 1]
            proba_p1_cal = iso_p1.predict(proba_p1) if iso_p1 is not None else None

        proba_p60 = clf_p60.predict_proba(Xte)[:, 1]
        proba_p60_cal = iso_p60.predict(proba_p60) if iso_p60 is not None else None

    # -------- metrics --------
    mae_mean = mean_absolute_error(yte_min, pred_min_mean)
    logging.info("Test MAE (minutes, mean regressor): %.3f", mae_mean)

    # Quantile metrics: pinball + coverage/width for central interval if provided
    for q in quants:
        loss = _pinball(yte_min.to_numpy(), pred_quants[q], q)
        logging.info("Pinball loss q=%.2f: %.3f", q, loss)

    # coverage & width for q_low / q_high if both present (use first and last)
    q_low, q_high = quants[0], quants[-1]
    ql = pred_quants[q_low]
    qh = pred_quants[q_high]
    inside = ((yte_min.to_numpy() >= ql) & (yte_min.to_numpy() <= qh)).mean()
    width = float(np.mean(qh - ql))
    logging.info("Interval q=%.2f–%.2f coverage: %.3f  avg width: %.3f", q_low, q_high, inside, width)

    auc1  = None if p1_single else (_safe_auc(yte_p1,  proba_p1))
    auc60 = _safe_auc(yte_p60, proba_p60)
    logging.info("Test AUC (played1): %s", "n/a (single class)" if auc1 is None else f"{auc1:.3f}")
    logging.info("Test AUC (played60): %s", "n/a (single class)" if auc60 is None else f"{auc60:.3f}")

    # Brier
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

    # MAE by position on TEST (mean predictor)
    logging.info("MAE by position (TEST, mean regressor):")
    for pos in ["GK","DEF","MID","FWD"]:
        mask = (df_feat.loc[test_idx, "pos"] == pos).to_numpy()
        if mask.sum() == 0:
            continue
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

    # dump predictions
    dump = df_feat.loc[test_idx, ["season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes","days_since_last","fdr"]].copy()
    dump["pred_exp_minutes"] = pred_min_mean  # mean regressor (for backward compat)
    # also write median if available
    if 0.5 in pred_quants:
        dump["pred_exp_minutes_med"] = pred_quants[0.5]
    # write all configured quantiles
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

    # expected appearance points
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
