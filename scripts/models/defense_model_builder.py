#!/usr/bin/env python3
r"""
defense_model_builder.py – v7.2

Fixes vs v7.1:
• BUGFIX: _write_missing_join_csv now uses positional boolean indexing
  (frame.iloc[miss_mask.to_numpy()]) to avoid index-alignment errors.

Features retained from v7.1:
• Predict-only mode (--predict-season [--predict-gws]).
• --use-z selects z-variants where available.
• Metrics gated when labels are missing (future-season).
• Coverage CSVs, feature artifacts, monotone logging, safe joins.

Outputs:
  data/models/defense/<model_version>/
    ├─ cs_lgbm.txt
    ├─ cs_isotonic.joblib           (if --calibrate)
    ├─ dcp_DEF_lgbm.joblib          (and MID/FWD if trained)
    ├─ artifacts/
    │   ├─ cs_features.json
    │   ├─ dcp_features.json
    │   ├─ cs_feature_importances.csv
    │   ├─ dcp_DEF_feature_importances.csv (etc)
    │   └─ monotone_constraints.json       (if used)
    └─ predictions/
        ├─ defence_predictions.csv
        └─ missing_minutes_join.csv        (if any rows dropped)
"""

from __future__ import annotations
import argparse, logging, re, math, json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error
from sklearn.isotonic import IsotonicRegression
import joblib


# ───────────────── IO ─────────────────

def _find_seasons(features_dir: Path, version: str) -> List[str]:
    base = features_dir / version
    if not base.exists():
        raise FileNotFoundError(f"{base} does not exist")
    dirs = [d for d in base.iterdir() if d.is_dir()]
    seasons = [d.name for d in dirs if re.match(r"^\d{4}-\d{4}$", d.name)]
    if not seasons:
        seasons = [d.name for d in dirs if (d / "players_form.csv").is_file()]
    if not seasons:
        raise FileNotFoundError(f"No season folders under {base}")
    return sorted(seasons)


def _load_players(features_dir: Path, version: str, seasons: List[str]) -> pd.DataFrame:
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
        raise FileNotFoundError("No players_form.csv files loaded")
    df = pd.concat(frames, ignore_index=True)

    need = {"season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"players_form missing required columns: {miss}")

    if "ga" not in df.columns:
        df["ga"] = np.nan

    if "clr" not in df.columns:
        logging.warning("`clr` column not found in players_form; treating as 0 for DCP.")
        df["clr"] = 0
    for c in ["blocks","tkl","int","recoveries","fdr_home","fdr_away","days_since_last","is_active"]:
        if c not in df.columns:
            df[c] = np.nan

    df["rid"] = df.index.astype(int)
    return df


def _load_minutes_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"minutes_preds file not found at {path}")
    mp = pd.read_csv(path, parse_dates=["date_played"])
    need = {"season","gw_orig","date_played","player_id","pred_exp_minutes"}
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
    return mp[key + ["pred_exp_minutes","prob_played60_use"]].copy()


# ─────────────── team_form merge ───────────────

def _merge_team_features(df_players: pd.DataFrame, team_form_dir: Path, version: str) -> pd.DataFrame:
    seasons = sorted(df_players["season"].unique())
    t_frames = []
    for s in seasons:
        fp = team_form_dir / version / s / "team_form.csv"
        if not fp.is_file():
            logging.warning("team_form missing for %s", s)
            continue
        tf = pd.read_csv(fp, parse_dates=["date_played"])
        tf["season"] = s
        t_frames.append(tf)
    if not t_frames:
        logging.warning("No team_form files found – skipping team features")
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

    use_team = ["season","gw_orig","team_id","team_att_z_venue","team_def_xga_venue","team_def_xga_venue_z","team_possession_venue","opp_id"]
    use_team = [c for c in use_team if c in tf.columns]
    T = tf[use_team].drop_duplicates()
    df = df_players.merge(T, on=["season","gw_orig","team_id"], how="left")

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


# ─────────────── splits ───────────────

def _time_split_last_n(df: pd.DataFrame, test_season: str, last_n_gws: int) -> Tuple[pd.Index, pd.Index]:
    gws = sorted(df.loc[df["season"]==test_season, "gw_orig"].dropna().unique())
    if not gws:
        raise ValueError(f"No gw_orig for season {test_season}")
    test_gws = set(gws[-last_n_gws:])
    test_idx = df.index[(df["season"]==test_season) & (df["gw_orig"].isin(test_gws))]
    train_idx = df.index.difference(test_idx)
    if len(test_idx)==0:
        raise ValueError("Empty test split")
    return train_idx, test_idx


def _select_calibration_tail(train_df: pd.DataFrame, frac: float, test_season: str) -> Tuple[pd.Index, pd.Index]:
    tr = train_df.sort_values(["season","date_played","gw_orig"]).copy()
    n = len(tr); k = max(1, int(n*frac))
    idx_ts = tr.index[tr["season"]==test_season]
    cal_idx = idx_ts[-k:] if len(idx_ts)>=k else tr.index[-k:]
    fit_idx = tr.index.difference(cal_idx)
    return fit_idx, cal_idx


def _build_inference_index(df: pd.DataFrame, season: str, gws: Optional[List[int]]) -> pd.Index:
    mask = (df["season"] == season)
    if gws is not None:
        mask &= df["gw_orig"].isin(gws)
    idx = df.index[mask]
    if len(idx) == 0:
        raise ValueError(f"No rows available to predict for season={season} gws={gws}")
    return idx


# ─────────────── features ───────────────

def _build_cs_features(df: pd.DataFrame, use_z: bool) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df["venue_bin"] = (df["venue"]=="Home").astype(int)
    df["fdr"] = np.where(df["venue"]=="Home", df.get("fdr_home", np.nan), df.get("fdr_away", np.nan)).astype(float)
    feats = ["venue_bin","fdr"]

    if use_z and "team_def_xga_venue_z" in df.columns:
        choose = ["opp_att_z_venue","team_def_xga_venue_z","team_possession_venue"]
    else:
        choose = ["opp_att_z_venue","team_def_xga_venue","team_possession_venue"]
    feats += [c for c in choose if c in df.columns]

    X = df[feats].copy()
    return X, feats


def _build_dcp_features(df: pd.DataFrame, na_thresh: float, use_z: bool) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df["venue_bin"] = (df["venue"]=="Home").astype(int)
    df["fdr"] = np.where(df["venue"]=="Home", df.get("fdr_home", np.nan), df.get("fdr_away", np.nan)).astype(float)
    base = ["venue_bin","fdr","team_possession_venue","opp_att_z_venue"]
    base.append("team_def_xga_venue_z" if use_z and "team_def_xga_venue_z" in df.columns else "team_def_xga_venue")

    roll = [c for c in df.columns if c.startswith("def_") and c.endswith(("_p90_roll","_p90_home_roll","_p90_away_roll","_p90_roll_z","_p90_home_roll_z","_p90_away_roll_z"))]
    roll = [c for c in roll if df[c].notna().mean() >= na_thresh]
    feats = [c for c in base if c in df.columns] + sorted(roll)
    X = df[feats].copy()
    return X, feats


# ─────────────── models ───────────────

def _lgbm_cls(monotone: Optional[List[int]] = None) -> lgb.LGBMClassifier:
    params = dict(objective="binary", n_estimators=2000, learning_rate=0.03,
                  num_leaves=127, min_data_in_leaf=20, subsample=0.9, colsample_bytree=0.9,
                  reg_lambda=1.0, n_jobs=-1, random_state=42, verbosity=-1)
    if monotone is not None:
        params["monotone_constraints"] = monotone
    return lgb.LGBMClassifier(**params)


def _lgbm_reg() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="regression", n_estimators=2000, learning_rate=0.03,
                             num_leaves=127, min_data_in_leaf=20, subsample=0.9, colsample_bytree=0.9,
                             reg_lambda=1.0, n_jobs=-1, random_state=42, verbosity=-1)


def _fit_isotonic(y_true: np.ndarray, p_raw: np.ndarray) -> Optional[IsotonicRegression]:
    if len(np.unique(y_true)) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, y_true)
    return iso


# ─────────────── math ───────────────

def _poisson_tail_prob_vec(lam: np.ndarray, k: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    k = np.asarray(k, dtype=int)
    out = np.zeros_like(lam, dtype=float)

    def tail_one(l, kk):
        if not np.isfinite(l) or l <= 0:
            return 0.0 if kk > 0 else 1.0
        term = math.exp(-l) * (l ** kk) / math.factorial(kk)
        s = term
        i = kk + 1
        for _ in range(200):
            term *= l / i
            s += term
            if term < 1e-12:
                break
            i += 1
        return min(max(s, 0.0), 1.0)

    for i in range(lam.shape[0]):
        out[i] = tail_one(lam[i], int(k[i]))
    return out


# ─────────────── artifacts / coverage ───────────────

def _save_feature_artifacts(outdir: Path,
                            cs_feats: List[str],
                            dcp_feats: List[str],
                            cs_model: lgb.LGBMClassifier,
                            dcp_models: Dict[str, lgb.LGBMRegressor],
                            monotone: Optional[List[int]]) -> None:
    art = outdir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    (art / "cs_features.json").write_text(json.dumps(cs_feats, indent=2))
    (art / "dcp_features.json").write_text(json.dumps(dcp_feats, indent=2))

    try:
        imp = pd.DataFrame({"feature": cs_feats, "importance": cs_model.feature_importances_})
        imp.to_csv(art / "cs_feature_importances.csv", index=False)
    except Exception:
        pass

    for p, m in dcp_models.items():
        try:
            imp = pd.DataFrame({"feature": dcp_feats, "importance": m.feature_importances_})
            imp.to_csv(art / f"dcp_{p}_feature_importances.csv", index=False)
        except Exception:
            pass

    if monotone is not None:
        mono_map = {f: int(monotone[i]) for i, f in enumerate(cs_feats)}
        (art / "monotone_constraints.json").write_text(json.dumps(mono_map, indent=2))


def _write_missing_join_csv(out_pred_dir: Path, frame: pd.DataFrame, miss_mask: pd.Series) -> None:
    """Write rows (from `frame`) where minutes/prob60 were missing, using positional boolean mask."""
    if isinstance(miss_mask, pd.Series):
        mask_np = miss_mask.to_numpy()
    else:
        mask_np = np.asarray(miss_mask, dtype=bool)
    if mask_np.any():
        miss = frame.iloc[mask_np].copy()
        miss.to_csv(out_pred_dir / "missing_minutes_join.csv", index=False)


# ─────────────── main ───────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--version", required=True)
    ap.add_argument("--team-form-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--minutes-preds", type=Path, required=True)
    ap.add_argument("--test-season", required=True)
    ap.add_argument("--test-last-n", type=int, default=10)

    ap.add_argument("--predict-season", type=str, default=None,
                    help="If set, run inference for this season (all GWs or --predict-gws).")
    ap.add_argument("--predict-gws", type=str, default=None,
                    help="Comma-separated GW list for inference, e.g., '1,2,3'.")

    ap.add_argument("--use-z", action="store_true")
    ap.add_argument("--na-thresh", type=float, default=0.70)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--cal-frac", type=float, default=0.2)
    ap.add_argument("--monotone-cs", action="store_true")
    ap.add_argument("--models-out", type=Path, default=Path("data/models/defense"))
    ap.add_argument("--model-version", default="v7_2")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    seasons = _find_seasons(args.features_dir, args.version)
    logging.info("Found seasons: %s", ", ".join(seasons))

    df = _load_players(args.features_dir, args.version, seasons)
    df = _merge_team_features(df, args.team_form_dir, args.version)

    df = df.sort_values(["season","date_played","gw_orig","player_id","team_id"]).reset_index(drop=True)
    df["rid"] = df.index.astype(int)

    Xcs_all, cs_feats   = _build_cs_features(df, use_z=args.use_z)
    Xdcp_all, dcp_feats = _build_dcp_features(df, na_thresh=args.na_thresh, use_z=args.use_z)

    if args.predict_season:
        train_idx, _ = _time_split_last_n(df, args.test_season, args.test_last_n)
        df_train = df.loc[train_idx].copy()
        pred_gws = None if not args.predict_gws else [int(x) for x in args.predict_gws.split(",")]
        pred_idx = _build_inference_index(df, args.predict_season, pred_gws)
        df_target = df.loc[pred_idx].copy()
        logging.info("Mode: PREDICT-ONLY for season %s%s",
                     args.predict_season,
                     f" GWs {args.predict_gws}" if args.predict_gws else " (all GWs)")
    else:
        train_idx, test_idx = _time_split_last_n(df, args.test_season, args.test_last_n)
        df_train = df.loc[train_idx].copy()
        df_target = df.loc[test_idx].copy()
        logging.info("Mode: EVAL (last %d GWs of %s)", args.test_last_n, args.test_season)

    mp = _load_minutes_predictions(args.minutes_preds)
    key = ["season","gw_orig","date_played","player_id"]

    target_keys = df_target[key + ["rid","team_id","player","pos","venue","minutes","ga"]].copy()
    join = target_keys.merge(mp, on=key, how="left", validate="many_to_one")

    miss_mask = join["pred_exp_minutes"].isna() | join["prob_played60_use"].isna()
    n_drop = int(miss_mask.sum())
    if n_drop:
        logging.warning("Dropping %d/%d target rows with missing predicted minutes/prob_played60.", n_drop, len(join))

    kept = join.loc[~miss_mask].reset_index(drop=True)

    # TRAIN CS
    y_cs_tr = (df_train["ga"] == 0).astype(float).to_numpy()
    y_cs_tr_valid = np.isfinite(y_cs_tr)
    if y_cs_tr_valid.sum() == 0:
        raise ValueError("No valid CS labels in training split (ga present and finite).")

    Xcs_tr = Xcs_all.loc[train_idx].copy()
    Xcs_tr = Xcs_tr.iloc[y_cs_tr_valid]
    y_cs_tr = y_cs_tr[y_cs_tr_valid].astype(int)

    mono_map = {"venue_bin": +1, "fdr": -1, "opp_att_z_venue": -1,
                "team_def_xga_venue": -1, "team_def_xga_venue_z": -1,
                "team_possession_venue": +1}
    monotone = [mono_map.get(f, 0) for f in cs_feats] if args.monotone_cs else None

    if args.monotone_cs:
        logging.info("Monotone constraints vector (aligned to cs_feats): %s", monotone)

    cs_model = _lgbm_cls(monotone=monotone)
    cs_model.fit(Xcs_tr, y_cs_tr)

    # optional isotonic calibration – only in eval mode
    iso_cs: Optional[IsotonicRegression] = None
    if args.calibrate and not args.predict_season:
        fit_idx, cal_idx = _select_calibration_tail(df_train, args.cal_frac, args.test_season)

        X_fit = Xcs_all.loc[fit_idx]
        y_fit = (df.loc[fit_idx,"ga"]==0).astype(float).to_numpy()
        fit_valid = np.isfinite(y_fit)
        X_fit = X_fit.iloc[fit_valid]
        y_fit = y_fit[fit_valid].astype(int)

        X_cal = Xcs_all.loc[cal_idx]
        y_cal = (df.loc[cal_idx,"ga"]==0).astype(float).to_numpy()
        cal_valid = np.isfinite(y_cal)
        X_cal = X_cal.iloc[cal_valid]
        y_cal = y_cal[cal_valid].astype(int)

        if len(np.unique(y_fit))>=2 and len(np.unique(y_cal))>=2:
            cs_model = _lgbm_cls(monotone=monotone)
            cs_model.fit(X_fit, y_fit)
            p_cal_raw = cs_model.predict_proba(X_cal)[:,1]
            iso_cs = _fit_isotonic(y_cal, p_cal_raw)
            logging.info("Isotonic calibration fitted on tail (frac=%.2f).", args.cal_frac)
        else:
            logging.info("Skipped isotonic calibration (insufficient class variation).")

    # TRAIN DCP per-position
    Xdcp_tr = Xdcp_all.loc[train_idx].copy()
    m = df_train["minutes"].fillna(0).clip(lower=0)
    m90 = (m / 90.0).replace(0, np.nan)

    c_clr = df_train["clr"].fillna(0)
    c_blk = df_train["blocks"].fillna(0)
    c_tkl = df_train["tkl"].fillna(0)
    c_int = df_train["int"].fillna(0)
    c_rec = df_train["recoveries"].fillna(0)

    pos_tr = df_train["pos"].astype(str).values
    y_dcp_p90 = np.zeros(len(df_train), dtype=float)
    is_def = (pos_tr=="DEF"); is_mid = (pos_tr=="MID"); is_fwd = (pos_tr=="FWD")
    d_def = ((c_clr+c_blk+c_tkl+c_int) / m90).fillna(0)
    d_out = ((c_clr+c_blk+c_tkl+c_int+c_rec) / m90).fillna(0)
    y_dcp_p90[is_def] = d_def.to_numpy()[is_def]
    y_dcp_p90[is_mid] = d_out.to_numpy()[is_mid]
    y_dcp_p90[is_fwd] = d_out.to_numpy()[is_fwd]
    y_dcp_p90 = np.clip(y_dcp_p90, 0, None)

    dcp_models: Dict[str, lgb.LGBMRegressor] = {}
    for p in ["DEF","MID","FWD"]:
        mask = (pos_tr == p)
        if mask.sum()==0:
            continue
        model = _lgbm_reg()
        model.fit(Xdcp_tr.loc[mask], y_dcp_p90[mask])
        dcp_models[p] = model
        logging.info("Trained DCP regressor for %s on %d rows.", p, mask.sum())

    # ───── TARGET aligned by rid ─────
    kept_rids = kept["rid"].astype(int).to_numpy()
    Xcs_te  = Xcs_all.loc[kept_rids].copy()
    Xdcp_te = Xdcp_all.loc[kept_rids].copy()

    # CS predictions
    p_cs_raw = cs_model.predict_proba(Xcs_te)[:,1]
    p_cs_cal = iso_cs.predict(p_cs_raw) if iso_cs is not None else None
    p_cs_use = p_cs_cal if p_cs_cal is not None else p_cs_raw

    # Metrics only in eval mode (and when labels exist)
    if not args.predict_season:
        yte_ga = df.loc[kept_rids, "ga"].to_numpy()
        y_has = np.isfinite(yte_ga)
        if y_has.any():
            yte_cs = (yte_ga[y_has] == 0).astype(int)
            p_eval = p_cs_use[y_has]
            if len(np.unique(yte_cs)) > 1:
                try:
                    auc_cs = roc_auc_score(yte_cs, p_eval)
                    logging.info("Test AUC (CS): %.3f", auc_cs)
                except Exception:
                    logging.info("Test AUC (CS): n/a")
            try:
                brier_cs = brier_score_loss(yte_cs, p_eval)
                logging.info("Brier (CS): %.4f", brier_cs)
            except Exception:
                pass

    # DCP per-90 predictions
    pos_te = df.loc[kept_rids, "pos"].astype(str).to_numpy()
    dcp_p90 = np.zeros(len(kept_rids), dtype=float)
    for p, model in dcp_models.items():
        mask = (pos_te == p)
        if mask.sum()==0:
            continue
        dcp_p90[mask] = np.clip(model.predict(Xdcp_te.loc[mask]), 0, None)

    exp_minutes = kept["pred_exp_minutes"].to_numpy()
    scale = exp_minutes / 90.0
    dcp_lambda_match = dcp_p90 * scale

    thresh = np.where(pos_te=="DEF", 10, 12)
    dcp_prob_hit = _poisson_tail_prob_vec(dcp_lambda_match, thresh)
    exp_dcp_points = 2.0 * dcp_prob_hit

    prob_p60 = kept["prob_played60_use"].to_numpy()
    cs_pts_by_pos = np.where(pos_te=="MID", 1.0, np.where((pos_te=="DEF")|(pos_te=="GK"), 4.0, 0.0))
    exp_cs_points = p_cs_use * prob_p60 * cs_pts_by_pos

    base_cols = ["season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes","ga"]
    dump = df.loc[kept_rids, base_cols].reset_index(drop=True)
    dump["pred_exp_minutes"] = exp_minutes
    dump["prob_played60_use"] = prob_p60
    dump["cs_prob_raw"] = p_cs_raw
    if p_cs_cal is not None:
        dump["cs_prob_cal"] = p_cs_cal
    dump["exp_cs_points"] = exp_cs_points
    dump["dcp_lambda_p90"] = dcp_p90
    dump["dcp_lambda_match"] = dcp_lambda_match
    dump["dcp_prob_hit"] = dcp_prob_hit
    dump["exp_dcp_points"] = exp_dcp_points

    for c in ["team_att_z_venue","opp_att_z_venue","team_def_xga_venue","team_def_xga_venue_z","team_possession_venue"]:
        if c in df.columns:
            dump[c] = df.loc[kept_rids, c].to_numpy()

    try:
        comp_cols = ["clr","blocks","tkl","int","recoveries","pos"]
        comp = df.loc[kept_rids, comp_cols].copy()
        clr = comp.get("clr", pd.Series(0, index=comp.index)).fillna(0)
        blk = comp.get("blocks", pd.Series(0, index=comp.index)).fillna(0)
        tkl = comp.get("tkl", pd.Series(0, index=comp.index)).fillna(0)
        inte= comp.get("int", pd.Series(0, index=comp.index)).fillna(0)
        rec = comp.get("recoveries", pd.Series(0, index=comp.index)).fillna(0)
        pos_a = comp.get("pos", pd.Series("UNK", index=comp.index)).astype(str)
        dcp_real = (clr + blk + tkl + inte + np.where(pos_a.isin(["MID","FWD"]), rec, 0)).to_numpy()
        valid = np.isfinite(dcp_real) & np.isfinite(dcp_lambda_match)
        if valid.any():
            mae_dcp = mean_absolute_error(dcp_real[valid], dcp_lambda_match[valid])
            logging.info("Test MAE (DCP count vs λ): %.3f", mae_dcp)
    except Exception:
        pass

    outdir = args.models_out / args.model_version
    pred_dir = outdir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: pass `join` (same length as mask) to the writer
    _write_missing_join_csv(pred_dir, join, miss_mask)

    cs_model.booster_.save_model(str(outdir / "cs_lgbm.txt"))
    if iso_cs is not None:
        joblib.dump(iso_cs, outdir / "cs_isotonic.joblib")
    for p, m in dcp_models.items():
        joblib.dump(m, outdir / f"dcp_{p}_lgbm.joblib")

    _save_feature_artifacts(outdir, cs_feats, dcp_feats, cs_model, dcp_models, monotone)

    fp = pred_dir / "defence_predictions.csv"
    dump.to_csv(fp, index=False)
    logging.info("Wrote predictions to %s", fp.resolve())
    logging.info("Models & predictions saved to %s", outdir.resolve())


if __name__ == "__main__":
    main()
