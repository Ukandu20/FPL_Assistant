#!/usr/bin/env python3
"""
minutes_model_builder.py – starter-aware minutes predictor with per-position heads

New in 1.1.0:
  • Bench cameo decomposition for minutes:
      p_cameo = P(minutes>0 | bench, X, pos)
      μ_cameo = E[minutes | bench & cameo, X, pos]  (per-position regressor)
      bench minutes = p_cameo * μ_cameo  (then caps)
  • Keeps GK routing + per-position thresholds and direct P60 head.

Components:
  • P(start | X, pos): per-position LightGBM classifiers (+global fallback), optional isotonic cal.
  • P(cameo | bench, X, pos): per-position cameo head (+global fallback), optional isotonic cal,
      with degenerate detection (if no DNPs in TRAIN for a pos → p_cameo≡1 for that pos).
  • μ_cameo(X, pos): per-position LightGBM regressor on bench&cameo rows (minutes>0) with fallback.
  • P(60+ | X): direct P60 head (+optional isotonic calibration).
  • Two regressors (legacy): minutes|start (L2). (We no longer use the old bench L1 head for minutes.)
  • Hybrid routing: thresholds (t_lo,t_hi) with optional per-position overrides; GK can bypass mixing.

Outputs:
  - expected_minutes.csv (includes pred_start_head, pred_bench_cameo_head, pred_bench_head, etc.)
  - risky_starters.csv
  - metrics.json (incl. version)
  - artifacts under <model-out>/ and <model-out>/versions/<vN>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import joblib

# ------------------------------- Versioning -----------------------------------

code_version = "1.1.0"

def _ensure_version_dirs(base: Path, bump: bool, tag: Optional[str]) -> Tuple[Path, str]:
    versions_dir = base / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    latest_file = base / "latest_version.txt"

    if tag:
        vname = tag
        vdir = versions_dir / vname
        vdir.mkdir(parents=True, exist_ok=True)
        latest_file.write_text(vname)
        return vdir, vname

    if latest_file.exists():
        cur = latest_file.read_text().strip()
        if cur and not bump:
            vname = cur
            vdir = versions_dir / vname
            vdir.mkdir(parents=True, exist_ok=True)
            return vdir, vname
        try:
            nxt = int(cur[1:]) + 1 if cur.startswith("v") and cur[1:].isdigit() else 1
        except Exception:
            nxt = 1
        vname = f"v{nxt}"
    else:
        vname = "v1"
    vdir = versions_dir / vname
    vdir.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(vname)
    return vdir, vname

# ------------------------------- I/O -----------------------------------------

KEEP_COLS = [
    "player_id", "player", "pos", "gw_orig", "date_played", "minutes",
    "is_starter", "days_since_last", "is_active", "venue", "fdr_home", "fdr_away",
    "team_id"
]

def load_minutes(seasons: List[str], fix_root: Path) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = fix_root / season / "player_minutes_calendar.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        df = pd.read_csv(path, parse_dates=["date_played"], dayfirst=False)
        if "minutes" not in df.columns and "min" in df.columns:
            df = df.rename(columns={"min": "minutes"})
        cols = [c for c in KEEP_COLS if c in df.columns]
        df = df[cols].copy()
        df["season"] = season
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["minutes"] = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0)
    out["gw_orig"] = pd.to_numeric(out["gw_orig"], errors="coerce")
    for c in ("is_starter","days_since_last","is_active"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "pos" not in out.columns:
        out["pos"] = "MID"
    return out

# ------------------------------- Features ------------------------------------

def make_features(df: pd.DataFrame,
                  halflife_min: float,
                  halflife_start: float,
                  days_cap: Optional[int],
                  use_log_days: bool,
                  use_fdr: bool,
                  add_team_rotation: bool) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "date_played", "gw_orig"]).copy()

    if use_fdr and {"venue", "fdr_home", "fdr_away"}.issubset(df.columns):
        df["fdr"] = np.where(
            df["venue"].astype(str).str.lower().eq("home"),
            df["fdr_home"].fillna(0.0),
            df["fdr_away"].fillna(0.0),
        )
    else:
        df["fdr"] = 0.0

    df["min_lag1"] = df.groupby(["player_id", "season"], sort=False)["minutes"].shift(1)
    df["played_last"] = (df["min_lag1"].fillna(0) >= 1).astype(int)

    df["min_ewm_hl2"] = (
        df.groupby(["player_id", "season"], sort=False)["minutes"]
          .transform(lambda s: s.shift(1).ewm(halflife=halflife_min, adjust=False).mean())
    )

    prev_date = df.groupby(["player_id", "season"], sort=False)["date_played"].shift(1)
    ds = (df["date_played"] - prev_date).dt.days
    ds = ds.clip(lower=0).fillna(14)
    if use_log_days:
        df["days_feat"] = np.log1p(ds)
    else:
        df["days_feat"] = ds if days_cap is None else ds.clip(upper=days_cap)
    df["long_gap14"] = (ds > 14).astype(int)

    df["is_starter"] = pd.to_numeric(df["is_starter"], errors="coerce").fillna(0).astype(int)
    prev_start_raw = df.groupby(["player_id", "season"], sort=False)["is_starter"].shift(1)
    had_prev       = prev_start_raw.notna().astype(int)
    prev_start     = prev_start_raw.fillna(0).astype(int) * had_prev
    prev_bench     = (1 - prev_start) * had_prev

    df["start_lag1"] = prev_start
    df["start_rate_hl3"] = (
        df.groupby(["player_id", "season"], sort=False)["is_starter"]
          .transform(lambda s: s.shift(1).ewm(halflife=halflife_start, adjust=False).mean())
    ).fillna(0.0).clip(0, 1)

    def _consecutive_ones(s: pd.Series) -> pd.Series:
        s = s.astype(int)
        grp = (s == 0).cumsum()
        return s.groupby(grp).cumsum()

    keys = [df["player_id"], df["season"]]
    df["start_streak"] = prev_start.groupby(keys, sort=False).transform(_consecutive_ones).astype(int)
    df["bench_streak"] = prev_bench.groupby(keys, sort=False).transform(_consecutive_ones).astype(int)

    if add_team_rotation and {"team_id","player_id","season","gw_orig","is_starter"}.issubset(df.columns):
        try:
            def team_rot_rate(s: pd.DataFrame) -> pd.Series:
                s = s.sort_values(["date_played","gw_orig"])
                starters = s.pivot_table(index=["season","gw_orig","team_id"], columns="player_id",
                                         values="is_starter", aggfunc="max").fillna(0)
                num_changes = (starters.diff().abs().sum(axis=1)).rolling(3, min_periods=1).mean()
                out = num_changes.groupby(level=[0,2]).transform(lambda x: x/11.0)
                return out.reindex(s.set_index(["season","gw_orig","team_id"]).index).to_numpy()
            df["team_rot3"] = df.groupby(["team_id"], group_keys=False).apply(team_rot_rate)
            df["team_rot3"] = pd.to_numeric(df["team_rot3"], errors="coerce").fillna(0.0).clip(0,1)
        except Exception:
            df["team_rot3"] = 0.0
    else:
        df["team_rot3"] = 0.0

    pos_map = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    df["pos_enc"] = df["pos"].map(pos_map).fillna(2).astype(int)

    df["minutes"] = df["minutes"].clip(0, 120)

    df["y60"] = (df["minutes"] >= 60).astype(int)
    df["y_played"] = (df["minutes"] > 0).astype(int)

    return df

# ------------------------------- Split ---------------------------------------

def chrono_split(df: pd.DataFrame, first_test_gw: int, first_test_season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df.query(
        "(season < @first_test_season) or "
        "(season == @first_test_season and gw_orig < @first_test_gw)"
    ).copy()
    test = df.query(
        "season == @first_test_season and gw_orig >= @first_test_gw"
    ).copy()
    return train, test

def chrono_tail_index(df: pd.DataFrame, frac: float = 0.15) -> Tuple[pd.Index, pd.Index]:
    if len(df) < 10:
        return df.index, df.index
    dfo = df.sort_values(["season", "date_played", "gw_orig"])
    k = max(1, int(round(frac * len(dfo))))
    val_idx = dfo.index[-k:]
    fit_idx = dfo.index.difference(val_idx)
    return fit_idx, val_idx

def _chrono_tail_with_both_classes(df: pd.DataFrame, ycol: str,
                                   base_frac: float = 0.15,
                                   max_frac: float = 0.50) -> Tuple[pd.Index, pd.Index]:
    dfo = df.sort_values(["season", "date_played", "gw_orig"])
    n = len(dfo)
    for frac in [base_frac, 0.20, 0.25, 0.30, 0.40, max_frac]:
        k = max(1, int(round(frac * n)))
        val_idx = dfo.index[-k:]
        if dfo.loc[val_idx, ycol].nunique() >= 2:
            fit_idx = dfo.index.difference(val_idx)
            return fit_idx, val_idx
    k = max(1, int(round(base_frac * n)))
    val_idx = dfo.index[-k:]
    fit_idx = dfo.index.difference(val_idx)
    return fit_idx, val_idx

# ------------------------------- Models --------------------------------------

def train_start_classifier(train: pd.DataFrame, features: List[str],
                           is_unbalance: bool, scale_pos_weight: float,
                           monotone_features: Optional[set] = None) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    y = train["is_starter"].astype(int)
    X = train[features].fillna(0)
    params = dict(
        objective="binary",
        n_estimators=900,
        learning_rate=0.045,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    if is_unbalance:
        params["is_unbalance"] = True
    if scale_pos_weight > 0:
        params["scale_pos_weight"] = scale_pos_weight
    if monotone_features:
        mono = [1 if f in monotone_features else 0 for f in features]
        params["monotone_constraints"] = mono

    clf = lgb.LGBMClassifier(**params)
    fi, vi = chrono_tail_index(train, frac=0.15)
    clf.fit(
        X.loc[fi], y.loc[fi],
        eval_set=[(X.loc[vi], y.loc[vi])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)]
    )
    auc = bs = None
    try:
        p_val = clf.predict_proba(X.loc[vi])[:, 1]
        auc = roc_auc_score(y.loc[vi], p_val)
        bs = brier_score_loss(y.loc[vi], p_val)
    except Exception:
        pass
    return clf, {"val_auc": None if auc is None else float(auc),
                 "val_brier": None if bs is None else float(bs)}

def train_start_classifier_by_pos(train: pd.DataFrame, features: List[str],
                                  is_unbalance: bool, scale_pos_weight: float,
                                  monotone_features: Optional[set] = None
                                  ) -> Tuple[Dict[str, lgb.LGBMClassifier],
                                             Dict[str, Dict[str, float]],
                                             Dict[str, Optional[IsotonicRegression]]]:
    models: Dict[str, lgb.LGBMClassifier] = {}
    stats: Dict[str, Dict[str, float]] = {}
    calibs: Dict[str, Optional[IsotonicRegression]] = {}
    for pos in ["GK","DEF","MID","FWD"]:
        tr = train[train["pos"] == pos]
        if tr.empty:
            continue
        y = tr["is_starter"].astype(int)
        X = tr[features].fillna(0)
        params = dict(
            objective="binary",
            n_estimators=800,
            learning_rate=0.045,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        if is_unbalance: params["is_unbalance"] = True
        if scale_pos_weight > 0: params["scale_pos_weight"] = scale_pos_weight
        if monotone_features:
            mono = [1 if f in monotone_features else 0 for f in features]
            params["monotone_constraints"] = mono

        clf = lgb.LGBMClassifier(**params)
        fi, vi = chrono_tail_index(tr, frac=0.15)
        clf.fit(
            X.loc[fi], y.loc[fi],
            eval_set=[(X.loc[vi], y.loc[vi])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(70, verbose=False)]
        )
        models[pos] = clf
        calibs[pos] = None
        stats[pos] = {"rows": int(len(tr))}
        if y.loc[vi].nunique() >= 2:
            p_val = clf.predict_proba(X.loc[vi])[:,1]
            try:
                stats[pos]["val_auc"] = float(roc_auc_score(y.loc[vi], p_val))
            except Exception:
                stats[pos]["val_auc"] = None
            try:
                stats[pos]["val_brier"] = float(brier_score_loss(y.loc[vi], p_val))
            except Exception:
                stats[pos]["val_brier"] = None
            calibs[pos] = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(p_val, y.loc[vi])
    return models, stats, calibs

def train_regressor(train: pd.DataFrame, features: List[str], objective: str, min_rows: int = 200) -> Tuple[object, Dict[str, float]]:
    if len(train) < min_rows:
        prior = train.groupby("pos")["minutes"].median().to_dict()
        return ("CONST_PRIOR", prior), {"rows": int(len(train))}
    X = train[features].fillna(0)
    y = train["minutes"].clip(0, 120)
    params = dict(
        objective=objective,
        n_estimators=1200 if objective == "regression_l2" else 800,
        learning_rate=0.05,
        num_leaves=127 if objective == "regression_l2" else 63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model = lgb.LGBMRegressor(**params)
    fit_idx, val_idx = chrono_tail_index(train, frac=0.15)
    model.fit(
        X.loc[fit_idx], y.loc[fit_idx],
        eval_set=[(X.loc[val_idx], y.loc[val_idx])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)]
    )
    preds = np.clip(model.predict(X.loc[val_idx]), 0, 120)
    mae = mean_absolute_error(y.loc[val_idx], preds)
    return model, {"val_mae": float(mae), "rows": int(len(train))}

def train_prob_calibrator(train: pd.DataFrame, clf: lgb.LGBMClassifier, feat: List[str]) -> Optional[IsotonicRegression]:
    X = train[feat].fillna(0)
    y = train["is_starter"].astype(int)
    _, val_idx = chrono_tail_index(train, frac=0.15)
    if y.loc[val_idx].nunique() < 2:
        return None
    p_val_raw = clf.predict_proba(X.loc[val_idx])[:, 1]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(p_val_raw, y.loc[val_idx])
    return iso

def train_p60_direct(train: pd.DataFrame, features: List[str],
                     is_unbalance: bool, scale_pos_weight: float) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    y = train["y60"].astype(int)
    X = train[features].fillna(0)
    params = dict(
        objective="binary",
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=79,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=24,
        n_jobs=-1,
        verbosity=-1,
    )
    if is_unbalance: params["is_unbalance"] = True
    if scale_pos_weight > 0: params["scale_pos_weight"] = scale_pos_weight
    clf = lgb.LGBMClassifier(**params)
    fi, vi = chrono_tail_index(train, 0.15)
    clf.fit(
        X.loc[fi], y.loc[fi],
        eval_set=[(X.loc[vi], y.loc[vi])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(60, verbose=False)]
    )
    auc = bs = None
    try:
        p_val = clf.predict_proba(X.loc[vi])[:,1]
        auc = roc_auc_score(y.loc[vi], p_val)
        bs = brier_score_loss(y.loc[vi], p_val)
    except Exception:
        pass
    return clf, {"val_auc": None if auc is None else float(auc),
                 "val_brier": None if bs is None else float(bs)}

# --------- Cameo classification (bench) --------------------------------------

def train_cameo_given_bench(train: pd.DataFrame, features: List[str],
                            is_unbalance: bool,
                            scale_pos_weight: float
                           ) -> Tuple[Optional[lgb.LGBMClassifier],
                                      Dict[str, float],
                                      Optional[IsotonicRegression]]:
    tb = train[train["is_starter"] == 0]
    if tb.empty:
        return None, {"rows": 0}, None
    y = tb["y_played"].astype(int)
    X = tb[features].fillna(0)
    pos = int(y.sum()); neg = int(len(y) - pos)
    spw = scale_pos_weight if scale_pos_weight > 0 else (neg / max(pos, 1))
    params = dict(
        objective="binary", n_estimators=600, learning_rate=0.05, num_leaves=63,
        subsample=0.9, colsample_bytree=0.9, random_state=77, n_jobs=-1, verbosity=-1
    )
    if is_unbalance: params["is_unbalance"] = True
    if spw > 0: params["scale_pos_weight"] = spw
    clf = lgb.LGBMClassifier(**params)
    fi, vi = _chrono_tail_with_both_classes(tb, "y_played", base_frac=0.15, max_frac=0.50)
    clf.fit(
        X.loc[fi], y.loc[fi],
        eval_set=[(X.loc[vi], y.loc[vi])], eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    info = {"rows": int(len(tb))}
    iso: Optional[IsotonicRegression] = None
    if y.loc[vi].nunique() >= 2:
        p_val = clf.predict_proba(X.loc[vi])[:, 1]
        try:
            info["val_auc"] = float(roc_auc_score(y.loc[vi], p_val))
        except Exception:
            info["val_auc"] = None
        try:
            info["val_brier"] = float(brier_score_loss(y.loc[vi], p_val))
        except Exception:
            info["val_brier"] = None
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(p_val, y.loc[vi])
    return clf, info, iso

def train_cameo_given_bench_by_pos(train: pd.DataFrame, features: List[str],
                                   is_unbalance: bool,
                                   scale_pos_weight: float
                                  ) -> Tuple[Dict[str, lgb.LGBMClassifier],
                                             Dict[str, Dict[str, float]],
                                             Dict[str, Optional[IsotonicRegression]]]:
    models: Dict[str, lgb.LGBMClassifier] = {}
    stats: Dict[str, Dict[str, float]] = {}
    calibs: Dict[str, Optional[IsotonicRegression]] = {}
    for pos_tag in ["GK", "DEF", "MID", "FWD"]:
        tb = train[(train["is_starter"] == 0) & (train["pos"] == pos_tag)]
        if tb.empty:
            continue
        y = tb["y_played"].astype(int)
        X = tb[features].fillna(0)
        pos_pos = int(y.sum()); pos_neg = int(len(y) - pos_pos)
        spw = scale_pos_weight if scale_pos_weight > 0 else (pos_neg / max(pos_pos, 1))
        params = dict(
            objective="binary", n_estimators=500, learning_rate=0.05, num_leaves=63,
            subsample=0.9, colsample_bytree=0.9, random_state=177, n_jobs=-1, verbosity=-1
        )
        if is_unbalance: params["is_unbalance"] = True
        if spw > 0: params["scale_pos_weight"] = spw
        clf = lgb.LGBMClassifier(**params)
        fi, vi = _chrono_tail_with_both_classes(tb, "y_played", base_frac=0.15, max_frac=0.60)
        clf.fit(
            X.loc[fi], y.loc[fi],
            eval_set=[(X.loc[vi], y.loc[vi])], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(40, verbose=False)]
        )
        models[pos_tag] = clf
        calibs[pos_tag] = None
        stats[pos_tag] = {"rows": int(len(tb))}
        if y.loc[vi].nunique() >= 2:
            p_val = clf.predict_proba(X.loc[vi])[:, 1]
            try:
                stats[pos_tag]["val_auc"] = float(roc_auc_score(y.loc[vi], p_val))
            except Exception:
                stats[pos_tag]["val_auc"] = None
            try:
                stats[pos_tag]["val_brier"] = float(brier_score_loss(y.loc[vi], p_val))
            except Exception:
                stats[pos_tag]["val_brier"] = None
            calibs[pos_tag] = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(p_val, y.loc[vi])
    return models, stats, calibs

# --------- Cameo minutes regression (bench & cameo only) ---------------------

def train_cameo_minutes_by_pos(train: pd.DataFrame, features: List[str],
                               min_rows: int = 150
                               ) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]], Optional[object]]:
    """
    Train per-position regressors on bench&cameo rows (is_starter==0 & minutes>0).
    Returns (models_by_pos, stats_by_pos, global_fallback)
    """
    models: Dict[str, object] = {}
    stats: Dict[str, Dict[str, float]] = {}
    global_fallback = None

    cameo_rows = train[(train["is_starter"]==0) & (train["minutes"]>0)]
    # global fallback
    if len(cameo_rows) >= min_rows:
        global_fallback, ginfo = train_regressor(cameo_rows, features, objective="regression_l1", min_rows=min_rows)
    else:
        # constant prior
        prior = cameo_rows.groupby("pos")["minutes"].median().to_dict() if len(cameo_rows) else {"DEF":20,"MID":25,"FWD":20,"GK":1}
        global_fallback = ("CONST_PRIOR", prior)

    for pos in ["GK","DEF","MID","FWD"]:
        trp = cameo_rows[cameo_rows["pos"]==pos]
        if len(trp) < min_rows:
            continue
        model, info = train_regressor(trp, features, objective="regression_l1", min_rows=min_rows)
        models[pos] = model
        stats[pos] = info
    return models, stats, global_fallback

# ------------------------------- Predict utils --------------------------------

def predict_with_model(model, X: pd.DataFrame, default_val: float = 0.0, clip_hi: float | None = None) -> np.ndarray:
    if isinstance(model, tuple) and model[0] == "CONST_PRIOR":
        prior = model[1]
        pred = X.get("pos", pd.Series(index=X.index)).map(prior).fillna(default_val).to_numpy()
    else:
        Xn = X.select_dtypes(include=[np.number])
        pred = model.predict(Xn.fillna(0))
    pred = np.asarray(pred, dtype=float)
    if clip_hi is not None:
        pred = np.clip(pred, 0, clip_hi)
    return np.clip(pred, 0, 120)

def taper_start_minutes(pred_start: np.ndarray, p_start: np.ndarray, pos_series: pd.Series) -> np.ndarray:
    non_gk = (pos_series.values != "GK")
    w = np.clip((p_start - 0.2) / (0.6 - 0.2), 0.0, 1.0)
    out = pred_start.copy()
    out[non_gk] = (0.5 + 0.5 * w[non_gk]) * out[non_gk]
    return out

def hybrid_route(p: np.ndarray, start_pred: np.ndarray, bench_pred: np.ndarray,
                 mix_pred: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    return np.where(p >= t_hi, start_pred, np.where(p <= t_lo, bench_pred, mix_pred))

def per_position_bench_cap_from_train(pos_ser: pd.Series, caps_map: Dict[str, float]) -> np.ndarray:
    return pos_ser.map(caps_map).fillna(25.0).to_numpy()

# --------- Autotune thresholds (bench-aware) ---------

def autotune_thresholds(train: pd.DataFrame,
                        predict_p_start_fn,
                        reg_start,
                        feat_reg: List[str],
                        bench_cap_map: Optional[Dict[str, float]],
                        use_taper: bool, use_pos_caps: bool) -> Tuple[float, float]:
    fi, vi = chrono_tail_index(train, 0.15)
    Xr = train.loc[vi, feat_reg].fillna(0)
    yv = train.loc[vi, "minutes"].to_numpy()
    bench_mask = train.loc[vi, "is_starter"].astype(int).eq(0).to_numpy()

    p = predict_p_start_fn(train.loc[vi])
    ps = predict_with_model(reg_start, Xr, 60.0, 120)
    if use_taper:
        ps = taper_start_minutes(ps, p, train.loc[vi, "pos"])
    # Bench path for tuning: use a simple cameo proxy (median cameo minutes) to avoid training dependency.
    cameo_train = train[(train["is_starter"]==0) & (train["minutes"]>0)]
    cameo_median = cameo_train.groupby("pos")["minutes"].median().to_dict() if len(cameo_train) else {"DEF":20,"MID":25,"FWD":20,"GK":1}
    pb0 = train.loc[vi, "pos"].map(cameo_median).fillna(20).to_numpy()
    # Assume 60% cameo rate as proxy (doesn't need to be perfect for threshold tuning)
    pb = 0.6 * pb0
    if use_pos_caps and bench_cap_map is not None:
        pb = np.minimum(pb, per_position_bench_cap_from_train(train.loc[vi, "pos"], bench_cap_map))

    best = (1e9, 0.2, 0.8)
    mix = p*ps + (1-p)*pb
    for t_lo in np.linspace(0.15, 0.30, 4):
        for t_hi in np.linspace(0.65, 0.80, 4):
            pred = hybrid_route(p, ps, pb, mix, float(t_lo), float(t_hi))
            mae_all = mean_absolute_error(yv, pred)
            mae_ben = mean_absolute_error(yv[bench_mask], pred[bench_mask]) if bench_mask.any() else mae_all
            J = mae_all + 0.5 * mae_ben
            if J < best[0]:
                best = (J, float(t_lo), float(t_hi))
    return best[1], best[2]

# ------------------------------- Parsing utils --------------------------------

def parse_pos_thresholds(s: Optional[str]) -> Dict[str, Tuple[float,float]]:
    out: Dict[str, Tuple[float,float]] = {}
    if not s:
        return out
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":")
        a, b = v.split(",")
        out[k.strip().upper()] = (float(a), float(b))
    return out

# ------------------------------- Main ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True,
                    help="Comma-separated seasons, last is test season (e.g., 2020-2021,...,2024-2025)")
    ap.add_argument("--first-test-gw", type=int, default=26)
    ap.add_argument("--fix-root", default="data/processed/registry/fixtures")
    ap.add_argument("--model-out", default="data/models/minutes")
    ap.add_argument("--log-level", default="INFO")

    # Feature knobs
    ap.add_argument("--halflife-min", type=float, default=2.0)
    ap.add_argument("--halflife-start", type=float, default=3.0)
    ap.add_argument("--days-cap", type=int, default=14, help="Set -1 to disable")
    ap.add_argument("--use-log-days", action="store_true")
    ap.add_argument("--use-fdr", action="store_true")
    ap.add_argument("--add-team-rotation", action="store_true")

    # AUC / imbalance knobs
    ap.add_argument("--is-unbalance", action="store_true")
    ap.add_argument("--scale-pos-weight", type=float, default=0.0)
    ap.add_argument("--gate-monotone", action="store_true", help="Monotone constraints on p_start (safe features only)")
    ap.add_argument("--gate-blend", type=float, default=0.5, help="Blend weight for per-pos vs global start gate (1.0=pos only)")

    # Calibration + routing knobs
    ap.add_argument("--use-calibration", action="store_true", help="Isotonic-calibrate p_start (pos & global where available)")
    ap.add_argument("--t-lo", type=float, default=0.20)
    ap.add_argument("--t-hi", type=float, default=0.80)
    ap.add_argument("--use-taper", action="store_true")
    ap.add_argument("--use-pos-bench-caps", action="store_true")
    ap.add_argument("--pos-thresholds", type=str, default="", help="Per-pos thresholds e.g. 'GK:0.15,0.55;DEF:0.25,0.70;MID:0.25,0.70;FWD:0.25,0.70'")
    ap.add_argument("--no-mix-gk", action="store_true", help="Disable GK soft mix (ambiguous → route to start)")

    # Guardrails
    ap.add_argument("--bench-cap", type=float, default=45.0, help="(legacy, unused if --use-pos-bench-caps set)")
    ap.add_argument("--pstart-cap10", type=float, default=0.05)
    ap.add_argument("--pstart-cap30", type=float, default=0.20)

    # P60 head
    ap.add_argument("--p60-mode", choices=["direct","structured"], default="direct")
    ap.add_argument("--use-p60-calibration", action="store_true")

    # Versioning
    ap.add_argument("--bump-version", action="store_true")
    ap.add_argument("--version-tag", type=str, default="")

    # Risky starters report
    ap.add_argument("--risky-start-thresh", type=float, default=0.70)
    ap.add_argument("--risky-topk", type=int, default=50)

    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    first_test_season = seasons[-1]
    fix_root = Path(args.fix_root)
    out_dir = Path(args.model_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    version_dir, version_name = _ensure_version_dirs(out_dir, bump=args.bump_version, tag=(args.version_tag or None))

    df = load_minutes(seasons, fix_root)
    days_cap = None if (args.days_cap is not None and args.days_cap < 0) else args.days_cap
    df = make_features(df,
                       halflife_min=args.halflife_min,
                       halflife_start=args.halflife_start,
                       days_cap=days_cap,
                       use_log_days=args.use_log_days,
                       use_fdr=args.use_fdr,
                       add_team_rotation=args.add_team_rotation)

    train, test = chrono_split(df, args.first_test_gw, first_test_season)
    train = train.dropna(subset=["min_lag1", "min_ewm_hl2"])

    cameo_train = train[(train["is_starter"]==0) & (train["minutes"]>0)]
    if cameo_train.empty:
        bench_caps = {"GK":5.0, "DEF":20.0, "MID":30.0, "FWD":30.0}
    else:
        bench_caps = cameo_train.groupby("pos")["minutes"].quantile(0.95).to_dict()

    feat_start = ["start_lag1", "start_rate_hl3", "min_lag1", "min_ewm_hl2",
                  "played_last", "days_feat", "long_gap14", "start_streak", "bench_streak",
                  "pos_enc", "team_rot3", "fdr"]
    feat_reg = ["min_lag1", "min_ewm_hl2", "played_last", "days_feat", "long_gap14",
                "start_lag1", "start_rate_hl3", "pos_enc", "team_rot3", "fdr"]
    feat_p60 = feat_start
    feat_cameo = ["min_lag1","min_ewm_hl2","played_last","days_feat","long_gap14",
                  "start_rate_hl3","bench_streak","pos_enc","team_rot3","fdr"]
    feat_cameo_min = ["min_lag1","min_ewm_hl2","played_last","days_feat","long_gap14",
                      "start_rate_hl3","bench_streak","pos_enc","team_rot3","fdr"]

    # Detect bench subsets with no DNPs (single class)
    single_class_bench: set[str] = set()
    for pos in ["GK","DEF","MID","FWD"]:
        tb = train[(train["is_starter"]==0) & (train["pos"]==pos)]
        if not tb.empty and tb["y_played"].nunique() < 2:
            single_class_bench.add(pos)

    # Start gate
    mono = {"start_rate_hl3","min_lag1"} if args.gate_monotone else None
    gate_global, _ = train_start_classifier(train, feat_start,
                                            is_unbalance=args.is_unbalance,
                                            scale_pos_weight=args.scale_pos_weight,
                                            monotone_features=mono)
    gate_global_calib = train_prob_calibrator(train, gate_global, feat_start) if args.use_calibration else None

    gate_by_pos, gate_pos_stats, gate_pos_calibs = train_start_classifier_by_pos(
        train, feat_start, is_unbalance=args.is_unbalance,
        scale_pos_weight=args.scale_pos_weight, monotone_features=mono
    )

    def predict_p_start(df_in: pd.DataFrame) -> np.ndarray:
        X_all = df_in[feat_start].fillna(0)
        n = len(df_in)
        p_pos = np.full(n, np.nan, dtype=float)
        for pos in ["GK","DEF","MID","FWD"]:
            m = gate_by_pos.get(pos)
            if m is None:
                continue
            mask = (df_in["pos"].values == pos)
            if not mask.any():
                continue
            idx = np.flatnonzero(mask)
            X_pos = X_all.iloc[idx]
            pv = m.predict_proba(X_pos)[:, 1]
            iso = gate_pos_calibs.get(pos) if args.use_calibration else None
            if iso is not None:
                pv = iso.transform(pv)
            p_pos[idx] = pv
        p_glob = gate_global.predict_proba(X_all)[:, 1]
        if args.use_calibration and gate_global_calib is not None:
            p_glob = gate_global_calib.transform(p_glob)
        blend = float(np.clip(args.gate_blend, 0.0, 1.0))
        p = np.where(~np.isnan(p_pos), blend * p_pos + (1.0 - blend) * p_glob, p_glob)
        return np.clip(p, 0, 1)

    # Start minutes regressor (unchanged)
    reg_start, _ = train_regressor(train[train["is_starter"] == 1], feat_reg, objective="regression_l2")

    # P60 head (direct)
    if args.p60_mode != "direct":
        raise NotImplementedError("structured P60 disabled in this build; use --p60-mode direct")
    p60_direct, _ = train_p60_direct(train, feat_p60,
                                     is_unbalance=args.is_unbalance, scale_pos_weight=args.scale_pos_weight)
    p60_direct_calib = None
    if args.use_p60_calibration and p60_direct is not None:
        _, vi = chrono_tail_index(train, 0.15)
        Xv = train.loc[vi, feat_p60].fillna(0)
        yv = train.loc[vi, "y60"].astype(int)
        if yv.nunique() >= 2:
            pv = p60_direct.predict_proba(Xv)[:,1]
            p60_direct_calib = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(pv, yv)

    # Cameo classifiers (bench)
    cameo_global, cameo_global_info, cameo_global_calib = train_cameo_given_bench(
        train, feat_cameo, is_unbalance=args.is_unbalance, scale_pos_weight=args.scale_pos_weight
    )
    cameo_models_by_pos, cameo_pos_stats, cameo_calibs_by_pos = train_cameo_given_bench_by_pos(
        train, feat_cameo, is_unbalance=args.is_unbalance, scale_pos_weight=args.scale_pos_weight
    )

    # Cameo MINUTES regressors (bench & cameo only)
    cameo_min_by_pos, cameo_min_stats, cameo_min_global = train_cameo_minutes_by_pos(
        train, feat_cameo_min, min_rows=150
    )

    # Autotune thresholds (bench-aware objective; uses proxy pb)
    try:
        t_lo_auto, t_hi_auto = autotune_thresholds(train, predict_p_start,
                                                   reg_start, feat_reg,
                                                   bench_cap_map=bench_caps if args.use_pos_bench_caps else None,
                                                   use_taper=args.use_taper,
                                                   use_pos_caps=args.use_pos_bench_caps)
        # Keep user overrides if they provide per-pos thresholds
        if not args.pos_thresholds:
            args.t_lo, args.t_hi = t_lo_auto, t_hi_auto
    except Exception:
        pass

    pos_thresholds = parse_pos_thresholds(args.pos_thresholds)

    # ---------- Predict on TEST ----------
    p_start = predict_p_start(test)

    Xr = test[feat_reg]
    pred_start = predict_with_model(reg_start, Xr, default_val=60.0, clip_hi=120)
    if args.use_taper:
        pred_start = taper_start_minutes(pred_start, p_start, test["pos"])

    # p60 direct
    p60 = p60_direct.predict_proba(test[feat_p60].fillna(0))[:,1]
    if p60_direct_calib is not None:
        p60 = p60_direct_calib.transform(p60)

    # p_cameo
    if len(cameo_models_by_pos) > 0:
        p_cameo = np.zeros(len(test), dtype=float)
        used = np.zeros(len(test), dtype=bool)
        for pos in ["GK","DEF","MID","FWD"]:
            if pos in cameo_models_by_pos:
                m = cameo_models_by_pos[pos]
                mask = (test["pos"] == pos).values
                if mask.any():
                    pv = m.predict_proba(test.loc[mask, feat_cameo].fillna(0))[:, 1]
                    iso = cameo_calibs_by_pos.get(pos)
                    if iso is not None:
                        pv = iso.transform(pv)
                    p_cameo[mask] = pv
                    used[mask] = True
        if (~used).any() and cameo_global is not None:
            pv = cameo_global.predict_proba(test.loc[~used, feat_cameo].fillna(0))[:, 1]
            if cameo_global_calib is not None:
                pv = cameo_global_calib.transform(pv)
            p_cameo[~used] = pv
    else:
        if cameo_global is not None:
            p_cameo = cameo_global.predict_proba(test[feat_cameo].fillna(0))[:, 1]
            if cameo_global_calib is not None:
                p_cameo = cameo_global_calib.transform(p_cameo)
        else:
            cameo = train[(train["is_starter"] == 0)]
            base = {"GK": 1.0, "DEF": 1.0, "MID": 1.0, "FWD": 1.0}
            if not cameo.empty:
                base.update(cameo.groupby("pos")["y_played"].mean().to_dict())
            p_cameo = test["pos"].map(base).fillna(1.0).to_numpy()

    # force p_cameo=1.0 for positions with no DNPs in TRAIN benches
    if single_class_bench:
        for pos in single_class_bench:
            m = (test["pos"].values == pos)
            if m.any():
                p_cameo[m] = 1.0
    p_cameo = np.clip(p_cameo, 0, 1)

    # μ_cameo: minutes given cameo (per-pos regressors with global fallback)
    def predict_cameo_minutes(df_in: pd.DataFrame) -> np.ndarray:
        Xc = df_in[feat_cameo_min].fillna(0)
        mu = np.zeros(len(df_in), dtype=float)
        used = np.zeros(len(df_in), dtype=bool)
        for pos in ["GK","DEF","MID","FWD"]:
            model = cameo_min_by_pos.get(pos)
            if model is None:
                continue
            mask = (df_in["pos"].values == pos)
            if not mask.any():
                continue
            mu[mask] = predict_with_model(model, Xc.loc[mask], default_val=15.0, clip_hi=None)
            used[mask] = True
        if (~used).any() and cameo_min_global is not None:
            mu[~used] = predict_with_model(cameo_min_global, Xc.loc[~used], default_val=15.0, clip_hi=None)
        # Cameo minutes rarely exceed ~45; clip softly
        return np.clip(mu, 0, 60)

    mu_cameo = predict_cameo_minutes(test)

    # bench minutes via decomposition
    pred_bench_cameo = mu_cameo
    pred_bench = p_cameo * pred_bench_cameo

    # apply caps
    if args.use_pos_bench_caps:
        pred_bench = np.minimum(pred_bench, per_position_bench_cap_from_train(test["pos"], bench_caps))
    else:
        pred_bench = np.clip(pred_bench, 0, args.bench_cap)

    # thresholds per-row
    tlo = np.full(len(test), args.t_lo, dtype=float)
    thi = np.full(len(test), args.t_hi, dtype=float)
    pos_thresholds = parse_pos_thresholds(args.pos_thresholds)
    for pos, (a, b) in pos_thresholds.items():
        m = (test["pos"].values == pos)
        if m.any():
            tlo[m] = a
            thi[m] = b

    # soft mix
    mix_pred = p_start * pred_start + (1.0 - p_start) * pred_bench

    # GK routing with optional no-mix
    minutes_pred = np.empty(len(test), dtype=float)
    is_gk = (test["pos"].values == "GK")
    if args.no_mix_gk and is_gk.any():
        minutes_pred[is_gk] = np.where(
            p_start[is_gk] >= thi[is_gk], pred_start[is_gk],
            np.where(p_start[is_gk] <= tlo[is_gk], pred_bench[is_gk], pred_start[is_gk])
        )
    if (~is_gk).any():
        idx = ~is_gk
        minutes_pred[idx] = np.where(
            p_start[idx] >= thi[idx], pred_start[idx],
            np.where(p_start[idx] <= tlo[idx], pred_bench[idx], mix_pred[idx])
        )

    # Guardrails
    minutes_pred = np.where(p_start < args.pstart_cap10, np.minimum(minutes_pred, 10.0), minutes_pred)
    minutes_pred = np.where(p_start < args.pstart_cap30, np.minimum(minutes_pred, 30.0), minutes_pred)
    minutes_pred = np.clip(minutes_pred, 0, 120)

    # p60 and EV minute points
    if p60_direct_calib is not None:
        p60 = p60_direct_calib.transform(p60)
    p_play = np.clip(p_start + (1.0 - p_start) * p_cameo, 0, 1)
    exp_min_points = np.clip(p_play + p60, 0, 2)

    # Metrics
    y_true = test["minutes"].to_numpy()
    true_pts = np.select([y_true >= 60, y_true > 0], [2, 1], default=0)
    metrics = {
        "code_version": code_version,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "minutes_test_MAE": round(float(mean_absolute_error(y_true, minutes_pred)), 4),
    }
    try:
        ys = test["is_starter"].astype(int)
        metrics["start_AUC_test"] = round(float(roc_auc_score(ys, p_start)), 4)
        metrics["start_Brier_test"] = round(float(brier_score_loss(ys, p_start)), 4)
    except Exception:
        pass
    try:
        mask_s = test["is_starter"].astype(int).eq(1).to_numpy()
        mask_b = ~mask_s
        if mask_s.any():
            metrics["minutes_MAE_Started"] = round(float(mean_absolute_error(y_true[mask_s], minutes_pred[mask_s])), 4)
        if mask_b.any():
            metrics["minutes_MAE_Benched"] = round(float(mean_absolute_error(y_true[mask_b], minutes_pred[mask_b])), 4)
        metrics["bench_share"] = float(mask_b.mean())
    except Exception:
        pass
    try:
        y60_true = test["y60"].astype(int)
        metrics["p60_AUC_test"] = round(float(roc_auc_score(y60_true, p60)), 4)
        metrics["p60_Brier_test"] = round(float(brier_score_loss(y60_true, p60)), 4)
    except Exception:
        pass
    metrics["ev_minutes_points_MAE"] = round(float(np.mean(np.abs(exp_min_points - true_pts))), 4)

    # Persist CSV (include heads)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        "season": test["season"].values,
        "gw_orig": test["gw_orig"].values,
        "date_played": test["date_played"].values,
        "player_id": test["player_id"].values,
        "player": test["player"].values,
        "pos": test["pos"].values,
        "is_starter": test["is_starter"].values,
        "minutes_true": y_true,
        "true_pts": true_pts,
        "p_start": p_start,
        "p_cameo": p_cameo,
        "p_play": p_play,
        "p60": p60,
        "pred_start_head": pred_start,
        "pred_bench_cameo_head": pred_bench_cameo,
        "pred_bench_head": pred_bench,
        "pred_minutes": minutes_pred,
        "exp_minutes_points": exp_min_points
    })
    pred_df.to_csv(out_dir / "expected_minutes.csv", index=False)
    pred_df.to_csv(version_dir / "expected_minutes.csv", index=False)

    # Risky starters report
    eps = 1e-9
    risk_sub60 = np.maximum(0.0, p_start - p60)
    q_start_hat = np.clip(p60 / np.maximum(p_start, eps), 0.0, 1.0)
    mask = p_start >= args.risky_start_thresh
    risky_df = pd.DataFrame({
        "season": test["season"].values,
        "gw_orig": test["gw_orig"].values,
        "date_played": test["date_played"].values,
        "player_id": test["player_id"].values,
        "player": test["player"].values,
        "pos": test["pos"].values,
        "p_start": p_start,
        "p60": p60,
        "risk_start_sub60": risk_sub60,
        "q_start_hat": q_start_hat,
        "pred_minutes": minutes_pred,
        "minutes_true": y_true,
        "true_pts": true_pts,
        "exp_minutes_points": exp_min_points
    })
    risky_out = risky_df[mask].sort_values("risk_start_sub60", ascending=False).head(args.risky_topk)
    risky_out.to_csv(out_dir / "risky_starters.csv", index=False)
    risky_out.to_csv(version_dir / "risky_starters.csv", index=False)

    # Save models
    try:
        gate_global.booster_.save_model(out_dir / "start_classifier_global.txt")
        gate_global.booster_.save_model(version_dir / "start_classifier_global.txt")
        if gate_global_calib is not None:
            joblib.dump(gate_global_calib, out_dir / "start_classifier_global_iso.joblib")
            joblib.dump(gate_global_calib, version_dir / "start_classifier_global_iso.joblib")
        for pos, m in gate_by_pos.items():
            m.booster_.save_model(out_dir / f"start_classifier_{pos}.txt")
            m.booster_.save_model(version_dir / f"start_classifier_{pos}.txt")
            if gate_pos_calibs.get(pos) is not None:
                joblib.dump(gate_pos_calibs[pos], out_dir / f"start_classifier_{pos}_iso.joblib")
                joblib.dump(gate_pos_calibs[pos], version_dir / f"start_classifier_{pos}_iso.joblib")
    except Exception:
        pass

    p60_direct.booster_.save_model(out_dir / "p60_direct.txt")
    p60_direct.booster_.save_model(version_dir / "p60_direct.txt")
    if p60_direct_calib is not None:
        joblib.dump(p60_direct_calib, out_dir / "p60_direct_calib.joblib")
        joblib.dump(p60_direct_calib, version_dir / "p60_direct_calib.joblib")

    if cameo_global is not None:
        cameo_global.booster_.save_model(out_dir / "cameo_given_bench_global.txt")
        cameo_global.booster_.save_model(version_dir / "cameo_given_bench_global.txt")
        if cameo_global_calib is not None:
            joblib.dump(cameo_global_calib, out_dir / "cameo_given_bench_global_iso.joblib")
            joblib.dump(cameo_global_calib, version_dir / "cameo_given_bench_global_iso.joblib")
    for pos_tag, m in cameo_models_by_pos.items():
        m.booster_.save_model(out_dir / f"cameo_given_bench_{pos_tag}.txt")
        m.booster_.save_model(version_dir / f"cameo_given_bench_{pos_tag}.txt")
        if cameo_calibs_by_pos.get(pos_tag) is not None:
            joblib.dump(cameo_calibs_by_pos[pos_tag], out_dir / f"cameo_given_bench_{pos_tag}_iso.joblib")
            joblib.dump(cameo_calibs_by_pos[pos_tag], version_dir / f"cameo_given_bench_{pos_tag}_iso.joblib")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logging.info(json.dumps(metrics, indent=2))
    logging.info(f"Artifacts written to {out_dir} (latest) and {version_dir} (versioned: {version_name}).")

if __name__ == "__main__":
    main()
