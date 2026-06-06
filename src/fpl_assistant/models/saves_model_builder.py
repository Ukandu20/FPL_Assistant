#!/usr/bin/env python3
r"""
saves_model_builder.py — TRAIN/TEST ONLY — v3.2.0

Contract
--------
• Do NOT predict upcoming fixtures. Uses only players_form (played games).
• TRAIN uses true minutes to form per-90 target.
• TEST scales per-90 by minutes model's expected minutes (pred_minutes).
• Split: --seasons (last = TEST) + --first-test-gw.

Outputs
-------
<model-out>/ (latest) and <model-out>/versions/<vN>/ (versioned):
  - saves_predictions.csv  (no point predictions; only saves)
  - metrics.json, meta.json
  - artifacts/
      feature_importances.csv
      features_used.txt
      missing_pred_minutes.csv (if any)
  - models/
      lgbm_saves_p90.txt
      poisson_saves_p90.joblib (optional)
      poisson_imputer.joblib (optional)
"""

from __future__ import annotations
import argparse, json, logging, os, hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error
import joblib

CODE_VERSION = "3.2.0"

# ------------------------------- Versioning -----------------------------------

def _ensure_version_dirs(base: Path, bump: bool, tag: Optional[str]) -> Tuple[Path, str]:
    versions_dir = base / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    latest_file = base / "latest_version.txt"

    if tag:
        vname = tag.strip()
        vdir = versions_dir / vname
        vdir.mkdir(parents=True, exist_ok=True)
        latest_file.write_text(vname)
        return vdir, vname

    if latest_file.exists():
        cur = latest_file.read_text().strip()
        if cur and not bump:
            vname = cur
        else:
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

def _write_meta(outdir: Path, args: argparse.Namespace) -> None:
    try:
        host = getattr(os, "uname", lambda: type("x",(object,),{"nodename":None})())().nodename
    except Exception:
        host = None
    meta = {
        "code_version": CODE_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "cmdline": " ".join(os.sys.argv),
        "hostname": host,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

# ------------------------------- I/O ------------------------------------------

def _load_players(features_root: Path, form_version: str, seasons: List[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        fp = features_root / form_version / s / "players_form.csv"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing: {fp}")
        df = pd.read_csv(fp, parse_dates=["date_played"])
        df["season"] = s
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    needed = {"season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes","saves"}
    miss = needed - set(df.columns)
    if miss:
        raise KeyError(f"players_form missing columns: {miss}")

    # GK only; players_form contains played fixtures only (no upcoming)
    df = df[df["pos"].astype(str).str.upper().eq("GK")].copy()

    # harmonize keys
    for c in ("player_id","team_id"):
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["season"] = df["season"].astype(str)
    df["gw_orig"] = pd.to_numeric(df["gw_orig"], errors="coerce").astype("Int64")

    return df.sort_values(["season","date_played","gw_orig","player_id","team_id"]).reset_index(drop=True)

def _load_team_form(features_root: Path, form_version: str, seasons: List[str]) -> Optional[pd.DataFrame]:
    frames = []
    for s in seasons:
        fp = features_root / form_version / s / "team_form.csv"
        if not fp.is_file():
            logging.warning("team_form missing for %s", s)
            continue
        t = pd.read_csv(fp, parse_dates=["date_played"])
        t["season"] = s
        frames.append(t)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def _merge_team_z(players: pd.DataFrame, team_form: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = players.copy()
    if team_form is None:
        out["team_def_z_venue"] = np.nan
        out["opp_att_z_venue"] = np.nan
        return out

    tf = team_form.copy()
    need = {"season","gw_orig","team_id","venue","home_id","away_id",
            "def_xga_home_roll_z","def_xga_away_roll_z","att_xg_home_roll_z","att_xg_away_roll_z"}
    if not need.issubset(tf.columns):
        out["team_def_z_venue"] = np.nan
        out["opp_att_z_venue"] = np.nan
        logging.warning("team_form lacks z inputs; filling NaNs.")
        return out

    tf["def_z_at_venue"] = np.where(tf["venue"].eq("Home"), tf["def_xga_home_roll_z"], tf["def_xga_away_roll_z"])
    tf["att_z_at_venue"] = np.where(tf["venue"].eq("Home"), tf["att_xg_home_roll_z"], tf["att_xg_away_roll_z"])
    tf["opp_team_id"] = np.where(tf["team_id"].astype(str).str.lower().eq(tf["home_id"].astype(str).str.lower()),
                                 tf["away_id"], tf["home_id"])

    def_lu = tf[["season","gw_orig","team_id","def_z_at_venue"]].drop_duplicates().rename(columns={"def_z_at_venue":"team_def_z_venue"})
    att_lu = tf[["season","gw_orig","opp_team_id","att_z_at_venue"]].drop_duplicates().rename(columns={"opp_team_id":"opp_team_id","att_z_at_venue":"opp_att_z_venue"})
    opp_lu = tf[["season","gw_orig","team_id","opp_team_id"]].drop_duplicates()

    # normalize ids
    for d in (def_lu, att_lu, opp_lu):
        for c in set(d.columns) & {"team_id","opp_team_id"}:
            d[c] = d[c].astype(str).str.strip().str.lower()

    out = out.merge(opp_lu, on=["season","gw_orig","team_id"], how="left")
    out = out.merge(def_lu, on=["season","gw_orig","team_id"], how="left")
    out = out.merge(att_lu, on=["season","gw_orig","opp_team_id"], how="left")
    for c in ["team_def_z_venue","opp_att_z_venue"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _load_expected_minutes(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    expected_minutes.csv from minutes model.
    Need at least: season, gw_orig, date_played, player_id, pred_minutes.
    """
    if path is None:
        return None
    if not path.is_file():
        logging.warning("expected_minutes file not found at %s", path)
        return None
    df = pd.read_csv(path, parse_dates=["date_played"])
    base_need = {"season","gw_orig","date_played","player_id","pred_minutes"}
    missing = base_need - set(df.columns)
    if missing:
        logging.warning("expected_minutes missing %s", missing)
        return None
    # normalize ids
    for c in ("player_id",):
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["season"] = df["season"].astype(str)
    df["gw_orig"] = pd.to_numeric(df["gw_orig"], errors="coerce").astype("Int64")
    key = ["season","gw_orig","date_played","player_id"]
    return df.sort_values(key).drop_duplicates(subset=key, keep="last")[key + ["pred_minutes"]].copy()

# ------------------------------- Split ----------------------------------------

def _chrono_split(df: pd.DataFrame, seasons: List[str], first_test_gw: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_season = seasons[-1]
    g = pd.to_numeric(df["gw_orig"], errors="coerce")
    s = df["season"].astype(str)
    train_mask = (s < test_season) | ((s == test_season) & (g < first_test_gw))
    test_mask  = (s == test_season) & (g >= first_test_gw)
    train = df.loc[train_mask].copy()
    test  = df.loc[test_mask].copy()
    if train.empty or test.empty:
        # fallback by date boundary if the GW number isn't fully present
        cutoff = pd.to_datetime(test["date_played"]).min()
        if pd.notna(cutoff):
            train = df[(s < test_season) | ((s == test_season) & (df["date_played"] < cutoff))].copy()
            test  = df[(s == test_season) & (df["date_played"] >= cutoff)].copy()
    if train.empty or test.empty:
        raise ValueError("Split produced empty train or test; check --seasons/--first-test-gw.")
    return train, test

def _tail_index(df: pd.DataFrame, frac: float = 0.15) -> Tuple[pd.Index, pd.Index]:
    # robust: if time columns exist, sort by them; else fall back to index order
    cols = set(df.columns)
    if {"season","date_played","gw_orig"}.issubset(cols):
        dfo = df.sort_values(["season","date_played","gw_orig"])
    else:
        dfo = df.sort_index()
    n = len(dfo)
    if n < 10:
        return dfo.index, dfo.index
    k = max(1, int(round(frac * n)))
    val_idx = dfo.index[-k:]
    fit_idx = dfo.index.difference(val_idx)
    return fit_idx, val_idx

# ------------------------------- Features -------------------------------------

def _build_features(df: pd.DataFrame, na_thresh: float) -> Tuple[pd.DataFrame, List[str]]:
    feats: List[str] = []
    X = df.copy()
    X["venue_bin"] = (X["venue"].astype(str) == "Home").astype(int); feats.append("venue_bin")
    for c in ["team_def_z_venue","opp_att_z_venue"]:
        if c in X.columns:
            feats.append(c)
    # GK rolls
    roll_candidates = [
        "gk_saves_p90_roll","gk_saves_p90_home_roll","gk_saves_p90_away_roll",
        "gk_sot_against_p90_roll","gk_sot_against_p90_home_roll","gk_sot_against_p90_away_roll",
        "gk_saves_p90_roll_z","gk_saves_p90_home_roll_z","gk_saves_p90_away_roll_z",
        "gk_sot_against_p90_roll_z","gk_sot_against_p90_home_roll_z","gk_sot_against_p90_away_roll_z",
    ]
    keep = [c for c in roll_candidates if c in X.columns and X[c].notna().mean() >= na_thresh]
    feats.extend(sorted(set(keep)))
    return X[feats].copy(), feats

# ------------------------------- Models ---------------------------------------

def _lgbm_regressor() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.035,
        num_leaves=127,
        min_data_in_leaf=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

def _train_lgbm_p90(train_rows: pd.DataFrame, Xtr: pd.DataFrame) -> Tuple[lgb.LGBMRegressor, Dict[str, Any]]:
    """
    train_rows: TRAIN subset of the original dataframe (has season/date_played/gw_orig and y_saves_p90).
    Xtr: feature matrix aligned to train_rows.index
    """
    y = train_rows["y_saves_p90"]
    fi, vi = _tail_index(train_rows, frac=0.15)
    model = _lgbm_regressor()
    model.fit(
        Xtr.loc[fi], y.loc[fi],
        eval_set=[(Xtr.loc[vi], y.loc[vi])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)]
    )
    info: Dict[str, Any] = {"best_iteration": int(getattr(model, "best_iteration_", model.n_estimators))}
    try:
        pred_val = np.clip(model.predict(Xtr.loc[vi]), 0, None)
        info["val_mae_p90"] = float(mean_absolute_error(y.loc[vi], pred_val))
    except Exception:
        pass
    return model, info

def _train_poisson(Xtr: pd.DataFrame, ytr: np.ndarray) -> Tuple[TweedieRegressor, SimpleImputer]:
    imp = SimpleImputer(strategy="median")
    Xn = imp.fit_transform(Xtr)
    glm = TweedieRegressor(power=1.0, link="log", alpha=5e-4, max_iter=5000, tol=1e-6)
    glm.fit(Xn, ytr)
    return glm, imp

# ------------------------------- Helpers --------------------------------------

def _sha1_of_list(xs: List[str]) -> str:
    h = hashlib.sha1()
    for s in xs:
        h.update(s.encode("utf-8")); h.update(b"|")
    return h.hexdigest()

# ------------------------------- Main -----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True,
                    help="Comma-separated seasons; last is TEST (e.g. 2022-2023,2023-2024,2024-2025)")
    ap.add_argument("--first-test-gw", type=int, default=26)

    ap.add_argument("--features-root", type=Path, default=Path("data/processed/registry/features"))
    ap.add_argument("--form-version", required=True)

    ap.add_argument("--na-thresh", type=float, default=0.70)
    ap.add_argument("--poisson-head", action="store_true")

    ap.add_argument("--minutes-preds", type=Path, help="expected_minutes.csv from minutes model")
    ap.add_argument("--require-pred-minutes", action="store_true", help="Fail if pred_minutes missing for any TEST row")

    ap.add_argument("--model-out", type=Path, default=Path("data/models/saves"))
    ap.add_argument("--bump-version", action="store_true")
    ap.add_argument("--version-tag", type=str, default="")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    test_season = seasons[-1]

    latest_dir = args.model_out; latest_dir.mkdir(parents=True, exist_ok=True)
    version_dir, version_name = _ensure_version_dirs(latest_dir, bump=args.bump_version, tag=(args.version_tag or None))
    logging.info("Writing artifacts to %s (versioned: %s) and %s (latest).", version_dir, version_name, latest_dir)

    # ---- Load & prep (played matches only) ----
    df = _load_players(args.features_root, args.form_version, seasons)
    team_form = _load_team_form(args.features_root, args.form_version, seasons)
    df = _merge_team_z(df, team_form)

    # Features
    X_all, feat_cols = _build_features(df, na_thresh=args.na_thresh)
    feat_sha = _sha1_of_list(list(X_all.columns))

    # Targets per-90 for TRAIN (true minutes)
    m = df["minutes"].fillna(0).clip(lower=0)
    m90 = (m / 90.0).replace(0, np.nan)
    df["y_saves_p90"] = (df["saves"] / m90).astype(float)

    # Split
    train_df, test_df = _chrono_split(df, seasons, args.first_test_gw)
    test_df = test_df.copy()
    test_df["rid"] = np.arange(len(test_df))  # immutable identity for TEST rows

    # TRAIN rows: finite per-90 label
    train_rows = train_df[np.isfinite(train_df["y_saves_p90"])].copy()
    if train_rows.empty:
        raise ValueError("No valid GK training rows (minutes>0 and saves present).")

    Xtr = X_all.loc[train_rows.index]
    model_lgb, lgb_info = _train_lgbm_p90(train_rows, Xtr)

    # Optional Poisson head
    glm = None; imputer = None
    if args.poisson_head:
        ytr = train_rows["y_saves_p90"].to_numpy()
        glm, imputer = _train_poisson(Xtr, ytr)

    # ---- Build 'out' from TEST + minutes ----
    key = ["season","gw_orig","date_played","player_id"]
    out = test_df[["season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes","rid"]].copy()

    em = _load_expected_minutes(args.minutes_preds)
    if args.require_pred_minutes and em is None:
        raise ValueError("--require-pred-minutes was set but --minutes-preds is missing/invalid.")
    if em is not None:
        out = out.merge(em, on=key, how="left", validate="many_to_one")
        if "pred_minutes" not in out.columns:
            raise ValueError("expected_minutes is missing 'pred_minutes'.")
        miss_min = out["pred_minutes"].isna()
        if miss_min.any():
            miss_df = out.loc[miss_min, key + ["team_id","player","pos","minutes"]]
            for target in (latest_dir, version_dir):
                (target / "artifacts").mkdir(parents=True, exist_ok=True)
                miss_df.to_csv(target / "artifacts" / "missing_pred_minutes.csv", index=False)
            msg = f"{int(miss_min.sum())}/{len(out)} TEST rows lack pred_minutes. See artifacts/missing_pred_minutes.csv."
            if args.require_pred_minutes:
                raise ValueError(msg)
            logging.warning(msg)
        out = out.loc[~miss_min].reset_index(drop=True)
    else:
        # Fallback: use observed minutes to scale if minutes preds not provided
        out["pred_minutes"] = out["minutes"].astype(float)

    # Merge true labels for TEST and drop any unplayed (defensive)
    truth = test_df.set_index("rid")
    out["saves_true"]   = truth.loc[out["rid"], "saves"].to_numpy()
    out["minutes_true"] = truth.loc[out["rid"], "minutes"].to_numpy()
    keep_mask = np.isfinite(out["saves_true"].to_numpy())
    if (~keep_mask).any():
        logging.warning("Dropping %d TEST rows without true saves label (unplayed).", int((~keep_mask).sum()))
    out = out.loc[keep_mask].reset_index(drop=True)

    # ---- Align predictions to FINAL out rows (rid -> original index) ----
    rid_to_orig = (
        test_df.reset_index()  # adds 'index' = original row index
               .rename(columns={"index": "orig_index"})
               .set_index("rid")["orig_index"]
    )
    orig_idx = rid_to_orig.loc[out["rid"]].to_numpy()
    Xte_aligned = X_all.loc[orig_idx]

    # Predict per-90
    p90_mean = np.clip(model_lgb.predict(Xte_aligned), 0, None)
    if args.poisson_head:
        Xte_np = imputer.transform(Xte_aligned)
        p90_pois = np.clip(glm.predict(Xte_np), 0, None)
    else:
        p90_pois = np.full(len(out), np.nan)

    # Scale to per-match using pred_minutes
    scale = out["pred_minutes"].to_numpy() / 90.0
    pred_saves_mean = p90_mean * scale
    pred_saves_pois = p90_pois * scale if args.poisson_head else np.full(len(out), np.nan)

    # ---- Metrics ----
    y_true = out["saves_true"].to_numpy()
    # p90 truth from observed minutes (avoid mixing with minutes model)
    with np.errstate(divide="ignore", invalid="ignore"):
        obs_scale = np.where(out["minutes_true"].to_numpy() > 0, out["minutes_true"].to_numpy() / 90.0, np.nan)
        y_p90_true = np.where(obs_scale > 0, y_true / obs_scale, np.nan)
    p90_mask = np.isfinite(y_p90_true)

    metrics: Dict[str, Any] = {
        "code_version": CODE_VERSION,
        "n_train": int(len(Xtr)),
        "n_test_rows_written": int(len(out)),
        "poisson_head": bool(args.poisson_head),
        "require_pred_minutes": bool(args.require_pred_minutes),
        "test_season": test_season,
        "first_test_gw": int(args.first_test_gw),
        "features_sha1": feat_sha,
    }
    if p90_mask.any():
        metrics["mae_p90"] = float(mean_absolute_error(y_p90_true[p90_mask], p90_mean[p90_mask]))
        if args.poisson_head and np.isfinite(p90_pois).any():
            metrics["mae_p90_pois"] = float(mean_absolute_error(y_p90_true[p90_mask], p90_pois[p90_mask]))

    # Match-level MAEs
    pred_match_actualmins = p90_mean * np.where(out["minutes_true"].to_numpy() > 0, out["minutes_true"].to_numpy() / 90.0, 0.0)
    metrics["mae_match_actualmins_meanHead"] = float(mean_absolute_error(y_true, pred_match_actualmins))
    metrics["mae_match_predmins_meanHead"]   = float(mean_absolute_error(y_true, pred_saves_mean))
    if args.poisson_head and np.isfinite(pred_saves_pois).any():
        pred_match_actualmins_pois = p90_pois * np.where(out["minutes_true"].to_numpy() > 0, out["minutes_true"].to_numpy() / 90.0, 0.0)
        metrics["mae_match_actualmins_poisHead"] = float(mean_absolute_error(y_true, pred_match_actualmins_pois))
        metrics["mae_match_predmins_poisHead"]   = float(mean_absolute_error(y_true, pred_saves_pois))

    # ---- Assemble output (no point predictions) ----
    out["pred_saves_p90_mean"]    = p90_mean
    out["pred_saves_mean"]        = pred_saves_mean
    out["pred_saves_p90_poisson"] = p90_pois
    out["pred_saves_poisson"]     = pred_saves_pois

    cols = [
        "season","gw_orig","date_played","player_id","team_id","player","pos","venue",
        "minutes_true","pred_minutes","saves_true",
        "pred_saves_p90_mean","pred_saves_mean",
        "pred_saves_p90_poisson","pred_saves_poisson"
    ]
    out = out[cols].copy()

    # ---- Persist ----
    for target in (latest_dir, version_dir):
        (target / "artifacts").mkdir(parents=True, exist_ok=True)
        (target / "models").mkdir(parents=True, exist_ok=True)
        out.to_csv(target / "saves_predictions.csv", index=False)
        try:
            fi = pd.DataFrame({"feature": list(X_all.columns), "importance": model_lgb.feature_importances_})
            fi.to_csv(target / "artifacts" / "feature_importances.csv", index=False)
        except Exception:
            pass
        (target / "artifacts" / "features_used.txt").write_text("\n".join(list(X_all.columns)), encoding="utf-8")
        (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _write_meta(target, args)

    # Save models
    try:
        model_lgb.booster_.save_model(str(latest_dir / "models" / "lgbm_saves_p90.txt"))
        model_lgb.booster_.save_model(str(version_dir / "models" / "lgbm_saves_p90.txt"))
    except Exception:
        pass
    if args.poisson_head and (glm is not None) and (imputer is not None):
        joblib.dump(glm, latest_dir / "models" / "poisson_saves_p90.joblib")
        joblib.dump(imputer, latest_dir / "models" / "poisson_imputer.joblib")
        joblib.dump(glm, version_dir / "models" / "poisson_saves_p90.joblib")
        joblib.dump(imputer, version_dir / "models" / "poisson_imputer.joblib")

    logging.info(json.dumps(metrics, indent=2))
    logging.info("Artifacts written to %s (latest) and %s (versioned: %s).", latest_dir, version_dir, version_name)

if __name__ == "__main__":
    main()
