#!/usr/bin/env python3
"""
saves_model_builder.py – v3.3

Goalkeepers' saves model.

Fixes / features:
  • Robust minutes merge (dynamic keys, normalized).
  • No KeyError if minutes CSV lacks prob_* columns.
  • Correct opponent feature: opp_att_z_venue = opponent's attack @ their venue.
  • Team z features auto-built from team_form when needed.
  • LGBM mean head for saves per-90, optional Poisson GLM head.
  • Uses expected minutes for the particular game on TEST to convert per-90 -> per-match.
  • Optional dropping of TEST rows missing minutes.
  • Drops minutes==0 from training target to avoid DNP bias.

Outputs:
  data/models/saves/<model_version>/models/
    - lgbm_saves_p90.txt
    - poisson_saves_p90.joblib (optional)
    - poisson_imputer.joblib (optional)
  data/models/saves/<model_version>/features_used.txt
  data/models/saves/<model_version>/predictions/saves_predictions.csv
"""

from __future__ import annotations
import argparse
import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import TweedieRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import joblib


# ───────────────────────────── IO helpers ─────────────────────────────

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
    need = {
        "season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes",
        "saves"
    }
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"players_form missing required columns: {miss}")
    return df


def _harmonize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date_played" in out.columns:
        out["date_played"] = pd.to_datetime(out["date_played"], errors="coerce").dt.date
    if "gw_orig" in out.columns:
        out["gw_orig"] = pd.to_numeric(out["gw_orig"], errors="coerce").astype("Int64")
    if "gw_played" in out.columns:
        out["gw_played"] = pd.to_numeric(out["gw_played"], errors="coerce").astype("Int64")
    if "season" in out.columns:
        out["season"] = out["season"].astype(str)
    if "player_id" in out.columns:
        out["player_id"] = out["player_id"].astype(str).str.strip().str.lower()
    if "team_id" in out.columns:
        out["team_id"] = out["team_id"].astype(str).str.strip().str.lower()
    if "game_id" in out.columns:
        out["game_id"] = out["game_id"].astype(str).str.strip().str.lower()
    return out


def _load_minutes_predictions(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.is_file():
        logging.warning("minutes_preds file not found at %s – skipping minutes merge", path)
        return None
    mp = pd.read_csv(path, parse_dates=["date_played"])
    # required minimal columns for current pipeline
    need = ["season","gw_orig","date_played","player_id","pred_exp_minutes"]
    optional = [
        "team_id","gw_played","game_id",
        "prob_played1_cal","prob_played1_raw",
        "prob_played60_cal","prob_played60_raw",
        "pred_exp_minutes_med","pred_minutes_q0_1","pred_minutes_q0_5","pred_minutes_q0_9"
    ]
    cols = [c for c in (need + optional) if c in mp.columns]
    if not set(need).issubset(cols):
        missing = set(need) - set(cols)
        logging.warning("minutes_preds missing columns %s – skipping", missing)
        return None
    mp = mp[cols].copy()
    mp = _harmonize_keys(mp)
    return mp


# ───────────────────────────── team z merge ─────────────────────────────

def _merge_team_z(df_players: pd.DataFrame, team_form_dir: Path, version: str) -> pd.DataFrame:
    """
    Add team_def_z_venue (our defense @ our venue) and
    opp_att_z_venue (opponent attack @ their venue), built from *_roll_z columns.
    """
    seasons = sorted(df_players["season"].astype(str).unique())
    frames = []
    for s in seasons:
        fp = team_form_dir / version / s / "team_form.csv"
        if not fp.is_file():
            logging.warning("team_form missing for %s", s)
            continue
        tf = pd.read_csv(fp, parse_dates=["date_played"])
        tf["season"] = s
        frames.append(tf)

    if not frames:
        logging.warning("No team_form files loaded – filling team/opp z with NaN")
        df_players["team_def_z_venue"] = np.nan
        df_players["opp_att_z_venue"] = np.nan
        return df_players

    tf = pd.concat(frames, ignore_index=True)

    needed = {
        "season","gw_orig","team_id","venue","home_id","away_id",
        "def_xga_home_roll_z","def_xga_away_roll_z",
        "att_xg_home_roll_z","att_xg_away_roll_z",
    }
    missing = needed - set(tf.columns)
    if missing:
        logging.warning("team_form lacks z inputs (%s); filling with NaN", sorted(missing))
        df_players["team_def_z_venue"] = np.nan
        df_players["opp_att_z_venue"] = np.nan
        return df_players

    tf = tf[list(needed)].copy()

    # Our defense at our own venue
    tf["def_z_at_venue"] = np.where(
        tf["venue"].eq("Home"), tf["def_xga_home_roll_z"], tf["def_xga_away_roll_z"]
    )
    # Each team's attack at its own venue
    tf["att_z_at_venue"] = np.where(
        tf["venue"].eq("Home"), tf["att_xg_home_roll_z"], tf["att_xg_away_roll_z"]
    )
    # Opponent id per (season, gw_orig, team_id)
    tf["opp_team_id"] = np.where(
        tf["team_id"].astype(str).str.lower().eq(tf["home_id"].astype(str).str.lower()),
        tf["away_id"], tf["home_id"]
    )

    # Lookups
    def_lu = (
        tf[["season","gw_orig","team_id","def_z_at_venue"]]
        .drop_duplicates()
        .rename(columns={"def_z_at_venue":"team_def_z_venue"})
    )
    att_lu = (
        tf[["season","gw_orig","team_id","att_z_at_venue"]]
        .drop_duplicates()
        .rename(columns={"team_id":"opp_team_id","att_z_at_venue":"opp_att_z_venue"})
    )
    opp_lu = tf[["season","gw_orig","team_id","opp_team_id"]].drop_duplicates()

    out = _harmonize_keys(df_players)
    opp_lu = _harmonize_keys(opp_lu)
    def_lu = _harmonize_keys(def_lu)
    att_lu = _harmonize_keys(att_lu)

    # Attach opp_team_id to player rows (by our team_id)
    out = out.merge(opp_lu, on=["season","gw_orig","team_id"], how="left")
    # Merge our def z at venue
    out = out.merge(def_lu, on=["season","gw_orig","team_id"], how="left")
    # Merge opponent attack z at their venue
    out = out.merge(att_lu, on=["season","gw_orig","opp_team_id"], how="left")

    for c in ["team_def_z_venue","opp_att_z_venue"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    logging.info("team_form: built z cols -> team_def_z_venue (ours@venue), opp_att_z_venue (opp@venue)")
    return out


# ───────────────────────────── dataset / features ─────────────────────────────

def _time_split_last_n(df: pd.DataFrame, test_season: str, last_n_gws: int) -> Tuple[pd.Index, pd.Index]:
    gws = sorted(df.loc[df["season"] == test_season, "gw_orig"].dropna().unique())
    if not gws:
        raise ValueError(f"No gw_orig found for season {test_season}")
    test_gws = set(gws[-last_n_gws:])
    test_idx = df.index[(df["season"] == test_season) & (df["gw_orig"].isin(test_gws))]
    train_idx = df.index.difference(test_idx)
    if len(test_idx) == 0:
        raise ValueError("Test split produced 0 rows – check season/last-n")
    return train_idx, test_idx


def _build_features(df: pd.DataFrame, na_thresh: float) -> Tuple[pd.DataFrame, List[str]]:
    feats: List[str] = []
    # Venue indicator
    df = df.copy()
    df["venue_bin"] = (df["venue"].astype(str) == "Home").astype(int); feats.append("venue_bin")

    # Team/opponent contextual z’s (added by merge)
    for c in ["team_def_z_venue","opp_att_z_venue"]:
        if c in df.columns:
            feats.append(c)

    # Rolling candidates: keep only columns that exist and have enough coverage
    roll_candidates = [
        # raw per-90 rolls
        "gk_saves_p90_roll","gk_saves_p90_home_roll","gk_saves_p90_away_roll",
        "gk_sot_against_p90_roll","gk_sot_against_p90_home_roll","gk_sot_against_p90_away_roll",
        # z-scored rolls (if present)
        "gk_saves_p90_roll_z","gk_saves_p90_home_roll_z","gk_saves_p90_away_roll_z",
        "gk_sot_against_p90_roll_z","gk_sot_against_p90_home_roll_z","gk_sot_against_p90_away_roll_z",
    ]
    keep_roll = [c for c in roll_candidates if c in df.columns and df[c].notna().mean() >= na_thresh]
    feats.extend(sorted(set(keep_roll)))

    X = df[feats].copy()
    return X, feats


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


def _train_glm_poisson(X: np.ndarray, y: np.ndarray) -> TweedieRegressor:
    model = TweedieRegressor(power=1.0, link="log", alpha=0.0005, max_iter=5000, tol=1e-6)
    model.fit(X, y)
    return model


# ───────────────────────────── minutes merge helpers ─────────────────────────────

def _best_join_keys(left: pd.DataFrame, right: pd.DataFrame):
    """
    Prefer strongest keys if available, else fallback to (season, gw_orig, date_played, player_id).
    """
    cands = [
        ["season","game_id","player_id"],
        ["season","gw_played","player_id"],
        ["season","gw_orig","date_played","player_id"],
    ]
    for ks in cands:
        if all(k in left.columns for k in ks) and all(k in right.columns for k in ks):
            return ks
    raise KeyError("No compatible join key between dump and minutes predictions")


# ───────────────────────────── main ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--version", required=True)
    ap.add_argument("--team-form-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--test-season", required=True)
    ap.add_argument("--test-last-n", type=int, default=10)
    ap.add_argument("--na-thresh", type=float, default=0.70)
    ap.add_argument("--minutes-preds", type=Path, help="path to minutes_predictions.csv")
    ap.add_argument("--drop-missing-minutes", action="store_true", help="drop TEST rows without pred_exp_minutes")
    ap.add_argument("--poisson-heads", action="store_true")
    ap.add_argument("--models-out", type=Path, default=Path("data/models/saves"))
    ap.add_argument("--model-version", default="v3")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    seasons = _find_seasons(args.features_dir, args.version)
    logging.info("Found seasons: %s", ", ".join(seasons))

    df_all = _load_players(args.features_dir, args.version, seasons)
    df_all = df_all[df_all["pos"] == "GK"].copy()
    df_all = _harmonize_keys(df_all)

    # team z merge (adds team_def_z_venue, opp_att_z_venue)
    df_all = _merge_team_z(df_all, args.team_form_dir, args.version)

    # split
    train_idx, test_idx = _time_split_last_n(df_all, args.test_season, args.test_last_n)
    gk_train = df_all.index.isin(train_idx)
    gk_test  = df_all.index.isin(test_idx)

    # features
    X_all, feat_cols = _build_features(df_all, na_thresh=args.na_thresh)
    Xtr_df = X_all.loc[gk_train].copy()
    Xte_df = X_all.loc[gk_test].copy()

    # targets (per-90), avoid DNP bias by training only minutes > 0
    m = df_all["minutes"].fillna(0).clip(lower=0)
    train_mask = gk_train & (m > 0)
    Xtr_df = X_all.loc[train_mask].copy()
    ytr_p90 = (df_all.loc[train_mask, "saves"] / (m.loc[train_mask] / 90.0)).to_numpy()

    # Train mean head
    model_lgb = _lgbm_reg()
    model_lgb.fit(Xtr_df, ytr_p90)

    # Optional Poisson head
    model_pois = None
    imputer = None
    if args.poisson_heads:
        imputer = SimpleImputer(strategy="median")
        Xtr_np = imputer.fit_transform(Xtr_df)
        model_pois = _train_glm_poisson(Xtr_np, ytr_p90)

    # Prepare TEST dump + minutes merge
    dump_cols = ["season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes"]
    dump = df_all.loc[gk_test, dump_cols].copy()
    dump = _harmonize_keys(dump)

    # Merge expected minutes for the particular game
    mp = _load_minutes_predictions(args.minutes_preds)
    if mp is not None:
        left = dump.copy()
        right = mp.drop_duplicates()
        join_keys = _best_join_keys(left, right)

        select_cols = join_keys + ["pred_exp_minutes"]
        for c in ("prob_played1_cal","prob_played1_raw","prob_played60_cal","prob_played60_raw"):
            if c in right.columns and c not in select_cols:
                select_cols.append(c)

        merged = left.merge(right[select_cols], on=join_keys, how="left")
        dump = merged

        # appearance probabilities are logged for reference but NOT used in scaling
        for cand in ("prob_played1_cal","prob_played60_cal","prob_played1_raw","prob_played60_raw"):
            if cand in dump.columns:
                dump["p_appear"] = pd.to_numeric(dump[cand], errors="coerce")
                break

        miss_mask = dump["pred_exp_minutes"].isna()
        n_miss = int(miss_mask.sum())
        if n_miss:
            logging.warning(
                "Minutes merge: %d/%d GK test rows missing pred_exp_minutes after join.",
                n_miss, len(dump)
            )
            logging.warning(
                "Sample missing keys (up to 5):\n%s",
                dump.loc[miss_mask, join_keys].head(5).to_string(index=False)
            )
    else:
        dump["pred_exp_minutes"] = np.nan

    # Optionally drop rows without minutes and re-align Xte_df
    if args.drop_missing_minutes:
        before = len(dump)
        keep_mask = dump["pred_exp_minutes"].notna()
        dropped = before - int(keep_mask.sum())
        if dropped:
            logging.warning("Dropping %d/%d GK test rows with missing predicted minutes.", dropped, before)
        # Xte_df and dump are aligned row-wise (both built from gk_test mask)
        Xte_df = Xte_df.loc[keep_mask.values].copy()
        dump   = dump.loc[keep_mask].copy()
    else:
        dump["pred_exp_minutes"] = dump["pred_exp_minutes"].fillna(0.0)

    # If nothing left, still save artifacts & an empty predictions file
    if Xte_df.shape[0] == 0 or dump.shape[0] == 0:
        logging.warning("No GK test rows remain after minutes handling. Saving models and writing empty predictions.")
        out_base = args.models_out / args.model_version
        (out_base / "models").mkdir(parents=True, exist_ok=True)
        (out_base / "predictions").mkdir(parents=True, exist_ok=True)
        model_lgb.booster_.save_model(out_base / "models" / "lgbm_saves_p90.txt")
        if args.poisson_heads and model_pois is not None and imputer is not None:
            joblib.dump(model_pois, out_base / "models" / "poisson_saves_p90.joblib")
            joblib.dump(imputer,   out_base / "models" / "poisson_imputer.joblib")
        (out_base / "features_used.txt").write_text("\n".join(feat_cols), encoding="utf-8")
        empty_cols = dump_cols + [
            "pred_exp_minutes",
            "pred_saves_p90_mean","pred_saves_mean","exp_save_points_mean"
        ]
        if args.poisson_heads:
            empty_cols += ["pred_saves_p90_poisson","pred_saves_poisson","exp_save_points_poisson"]
        pd.DataFrame(columns=empty_cols).to_csv(out_base / "predictions" / "saves_predictions.csv", index=False)
        logging.info("Models & predictions saved to %s", out_base.resolve())
        return

    # Predict per-90
    gk_p90_mean = model_lgb.predict(Xte_df)
    gk_p90_mean = np.clip(gk_p90_mean, 0, None)

    # per-match scaling by expected minutes for the game (no extra p_appear multiplier)
    scale = (pd.to_numeric(dump["pred_exp_minutes"], errors="coerce").fillna(0.0).to_numpy() / 90.0)
    pred_saves_mean = gk_p90_mean * scale
    exp_save_points_mean = pred_saves_mean / 3.0  # FPL rule: 1pt per 3 saves (expected)

    # Optional Poisson head
    if args.poisson_heads and model_pois is not None:
        Xte_np = imputer.transform(Xte_df)
        gk_p90_pois = model_pois.predict(Xte_np)
        gk_p90_pois = np.clip(gk_p90_pois, 0, None)
        pred_saves_pois = gk_p90_pois * scale
        exp_save_points_pois = pred_saves_pois / 3.0
    else:
        gk_p90_pois = pred_saves_pois = exp_save_points_pois = None

    # Evaluate (optional)
    yte = df_all.loc[gk_test].loc[Xte_df.index, "saves"].to_numpy()
    try:
        mae_mean = mean_absolute_error(yte, pred_saves_mean)
        logging.info("Test MAE (saves, mean head): %.4f", mae_mean)
    except Exception:
        pass
    if args.poisson_heads and pred_saves_pois is not None:
        try:
            mae_pois = mean_absolute_error(yte, pred_saves_pois)
            logging.info("Test MAE (saves, poisson): %.4f", mae_pois)
        except Exception:
            pass

    # Assemble output
    out = dump.copy()
    out["pred_saves_p90_mean"] = gk_p90_mean
    out["pred_saves_mean"] = pred_saves_mean
    out["exp_save_points_mean"] = exp_save_points_mean
    if args.poisson_heads and gk_p90_pois is not None:
        out["pred_saves_p90_poisson"] = gk_p90_pois
        out["pred_saves_poisson"] = pred_saves_pois
        out["exp_save_points_poisson"] = exp_save_points_pois

    # Save models + predictions
    out_base = args.models_out / args.model_version
    (out_base / "models").mkdir(parents=True, exist_ok=True)
    (out_base / "predictions").mkdir(parents=True, exist_ok=True)

    model_lgb.booster_.save_model(out_base / "models" / "lgbm_saves_p90.txt")
    if args.poisson_heads and model_pois is not None and imputer is not None:
        joblib.dump(model_pois, out_base / "models" / "poisson_saves_p90.joblib")
        joblib.dump(imputer,   out_base / "models" / "poisson_imputer.joblib")

    (out_base / "features_used.txt").write_text("\n".join(feat_cols), encoding="utf-8")

    fp = out_base / "predictions" / "saves_predictions.csv"
    # Include useful ordering keys if present
    sort_keys = [k for k in ["season","gw_orig","date_played","team_id","player_id"] if k in out.columns]
    out.sort_values(sort_keys).to_csv(fp, index=False)
    logging.info("Wrote predictions to %s", fp.resolve())
    logging.info("Models & predictions saved to %s", out_base.resolve())


if __name__ == "__main__":
    main()
