#!/usr/bin/env python3
r"""
goals_assists_model_builder.py – v10

Adds future-season inference:
• --predict-season [--predict-gws 1,2,3] to emit predictions for any season/GWs (no labels required).
• Training uses all labeled rows; prediction target set is separate and can be unlabeled.
• Robust join to minutes with coverage CSV; alignment via rid.

Retains / improves:
• Non-negative clamp on per-90 outputs (LGBM & Poisson).
• Optional Poisson heads (TweedieRegressor, power=1, log link).
• Uses predicted minutes to convert per-90 -> per-match on TARGET (eval or predict).
• Saves models and artifacts (feature lists, importances).
• Shot features (shots/sot per-90 + 5-match rolling overall & H/A).
• Team context merge: team_att_z_venue, opp_def_z_venue.

Outputs under data/models/goals_assists/<model_version>/predictions/goals_assists_predictions.csv:
  season, gw_orig, date_played, player_id, team_id, player, pos, venue, minutes,
  pred_exp_minutes,
  pred_goals_p90_mean,   pred_assists_p90_mean,
  pred_goals_mean,       pred_assists_mean,
  pred_goals_p90_poisson,pred_assists_p90_poisson,
  pred_goals_poisson,    pred_assists_poisson,
  team_att_z_venue, opp_def_z_venue
"""

from __future__ import annotations
import argparse, logging, re, json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import TweedieRegressor  # Poisson heads
import joblib

# ───────────── IO helpers ─────────────
def _find_seasons(features_dir: Path, version: str) -> List[str]:
    base = features_dir / version
    if not base.exists():
        raise FileNotFoundError(f"{base} does not exist")
    dirs = [d for d in base.iterdir() if d.is_dir()]
    seasons = [d.name for d in dirs if re.match(r"^\d{4}-\d{4}$", d.name)]
    if not seasons:
        seasons = [d.name for d in dirs if (d / "players_form.csv").is_file()]
    if not seasons:
        raise FileNotFoundError(f"No season folders found under {base}")
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
        "season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes"
    }
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"players_form missing required columns: {miss}")

    # Targets may be absent (future season) -> create if missing
    if "gls" not in df.columns: df["gls"] = np.nan
    if "ast" not in df.columns: df["ast"] = np.nan

    # Unique row id for robust alignment
    df["rid"] = df.index.astype(int)
    return df


def _load_minutes_predictions(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        logging.warning("No minutes predictions path provided.")
        return None
    if not path.is_file():
        logging.warning("minutes_preds file not found at %s – skipping minutes join.", path)
        return None

    mp = pd.read_csv(path, parse_dates=["date_played"])
    need = {"season","gw_orig","date_played","player_id","pred_exp_minutes"}
    missing = need - set(mp.columns)
    if missing:
        logging.warning("minutes_preds missing columns %s – skipping minutes join.", missing)
        return None

    # Deduplicate by unique key
    key = ["season","gw_orig","date_played","player_id"]
    mp = mp.sort_values(key).drop_duplicates(subset=key, keep="last")
    return mp[key + ["pred_exp_minutes"]].copy()

# ───────────── team z merge ─────────────
def _try_merge(df: pd.DataFrame, t: pd.DataFrame, keys: List[str]) -> Optional[pd.DataFrame]:
    if not set(keys).issubset(df.columns) or not set(keys).issubset(t.columns):
        return None
    try:
        return df.merge(t, on=keys, how="left", suffixes=("","_tm"))
    except Exception:
        return None


def _merge_team_z(df_players: pd.DataFrame, team_form_dir: Path, version: str) -> pd.DataFrame:
    t_frames = []
    for s in sorted({*df_players["season"].unique()}):
        fp = team_form_dir / version / s / "team_form.csv"
        if not fp.is_file():
            logging.warning("team_form missing for %s", s)
            continue
        tf = pd.read_csv(fp, parse_dates=["date_played"])
        tf["season"] = s
        t_frames.append(tf)

    if not t_frames:
        logging.warning("No team_form files loaded – skipping team z merge")
        df_players["team_att_z_venue"] = np.nan
        df_players["opp_def_z_venue"] = np.nan
        return df_players

    tf = pd.concat(t_frames, ignore_index=True)

    prefer_cols = ["team_att_z_venue","opp_def_z_venue"]
    has_prepared = all(c in tf.columns for c in prefer_cols)

    if has_prepared:
        use = ["season","gw_orig","team_id","team_att_z_venue","opp_def_z_venue"]
        use = [c for c in use if c in tf.columns]
        t = tf[use].drop_duplicates()
    else:
        needed = ["season","gw_orig","team_id","venue",
                  "att_xg_home_roll_z","att_xg_away_roll_z",
                  "def_xga_home_roll_z","def_xga_away_roll_z"]
        if not set(needed).issubset(tf.columns):
            logging.warning("team_form has limited z columns; proceeding without team z.")
            df_players["team_att_z_venue"] = np.nan
            df_players["opp_def_z_venue"] = np.nan
            return df_players

        t = tf[needed].copy()
        t["team_att_z_venue"] = np.where(
            t["venue"] == "Home", t["att_xg_home_roll_z"], t["att_xg_away_roll_z"]
        )
        t["opp_def_z_venue"] = np.where(
            t["venue"] == "Home", t["def_xga_away_roll_z"], t["def_xga_home_roll_z"]
        )
        t = t.drop(columns=["venue","att_xg_home_roll_z","att_xg_away_roll_z",
                            "def_xga_home_roll_z","def_xga_away_roll_z"]).drop_duplicates()

    for c in ["team_att_z_venue","opp_def_z_venue"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    out = None
    for keys in (["season","gw_orig","team_id"], ["season","date_played","team_id"]):
        out = _try_merge(df_players, t, keys)
        if out is not None:
            break
    if out is None:
        logging.warning("team_form join failed – leaving team z NaN")
        df_players["team_att_z_venue"] = np.nan
        df_players["opp_def_z_venue"] = np.nan
        return df_players
    return out

# ───────────── shots wiring ─────────────
def _add_shot_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds per-90 shots and shots-on-target (sot) and 5-match rolling
    (overall + split by venue). Handles a few common column names.
    Returns (df, added_feature_names).
    """
    feature_names: List[str] = []

    # find raw cols
    shots_col = None
    for cand in ("shots", "sh"):
        if cand in df.columns:
            shots_col = cand
            break

    sot_col = None
    for cand in ("sot", "shots_on_target"):
        if cand in df.columns:
            sot_col = cand
            break

    if shots_col is None and sot_col is None:
        logging.info("No shots/sot columns found – skipping shot features.")
        return df, feature_names

    # per-90 base
    m = df["minutes"].fillna(0).clip(lower=0)
    denom = (m / 90.0).replace(0, np.nan)
    if shots_col is not None:
        df["shots_p90"] = (df[shots_col] / denom).fillna(0)
        feature_names.append("shots_p90")
    if sot_col is not None:
        df["sot_p90"] = (df[sot_col] / denom).fillna(0)
        feature_names.append("sot_p90")

    # rolling 5 by player-season (overall + H/A)
    def _roll5(s: pd.Series) -> pd.Series:
        return s.rolling(5, min_periods=1).mean()

    for base in ("shots_p90", "sot_p90"):
        if base not in df.columns:
            continue
        # overall rolling
        df[f"{base}_roll5"] = (
            df.groupby(["player_id","season"], sort=False)[base]
              .apply(_roll5)
              .reset_index(level=[0,1], drop=True)
        )
        feature_names.append(f"{base}_roll5")

        # home/away splits
        mask_h = df["venue"] == "Home"
        df[f"{base}_home_roll5"] = 0.0
        df.loc[mask_h, f"{base}_home_roll5"] = (
            df.loc[mask_h].groupby(["player_id","season"], sort=False)[base]
              .apply(_roll5)
              .reset_index(level=[0,1], drop=True)
        )
        df.loc[~mask_h, f"{base}_home_roll5"] = np.nan
        feature_names.append(f"{base}_home_roll5")

        mask_a = df["venue"] == "Away"
        df[f"{base}_away_roll5"] = 0.0
        df.loc[mask_a, f"{base}_away_roll5"] = (
            df.loc[mask_a].groupby(["player_id","season"], sort=False)[base]
              .apply(_roll5)
              .reset_index(level=[0,1], drop=True)
        )
        df.loc[~mask_a, f"{base}_away_roll5"] = np.nan
        feature_names.append(f"{base}_away_roll5")

    used = [shots_col or "-", sot_col or "-"]
    logging.info("Shot features wired. Using raw columns -> shots: %s  sot: %s", *used)
    return df, feature_names

# ───────────── dataset build ─────────────
def _time_split_last_n(df: pd.DataFrame, test_season: str, last_n_gws: int) -> Tuple[pd.Index, pd.Index]:
    gws = sorted(df.loc[df["season"] == test_season, "gw_orig"].dropna().unique())
    if not gws:
        raise ValueError(f"No gw_orig found for season {test_season}")
    test_gws = set(gws[-last_n_gws:])
    test_idx = df.index[(df["season"] == test_season) & (df["gw_orig"].isin(test_gws))]
    train_idx = df.index.difference(test_idx)
    if len(test_idx) == 0:
        raise ValueError("Test split produced 0 rows – check inputs")
    return train_idx, test_idx


def _build_inference_index(df: pd.DataFrame, season: str, gws: Optional[List[int]]) -> pd.Index:
    mask = (df["season"] == season)
    if gws is not None:
        mask &= df["gw_orig"].isin(gws)
    idx = df.index[mask]
    if len(idx) == 0:
        raise ValueError(f"No rows available to predict for season={season} gws={gws}")
    return idx


def _build_features(df: pd.DataFrame, use_z: bool, na_thresh: float) -> Tuple[pd.DataFrame, List[str]]:
    feats: List[str] = []

    # base features
    df["venue_bin"] = (df["venue"] == "Home").astype(int); feats.append("venue_bin")
    df["fdr"] = np.where(df["venue"] == "Home", df.get("fdr_home", np.nan), df.get("fdr_away", np.nan)).astype(float)
    feats.append("fdr")
    if "days_since_last" in df.columns: feats.append("days_since_last")
    if "is_active" in df.columns: feats.append("is_active")
    if "minutes" in df.columns:
        df["prev_minutes"] = df.groupby(["player_id","season"], sort=False)["minutes"].shift(1)
        feats.append("prev_minutes")

    # team z features
    for c in ["team_att_z_venue","opp_def_z_venue"]:
        if c in df.columns:
            feats.append(c)

    # positional enc
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df["pos_enc"] = enc.fit_transform(df[["pos"]]); feats.append("pos_enc")

    def _maybe(cols):
        return [c for c in cols if c in df.columns]

    # rolling form features
    roll_cols = _maybe([
        "gls_gls_p90_roll","gls_gls_p90_home_roll","gls_gls_p90_away_roll",
        "gls_npxg_p90_roll","gls_npxg_p90_home_roll","gls_npxg_p90_away_roll",
        "ast_ast_p90_roll","ast_ast_p90_home_roll","ast_ast_p90_away_roll",
        "ast_xag_p90_roll","ast_xag_p90_home_roll","ast_xag_p90_away_roll",
    ])
    if use_z:
        roll_cols += _maybe([
            "gls_gls_p90_roll_z","gls_gls_p90_home_roll_z","gls_gls_p90_away_roll_z",
            "gls_npxg_p90_roll_z","gls_npxg_p90_home_roll_z","gls_npxg_p90_away_roll_z",
            "ast_ast_p90_roll_z","ast_ast_p90_home_roll_z","ast_ast_p90_away_roll_z",
            "ast_xag_p90_roll_z","ast_xag_p90_home_roll_z","ast_xag_p90_away_roll_z",
        ])
    keep = [c for c in roll_cols if df[c].notna().mean() >= na_thresh]
    feats.extend(sorted(keep))

    # add shots features
    df, shot_feats = _add_shot_features(df)
    feats.extend([c for c in shot_feats if df[c].notna().mean() >= na_thresh])

    X = df[feats].copy()
    return X, feats

# ───────────── models ─────────────
def _lgbm_reg() -> lgb.LGBMRegressor:
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

def _train_glm_poisson(Xtr: np.ndarray, ytr: np.ndarray, max_iter: int = 5000) -> TweedieRegressor:
    # Poisson: power=1, log link; handles zeros naturally
    model = TweedieRegressor(power=1.0, alpha=0.0005, link="log", max_iter=max_iter, tol=1e-6)
    model.fit(Xtr, ytr)
    return model

# ───────────── artifacts / coverage ─────────────
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

def _save_feature_artifacts(outdir: Path,
                            features: List[str],
                            g_model: lgb.LGBMRegressor,
                            a_model: lgb.LGBMRegressor) -> None:
    art = outdir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    try:
        pd.DataFrame({"feature": features, "importance": g_model.feature_importances_}).to_csv(art / "goals_feature_importances.csv", index=False)
        pd.DataFrame({"feature": features, "importance": a_model.feature_importances_}).to_csv(art / "assists_feature_importances.csv", index=False)
    except Exception:
        pass

# ───────────── main ─────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--version", required=True, help="features version, e.g. v9")
    ap.add_argument("--team-form-dir", type=Path, default=Path("data/processed/features"), help="root where <version>/<season>/team_form.csv lives")
    ap.add_argument("--test-season", required=True)
    ap.add_argument("--test-last-n", type=int, default=10)

    ap.add_argument("--predict-season", type=str, default=None,
                    help="If set, run inference for this season (all GWs or --predict-gws).")
    ap.add_argument("--predict-gws", type=str, default=None,
                    help="Comma-separated GW list for inference, e.g., '1,2,3'.")

    ap.add_argument("--use-z", action="store_true", help="use *_roll_z and team z features if available")
    ap.add_argument("--na-thresh", type=float, default=0.70)
    ap.add_argument("--minutes-preds", type=Path, help="path to minutes_predictions.csv to use predicted minutes on TARGET")
    ap.add_argument("--poisson-heads", action="store_true", help="fit GLM Poisson per-90 heads alongside LGBM")

    ap.add_argument("--models-out", type=Path, default=Path("data/models/goals_assists"))
    ap.add_argument("--model-version", default="v10")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    seasons = _find_seasons(args.features_dir, args.version)
    logging.info("Found seasons: %s", ", ".join(seasons))

    df = _load_players(args.features_dir, args.version, seasons)

    # team z merge
    df = _merge_team_z(df, args.team_form_dir, args.version)
    used_cols = [c for c in ("team_att_z_venue","opp_def_z_venue") if c in df.columns]
    logging.info("team_form: using columns -> %s", ", ".join(used_cols) if used_cols else "(none)")

    # stable sort + rid
    df = df.sort_values(["season","date_played","gw_orig","player_id"]).reset_index(drop=True)
    df["rid"] = df.index.astype(int)

    # build features over the whole DF (we'll slice later)
    X, feat_cols = _build_features(df, use_z=args.use_z, na_thresh=args.na_thresh)

    # targets – per-90 (can be NaN for unlabeled rows)
    m = df["minutes"].fillna(0).clip(lower=0)
    m90 = (m / 90.0).replace(0, np.nan)
    df["y_goals_p90"]   = (df["gls"] / m90).astype(float)
    df["y_assists_p90"] = (df["ast"] / m90).astype(float)

    # choose TRAIN and TARGET sets
    if args.predict_season:
        # TRAIN: everything except the predict window (we'll filter to rows with labels)
        train_idx, _ = _time_split_last_n(df, args.test_season, args.test_last_n)
        df_train = df.loc[train_idx].copy()
        # TARGET for inference
        pred_gws = None if not args.predict_gws else [int(x) for x in args.predict_gws.split(",")]
        target_idx = _build_inference_index(df, args.predict_season, pred_gws)
        df_target = df.loc[target_idx].copy()
        logging.info("Mode: PREDICT-ONLY for season %s%s",
                     args.predict_season,
                     f" GWs {args.predict_gws}" if args.predict_gws else " (all GWs)")
    else:
        train_idx, test_idx = _time_split_last_n(df, args.test_season, args.test_last_n)
        df_train = df.loc[train_idx].copy()
        df_target = df.loc[test_idx].copy()
        logging.info("Mode: EVAL (last %d GWs of %s)", args.test_last_n, args.test_season)

    # TRAIN rows must have labels
    y_g_tr = df_train["y_goals_p90"].to_numpy()
    y_a_tr = df_train["y_assists_p90"].to_numpy()
    tr_mask = np.isfinite(y_g_tr) & np.isfinite(y_a_tr)
    if tr_mask.sum() == 0:
        raise ValueError("No valid training rows with gls/ast labels in the training split.")
    Xtr = X.loc[df_train.index].iloc[tr_mask].copy()
    ytr_g_p90 = y_g_tr[tr_mask]
    ytr_a_p90 = y_a_tr[tr_mask]

    # Minutes predictions for TARGET (eval or predict)
    mp = _load_minutes_predictions(args.minutes_preds)
    dump_cols = ["season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes","rid"]
    dump = df_target[dump_cols].copy()
    dump["pred_exp_minutes"] = np.nan

    kept_idx = df_target.index.to_numpy()
    if mp is not None and len(dump) > 0:
        key = ["season","gw_orig","date_played","player_id"]
        join = dump.merge(mp, on=key, how="left", validate="many_to_one").rename(
            columns={"pred_exp_minutes_y":"pred_exp_minutes"}
        )
        if "pred_exp_minutes_x" in join.columns:
            join.drop(columns=["pred_exp_minutes_x"], inplace=True)

        miss_mask = join["pred_exp_minutes"].isna()
        n_miss = int(miss_mask.sum())
        if n_miss:
            logging.warning("Dropping %d/%d target rows with missing predicted minutes.", n_miss, len(join))
        # keep only rows with minutes
        kept = join.loc[~miss_mask].reset_index(drop=True)
        kept_rids = kept["rid"].astype(int).to_numpy()
        dump = kept.drop(columns=["rid"]).copy()
    else:
        if mp is None:
            logging.warning("No minutes predictions provided; cannot scale per-90 to per-match.")
        kept_rids = np.array([], dtype=int)
        dump = dump.iloc[:0].drop(columns=["rid"]) if "rid" in dump.columns else dump.iloc[:0]

    # Slice TARGET features by rid alignment
    Xte = X.loc[kept_rids].copy()

    # Optional ground-truth for EVAL mode
    if not args.predict_season and len(kept_rids) > 0:
        yte_g = df.loc[kept_rids, "gls"].to_numpy()
        yte_a = df.loc[kept_rids, "ast"].to_numpy()
    else:
        yte_g = yte_a = None

    logging.info("Train rows: %d • Target rows kept: %d", len(Xtr), len(Xte))

    # LGBM mean heads (per-90), clamp >= 0
    g_lgbm = _lgbm_reg().fit(Xtr, ytr_g_p90)
    a_lgbm = _lgbm_reg().fit(Xtr, ytr_a_p90)
    g_mean_p90 = np.clip(g_lgbm.predict(Xte), 0, None) if len(Xte) else np.array([])
    a_mean_p90 = np.clip(a_lgbm.predict(Xte), 0, None) if len(Xte) else np.array([])

    # Optional Poisson (GLM) heads – per-90 (median-imputed for NaN safety)
    if args.poisson_heads and len(Xte) > 0:
        med = Xtr.median(numeric_only=True)
        Xtr_glm = Xtr.fillna(med).to_numpy()
        Xte_glm = Xte.fillna(med).to_numpy()
        g_pois_model = _train_glm_poisson(Xtr_glm, ytr_g_p90, max_iter=5000)
        a_pois_model = _train_glm_poisson(Xtr_glm, ytr_a_p90, max_iter=5000)
        g_pois_p90 = np.clip(g_pois_model.predict(Xte_glm), 0, None)
        a_pois_p90 = np.clip(a_pois_model.predict(Xte_glm), 0, None)
    else:
        g_pois_model = a_pois_model = None
        g_pois_p90 = a_pois_p90 = None

    # convert per-90 to per-match using predicted minutes on TARGET
    if len(dump) > 0:
        exp_mins = dump["pred_exp_minutes"].to_numpy()
        scale = exp_mins / 90.0
        pred_goals_mean   = g_mean_p90 * scale
        pred_assists_mean = a_mean_p90 * scale
        if g_pois_p90 is not None:
            pred_goals_pois   = g_pois_p90 * scale
            pred_assists_pois = a_pois_p90 * scale
        else:
            pred_goals_pois = pred_assists_pois = None
    else:
        pred_goals_mean = pred_assists_mean = np.array([])
        pred_goals_pois = pred_assists_pois = None

    # Evaluate MAE on per-match (actual counts) — only in eval mode
    if (not args.predict_season) and len(dump) > 0 and yte_g is not None:
        try:
            mae_g_mean = mean_absolute_error(yte_g, pred_goals_mean)
            mae_a_mean = mean_absolute_error(yte_a, pred_assists_mean)
            logging.info("Test MAE (goals, mean head):   %.4f", mae_g_mean)
            logging.info("Test MAE (assists, mean head): %.4f", mae_a_mean)
            if pred_goals_pois is not None:
                mae_g_pois = mean_absolute_error(yte_g, pred_goals_pois)
                mae_a_pois = mean_absolute_error(yte_a, pred_assists_pois)
                logging.info("Test MAE (goals, poisson):   %.4f", mae_g_pois)
                logging.info("Test MAE (assists, poisson): %.4f", mae_a_pois)
        except Exception:
            pass

    # Assemble predictions dump (add back team context)
    for c in ("team_att_z_venue","opp_def_z_venue"):
        if c in df.columns and len(kept_rids) > 0:
            dump[c] = df.loc[kept_rids, c].to_numpy()

    dump["pred_goals_p90_mean"]    = g_mean_p90
    dump["pred_assists_p90_mean"]  = a_mean_p90
    dump["pred_goals_mean"]        = pred_goals_mean
    dump["pred_assists_mean"]      = pred_assists_mean

    if g_pois_p90 is not None:
        dump["pred_goals_p90_poisson"]   = g_pois_p90
        dump["pred_assists_p90_poisson"] = a_pois_p90
        dump["pred_goals_poisson"]       = pred_goals_pois
        dump["pred_assists_poisson"]     = pred_assists_pois

    # write predictions
    outdir = args.models_out / args.model_version
    pred_dir = outdir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    fp = pred_dir / "goals_assists_predictions.csv"
    dump.to_csv(fp, index=False)
    logging.info("Wrote predictions to %s", fp.resolve())

    # coverage CSV for minutes join (use the joined frame)
    if args.minutes_preds is not None and 'join' in locals():
        _write_missing_join_csv(pred_dir, join, join["pred_exp_minutes"].isna())

    # artifacts
    _save_feature_artifacts(outdir, feat_cols, g_lgbm, a_lgbm)
    (outdir / "features_used.txt").write_text("\n".join(feat_cols), encoding="utf-8")

    # save models
    g_lgbm.booster_.save_model(str(outdir / "goals_lgbm.txt"))
    a_lgbm.booster_.save_model(str(outdir / "assists_lgbm.txt"))
    if g_pois_model is not None:
        joblib.dump(g_pois_model, outdir / "goals_poisson.joblib")
    if a_pois_model is not None:
        joblib.dump(a_pois_model, outdir / "assists_poisson.joblib")

    logging.info("Models & predictions saved to %s", outdir.resolve())


if __name__ == "__main__":
    main()
