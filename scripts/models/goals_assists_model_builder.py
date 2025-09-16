#!/usr/bin/env python3
r"""
goals_assists_model_builder.py — TRAIN/TEST ONLY — v12.8.0

Adds GK handling + true labels in CSV + probabilities:
• --skip-gk: exclude GK from TRAIN & METRICS; hard-zero GK predictions in output.
• Fill missing GK G/A labels with 0 (prevents NaNs for keepers).
• Per-position LGBM per-90 heads (+ optional GLM-Poisson per-90 heads).
• Uses expected_minutes.csv (TEST): pred_minutes (+ mixture: p_start, p_cameo, pred_start_head, pred_bench_cameo_head).
• Poisson probabilities (p_goal, p_assist) + p_return_any, optional per-position isotonic calibration.
• Minutes-style versioning (--bump-version / --version-tag).
• EWMA-based recency for shots/SOT with per-position halflife; also consumes upstream *_ewm/*_roll features.

New in v12.8.0
--------------
• Persist training medians: artifacts/features_median.json (global) and features_median_by_pos.csv (optional).
• Row-level audit: artifacts/missing_features_by_row.csv (feat coverage + list of NaN features per written TEST row).
• Feature-level audit: artifacts/feature_na_summary.csv (train vs final-test NaN rate and delta).
• Metrics: per_pos_models now reflects actual availability; added per_pos_models_poisson.

Outputs (<model-out>/goals_assists_predictions.csv) with columns:
  season,gw_orig,date_played,player_id,team_id,player,pos,venue,minutes,pred_minutes,
  goals_true,assists_true,
  team_att_z_venue,opp_def_z_venue,
  pred_goals_p90_mean,pred_assists_p90_mean,
  pred_goals_mean,pred_assists_mean,
  pred_goals_p90_poisson,pred_assists_p90_poisson,
  pred_goals_poisson,pred_assists_poisson,
  p_goal,p_assist,p_return_any
  [+ optional lambda_* if --dump-lambdas]

Metrics (<model-out>/metrics.json):
  MAEs for per-match (mean/poisson) and per-90; Brier/ECE for p_goal/p_assist; flags & counts.
"""

from __future__ import annotations
import argparse, json, logging, os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, brier_score_loss
from sklearn.linear_model import TweedieRegressor
from sklearn.isotonic import IsotonicRegression
import joblib

CODE_VERSION = "12.8.0"

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

    needed = {"season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes"}
    miss = needed - set(df.columns)
    if miss:
        raise KeyError(f"players_form missing columns: {miss}")

    # Ensure targets exist & numeric
    if "gls" not in df.columns: df["gls"] = np.nan
    if "ast" not in df.columns: df["ast"] = np.nan
    for c in ["gls","ast"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    # Fill GK NaN labels with 0 (historically ~always zero; avoids ignored rows)
    gk_mask = df["pos"].astype(str).str.upper().eq("GK")
    nan_fix = int(df.loc[gk_mask, ["gls","ast"]].isna().sum().sum())
    df.loc[gk_mask, ["gls","ast"]] = df.loc[gk_mask, ["gls","ast"]].fillna(0.0)
    if nan_fix:
        logging.info("Filled %d missing GK G/A labels with 0.", nan_fix)

    return df

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
    if team_form is None:
        players["team_att_z_venue"] = np.nan
        players["opp_def_z_venue"] = np.nan
        return players

    if {"team_att_z_venue","opp_def_z_venue"}.issubset(team_form.columns):
        t = team_form[["season","gw_orig","team_id","team_att_z_venue","opp_def_z_venue"]].drop_duplicates()
    else:
        needed = {"season","gw_orig","team_id","venue",
                  "att_xg_home_roll_z","att_xg_away_roll_z",
                  "def_xga_home_roll_z","def_xga_away_roll_z"}
        if needed.issubset(team_form.columns):
            t = team_form[list(needed)].copy()
            t["team_att_z_venue"] = np.where(t["venue"].eq("Home"), t["att_xg_home_roll_z"], t["att_xg_away_roll_z"])
            t["opp_def_z_venue"] = np.where(t["venue"].eq("Home"), t["def_xga_away_roll_z"], t["def_xga_home_roll_z"])
            t = t.drop(columns=["venue","att_xg_home_roll_z","att_xg_away_roll_z",
                                "def_xga_home_roll_z","def_xga_away_roll_z"]).drop_duplicates()
        else:
            players["team_att_z_venue"] = np.nan
            players["opp_def_z_venue"] = np.nan
            return players

    for c in ["team_att_z_venue","opp_def_z_venue"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")

    return players.merge(t, on=["season","gw_orig","team_id"], how="left")

def _load_expected_minutes(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    expected_minutes.csv from minutes model.
    Need at least: season, gw_orig, date_played, player_id, pred_minutes.
    Mixture fields (optional but preferred): p_start, p_cameo, pred_start_head, pred_bench_cameo_head
    """
    if path is None: return None
    if not path.is_file():
        logging.warning("expected_minutes file not found at %s", path)
        return None
    df = pd.read_csv(path, parse_dates=["date_played"])
    base_need = {"season","gw_orig","date_played","player_id","pred_minutes"}
    missing = base_need - set(df.columns)
    if missing:
        logging.warning("expected_minutes missing %s", missing)
        return None
    key = ["season","gw_orig","date_played","player_id"]
    df = df.sort_values(key).drop_duplicates(subset=key, keep="last")
    keep = key + ["pred_minutes","p_start","p_cameo","pred_start_head","pred_bench_cameo_head"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()

# ------------------------------- Split ----------------------------------------

def _chrono_split(df: pd.DataFrame, seasons: List[str], first_test_gw: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_season = seasons[-1]
    g = pd.to_numeric(df["gw_orig"], errors="coerce")
    train_mask = (df["season"] < test_season) | ((df["season"] == test_season) & (g < first_test_gw))
    test_mask  = (df["season"] == test_season) & (g >= first_test_gw)
    train = df.loc[train_mask].copy()
    test  = df.loc[test_mask].copy()
    if train.empty or test.empty:
        cutoff = pd.to_datetime(test["date_played"]).min()
        train = df[(df["season"] < test_season) | ((df["season"] == test_season) & (df["date_played"] < cutoff))].copy()
        test  = df[(df["season"] == test_season) & (df["date_played"] >= cutoff)].copy()
    return train, test

def _tail_index(df: pd.DataFrame, frac: float = 0.15) -> Tuple[pd.Index, pd.Index]:
    if len(df) < 10:
        return df.index, df.index
    dfo = df.sort_values(["season","date_played","gw_orig"])
    k = max(1, int(round(frac * len(dfo))))
    val_idx = dfo.index[-k:]
    fit_idx = dfo.index.difference(val_idx)
    return fit_idx, val_idx

# ------------------------------- EWMA features --------------------------------

def _prefer(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    return None

def _collect_roll_like(df: pd.DataFrame, use_z: bool) -> List[str]:
    """
    Gather upstream *_ewm or *_roll (already EWMA'd upstream) for:
    gls_gls_p90, gls_npxg_p90, ast_ast_p90, ast_xag_p90 and their _home/_away (+ _z).
    """
    bases = ["gls_gls_p90","gls_npxg_p90","ast_ast_p90","ast_xag_p90"]
    suffs = ["", "_home", "_away"]
    feats: List[str] = []
    for b in bases:
        for s in suffs:
            ch = _prefer(df, [f"{b}{s}_ewm", f"{b}{s}_roll"])
            if ch: feats.append(ch)
            if use_z:
                chz = _prefer(df, [f"{b}{s}_ewm_z", f"{b}{s}_roll_z"])
                if chz: feats.append(chz)
    out, seen = [], set()
    for f in feats:
        if f not in seen:
            out.append(f); seen.add(f)
    return out

def _parse_halflife_pos(s: str, default_hl: float) -> Dict[str, float]:
    out = { "GK": default_hl, "DEF": default_hl, "MID": default_hl, "FWD": default_hl }
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part or ":" not in part: continue
        k, v = part.split(":", 1)
        k = k.strip().upper()
        try:
            out[k] = float(v.strip())
        except Exception:
            pass
    return out

def _add_shot_ewm_by_pos(df: pd.DataFrame, hl_map: Dict[str, float], min_periods: int, adjust: bool) -> Tuple[pd.DataFrame, List[str]]:
    feats: List[str] = []
    shots = next((c for c in ("shots","sh") if c in df.columns), None)
    sot   = next((c for c in ("sot","shots_on_target") if c in df.columns), None)
    if shots is None and sot is None:
        return df, feats

    df["pos"] = df["pos"].astype(str).str.upper()
    m = df["minutes"].fillna(0).clip(lower=0)
    denom = (m / 90.0).replace(0, np.nan)
    if shots is not None: df["_shots_p90_raw"] = (df[shots] / denom)
    if sot   is not None: df["_sot_p90_raw"]   = (df[sot]   / denom)

    def _ewm_lag_group(s: pd.Series) -> pd.Series:
        pos = s.name[0] if isinstance(s.name, tuple) else None
        hl = hl_map.get(str(pos).upper(), list(hl_map.values())[0])
        return s.shift(1).ewm(halflife=hl, min_periods=min_periods, adjust=adjust).mean()

    # overall
    if "_shots_p90_raw" in df.columns:
        df["shots_p90_ewm"] = (
            df.groupby(["pos","player_id","season"], sort=False)["_shots_p90_raw"]
              .apply(_ewm_lag_group).reset_index(level=[0,1,2], drop=True)
        ); feats.append("shots_p90_ewm")
    if "_sot_p90_raw" in df.columns:
        df["sot_p90_ewm"] = (
            df.groupby(["pos","player_id","season"], sort=False)["_sot_p90_raw"]
              .apply(_ewm_lag_group).reset_index(level=[0,1,2], drop=True)
        ); feats.append("sot_p90_ewm")

    # venue splits
    for base_raw, base in (("_shots_p90_raw","shots_p90"), ("_sot_p90_raw","sot_p90")):
        if base_raw not in df.columns: continue
        mask_h = df["venue"].astype(str).str.lower().eq("home")
        mask_a = df["venue"].astype(str).str.lower().eq("away")

        df[f"{base}_home_ewm"] = np.nan
        df.loc[mask_h, f"{base}_home_ewm"] = (
            df.loc[mask_h].groupby(["pos","player_id","season"], sort=False)[base_raw]
              .apply(_ewm_lag_group).reset_index(level=[0,1,2], drop=True)
        )
        df[f"{base}_away_ewm"] = np.nan
        df.loc[mask_a, f"{base}_away_ewm"] = (
            df.loc[mask_a].groupby(["pos","player_id","season"], sort=False)[base_raw]
              .apply(_ewm_lag_group).reset_index(level=[0,1,2], drop=True)
        )
        feats += [f"{base}_home_ewm", f"{base}_away_ewm"]

    return df, feats

def _build_features(df: pd.DataFrame, use_z: bool, na_thresh: float,
                    ewm_halflife: float, ewm_halflife_pos: Dict[str, float],
                    ewm_min_periods: int, ewm_adjust: bool
                    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
    feats: List[str] = []

    # Venue / FDR
    df["venue"] = df["venue"].astype(str)
    df["venue_bin"] = (df["venue"].str.lower() == "home").astype(int); feats.append("venue_bin")
    df["fdr"] = np.where(df["venue_bin"].eq(1), df.get("fdr_home", np.nan), df.get("fdr_away", np.nan)).astype(float)
    feats.append("fdr")

    # Recency/availability
    if "days_since_last" in df.columns: feats.append("days_since_last")
    if "is_active" in df.columns: feats.append("is_active")
    if "minutes" in df.columns:
        df["prev_minutes"] = df.groupby(["player_id","season"], sort=False)["minutes"].shift(1)
        feats.append("prev_minutes")

    # Team context
    for c in ["team_att_z_venue","opp_def_z_venue"]:
        if c in df.columns: feats.append(c)

    # Upstream EWMA/ROLL form
    roll_like = _collect_roll_like(df, use_z=use_z)
    keep = [c for c in roll_like if df[c].notna().mean() >= na_thresh]
    feats.extend(keep)

    # Shots/SOT EWMA (per-pos halflife)
    df, sh_feats = _add_shot_ewm_by_pos(
        df, hl_map=ewm_halflife_pos, min_periods=ewm_min_periods, adjust=ewm_adjust
    )
    sh_keep = [c for c in sh_feats if df[c].notna().mean() >= na_thresh]
    feats.extend(sh_keep)

    ewm_used = [c for c in feats if ("_ewm" in c) or ("_roll" in c)]
    return df[feats].copy(), feats, ewm_used

# ------------------------------- Models ---------------------------------------

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

def _train_glm_poisson(Xtr: np.ndarray, ytr: np.ndarray) -> TweedieRegressor:
    m = TweedieRegressor(power=1.0, alpha=5e-4, link="log", max_iter=5000, tol=1e-6)
    m.fit(Xtr, ytr)
    return m

def _fit_per_pos_models(Xtr: pd.DataFrame, ytr: np.ndarray, pos_tr: pd.Series,
                        min_rows: int = 150) -> Tuple[Dict[str, lgb.LGBMRegressor], lgb.LGBMRegressor]:
    models: Dict[str, lgb.LGBMRegressor] = {}
    for pos in ["GK","DEF","MID","FWD"]:
        mask = pos_tr.eq(pos).to_numpy()
        if int(mask.sum()) >= min_rows:
            models[pos] = _lgbm_reg().fit(Xtr.iloc[mask], ytr[mask])
    global_model = _lgbm_reg().fit(Xtr, ytr)
    return models, global_model

def _fit_per_pos_poisson(Xtr: pd.DataFrame, ytr: np.ndarray, pos_tr: pd.Series,
                         med: pd.Series, min_rows: int = 150) -> Tuple[Dict[str, TweedieRegressor], TweedieRegressor]:
    models: Dict[str, TweedieRegressor] = {}
    Xtr_glm = Xtr.fillna(med).to_numpy()
    for pos in ["GK","DEF","MID","FWD"]:
        mask = pos_tr.eq(pos).to_numpy()
        if int(mask.sum()) >= min_rows:
            models[pos] = _train_glm_poisson(Xtr.iloc[mask].fillna(med).to_numpy(), ytr[mask])
    global_model = _train_glm_poisson(Xtr_glm, ytr)
    return models, global_model

def _predict_per_pos(models_by_pos: Dict[str, object], global_model: object,
                     X: pd.DataFrame, pos: pd.Series, is_glm: bool, med: Optional[pd.Series]) -> np.ndarray:
    out = np.empty(len(X), dtype=float)
    out[:] = np.nan
    for tag in ["GK","DEF","MID","FWD"]:
        mask = pos.eq(tag).to_numpy()
        if not mask.any(): continue
        Xp = X.iloc[mask]
        if tag in models_by_pos:
            out[mask] = (models_by_pos[tag].predict(Xp.fillna(med).to_numpy())
                         if is_glm else models_by_pos[tag].predict(Xp))
        else:
            out[mask] = (global_model.predict(Xp.fillna(med).to_numpy())
                         if is_glm else global_model.predict(Xp))
    return out

# ------------------------------- Calibration ----------------------------------

def _compute_reliability(p: np.ndarray, y: np.ndarray, pos: pd.Series, bins: int = 10) -> Tuple[Dict[str, float], pd.DataFrame]:
    def one_reliab(pv, yv, label: str) -> Tuple[List[Dict[str, object]], float, float]:
        if len(pv) == 0:
            return [], float("nan"), float("nan")
        bins_edges = np.linspace(0.0, 1.0, bins+1)
        idx = np.digitize(pv, bins_edges[1:-1], right=True)
        rows = []
        n_tot = len(pv)
        ece = 0.0
        for b in range(bins):
            m = (idx == b)
            nb = int(m.sum())
            if nb == 0:
                continue
            p_mean = float(pv[m].mean())
            y_rate = float(yv[m].mean())
            gap = abs(p_mean - y_rate)
            ece += (nb / n_tot) * gap
            rows.append({"kind": label, "bin": b, "bin_lo": float(bins_edges[b]),
                         "bin_hi": float(bins_edges[b+1]), "n": nb,
                         "p_mean": p_mean, "y_rate": y_rate, "gap": gap})
        brier = float(brier_score_loss(yv, pv)) if n_tot > 0 else np.nan
        return rows, brier, ece

    out_rows: List[Dict[str, object]] = []
    rows_all, brier_all, ece_all = one_reliab(p, y, "ALL")
    out_rows += rows_all
    stats = {"brier": brier_all, "ece": ece_all}

    for tag in ["GK","DEF","MID","FWD"]:
        m = pos.astype(str).str.upper().eq(tag).to_numpy()
        if m.any():
            rows_pos, brier_pos, ece_pos = one_reliab(p[m], y[m], tag)
            out_rows += rows_pos
            stats[f"brier_{tag}"] = brier_pos
            stats[f"ece_{tag}"] = ece_pos

    return stats, pd.DataFrame(out_rows)

def _fit_isotonic_per_pos(p_raw: np.ndarray, y: np.ndarray, pos: pd.Series) -> Dict[str, IsotonicRegression]:
    calibs: Dict[str, IsotonicRegression] = {}
    for tag in ["GK","DEF","MID","FWD"]:
        m = pos.astype(str).str.upper().eq(tag).to_numpy()
        if m.sum() < 200:  # need enough points
            continue
        pr = p_raw[m]; yr = y[m]
        if yr.min() == yr.max():
            continue
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(pr, yr)
        calibs[tag] = iso
    return calibs

def _apply_isotonic_per_pos(p_raw: np.ndarray, pos: pd.Series, calibs: Dict[str, IsotonicRegression]) -> np.ndarray:
    pout = p_raw.copy()
    for tag, iso in calibs.items():
        m = pos.astype(str).str.upper().eq(tag).to_numpy()
        if m.any():
            pout[m] = iso.transform(pout[m])
    return np.clip(pout, 0, 1)

# ------------------------------- Main -----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True,
                    help="Comma-separated seasons; last is TEST season (e.g. 2021-2022,2022-2023,2023-2024)")
    ap.add_argument("--first-test-gw", type=int, default=26)

    ap.add_argument("--features-root", type=Path, default=Path("data/processed/registry/features"))
    ap.add_argument("--form-version", required=True)

    ap.add_argument("--use-z", action="store_true")
    ap.add_argument("--na-thresh", type=float, default=0.70)
    ap.add_argument("--ewm-halflife", type=float, default=3.0)
    ap.add_argument("--ewm-halflife-pos", type=str, default="", help='e.g. "GK:4,DEF:4,MID:3,FWD:2"')
    ap.add_argument("--ewm-min-periods", type=int, default=1)
    ap.add_argument("--ewm-adjust", action="store_true")

    ap.add_argument("--minutes-preds", type=Path,
                    help="expected_minutes.csv from minutes model")
    ap.add_argument("--require-pred-minutes", action="store_true")

    ap.add_argument("--poisson-heads", action="store_true")

    # Diagnostics + calibration
    ap.add_argument("--dump-lambdas", action="store_true")
    ap.add_argument("--calibrate-poisson", action="store_true",
                    help="Per-position isotonic calibration for p_goal and p_assist (fit on TRAIN tail using actual minutes)")
    ap.add_argument("--reliability-bins", type=int, default=10)

    # GK handling for FPL decisions
    ap.add_argument("--skip-gk", action="store_true",
                    help="Exclude GK from training & metrics; set GK predictions/probabilities to 0 in output.")

    ap.add_argument("--model-out", type=Path, default=Path("data/models/goals_assists"))
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

    # ---- Load & prep ----
    df = _load_players(args.features_root, args.form_version, seasons)
    team_form = _load_team_form(args.features_root, args.form_version, seasons)
    df = _merge_team_z(df, team_form)

    df = df.sort_values(["season","date_played","gw_orig","player_id"]).reset_index(drop=True)

    # Features
    hl_map = _parse_halflife_pos(args.ewm_halflife_pos, default_hl=args.ewm_halflife)
    X, feat_cols, ewm_used = _build_features(
        df, use_z=args.use_z, na_thresh=args.na_thresh,
        ewm_halflife=args.ewm_halflife, ewm_halflife_pos=hl_map,
        ewm_min_periods=args.ewm_min_periods, ewm_adjust=args.ewm_adjust
    )

    # Targets per-90 (TRAIN uses actual minutes)
    m = df["minutes"].fillna(0).clip(lower=0)
    m90 = (m / 90.0).replace(0, np.nan)
    df["y_goals_p90"]   = (df["gls"] / m90).astype(float)
    df["y_assists_p90"] = (df["ast"] / m90).astype(float)

    # Split
    train_df, test_df = _chrono_split(df, seasons, args.first_test_gw)
    test_df = test_df.copy()
    test_df["rid"] = np.arange(len(test_df))  # immutable identity for this TEST slice

    # Train data
    Xtr = X.loc[train_df.index]
    ytr_g = train_df["y_goals_p90"].to_numpy()
    ytr_a = train_df["y_assists_p90"].to_numpy()
    pos_tr = train_df["pos"].astype(str).str.upper()
    mask = np.isfinite(ytr_g) & np.isfinite(ytr_a)
    if args.skip_gk:
        mask = mask & ~pos_tr.eq("GK").to_numpy()
    if mask.sum() == 0:
        raise ValueError("No valid labeled rows in TRAIN.")
    Xtr = Xtr.iloc[mask]; ytr_g = ytr_g[mask]; ytr_a = ytr_a[mask]; pos_tr = pos_tr.iloc[mask]

    # Persist training medians (global) and per-pos medians (optional)
    med_train = Xtr.median(numeric_only=True)
    for target in (latest_dir, version_dir):
        (target / "artifacts").mkdir(parents=True, exist_ok=True)
        # Global medians used for GLM imputations and can be reused by forecast
        (target / "artifacts" / "features_median.json").write_text(
            med_train.to_json(), encoding="utf-8"
        )
        # Per-position medians (optional; useful for experiments / audits)
        try:
            med_by_pos = Xtr.assign(pos=pos_tr.to_numpy()).groupby("pos").median(numeric_only=True)
            med_by_pos.to_csv(target / "artifacts" / "features_median_by_pos.csv")
        except Exception:
            pass

    # Models
    g_pos_models, g_global = _fit_per_pos_models(Xtr, ytr_g, pos_tr, min_rows=150)
    a_pos_models, a_global = _fit_per_pos_models(Xtr, ytr_a, pos_tr, min_rows=150)

    if args.poisson_heads:
        g_pos_pois, g_global_pois = _fit_per_pos_poisson(Xtr, ytr_g, pos_tr, med=med_train, min_rows=150)
        a_pos_pois, a_global_pois = _fit_per_pos_poisson(Xtr, ytr_a, pos_tr, med=med_train, min_rows=150)
    else:
        g_pos_pois = a_pos_pois = {}
        g_global_pois = a_global_pois = None

    # Predict per-90 on TEST
    Xte = X.loc[test_df.index]
    pos_te = test_df["pos"].astype(str).str.upper()

    # LGBM keeps NaNs (matches training missing-value routing); clip to non-negative
    g_p90_mean = np.clip(_predict_per_pos(g_pos_models, g_global, Xte, pos_te, is_glm=False, med=None), 0, None)
    a_p90_mean = np.clip(_predict_per_pos(a_pos_models, a_global, Xte, pos_te, is_glm=False, med=None), 0, None)

    # GLM imputes with TRAIN medians; clip to non-negative
    if args.poisson_heads:
        g_p90_pois = np.clip(_predict_per_pos(g_pos_pois, g_global_pois, Xte, pos_te, is_glm=True, med=med_train), 0, None)
        a_p90_pois = np.clip(_predict_per_pos(a_pos_pois, a_global_pois, Xte, pos_te, is_glm=True, med=med_train), 0, None)
    else:
        g_p90_pois = np.full(len(Xte), np.nan); a_p90_pois = np.full(len(Xte), np.nan)

    g_p90_mean_s = pd.Series(g_p90_mean, index=test_df["rid"])
    a_p90_mean_s = pd.Series(a_p90_mean, index=test_df["rid"])
    g_p90_pois_s = pd.Series(g_p90_pois, index=test_df["rid"])
    a_p90_pois_s = pd.Series(a_p90_pois, index=test_df["rid"])

    # ---- Build 'out' from TEST + minutes ----
    key = ["season","gw_orig","date_played","player_id"]
    out = test_df[["season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes","rid"]].copy()

    em = _load_expected_minutes(args.minutes_prefs if hasattr(args,'minutes_prefs') else args.minutes_preds)
    # ^ allow old typo defensive
    if args.require_pred_minutes and em is None:
        raise ValueError("--require-pred-minutes was set but --minutes-preds file is missing/invalid.")

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
        out["pred_minutes"] = out["minutes"].astype(float)

    # Merge team z (many_to_one)
    tz = (df[["season","gw_orig","team_id","team_att_z_venue","opp_def_z_venue"]]
          .sort_values(["season","gw_orig","team_id"])
          .drop_duplicates(subset=["season","gw_orig","team_id"], keep="last"))
    out = out.merge(tz, on=["season","gw_orig","team_id"], how="left", validate="many_to_one")

    # ---- Align predictions to FINAL out rows ----
    rid_keep = out["rid"].to_numpy()
    g_p90_mean_v = g_p90_mean_s.reindex(rid_keep).to_numpy()
    a_p90_mean_v = a_p90_mean_s.reindex(rid_keep).to_numpy()
    g_p90_pois_v = g_p90_pois_s.reindex(rid_keep).to_numpy()
    a_p90_pois_v = a_p90_pois_s.reindex(rid_keep).to_numpy()

    assert len(out) == len(g_p90_mean_v) == len(a_p90_mean_v) == len(g_p90_pois_v) == len(a_p90_pois_v), \
        "Internal alignment error: prediction arrays must match out rows."

    # Scale to per-match using pred_minutes
    scale = out["pred_minutes"].to_numpy() / 90.0
    pred_goals_mean   = g_p90_mean_v * scale
    pred_assists_mean = a_p90_mean_v * scale
    pred_goals_pois   = g_p90_pois_v * scale
    pred_assists_pois = a_p90_pois_v * scale

    # Choose per-90 rates for probability policy (prefer Poisson)
    rg90 = np.where(~np.isnan(g_p90_pois_v), g_p90_pois_v, g_p90_mean_v)
    ra90 = np.where(~np.isnan(a_p90_pois_v), a_p90_pois_v, a_p90_mean_v)

    # ---- Probabilities (Poisson); state mixture if available ----
    have_mix = all(c in out.columns for c in ["p_start","p_cameo","pred_start_head","pred_bench_cameo_head"])
    if have_mix:
        ps = out["p_start"].clip(0,1).to_numpy()
        pc = out["p_cameo"].clip(0,1).to_numpy()
        ms = np.clip(out["pred_start_head"].to_numpy(), 0, None)
        mb = np.clip(out["pred_bench_cameo_head"].to_numpy(), 0, None)

        lam_g_s = rg90 * (ms / 90.0); lam_g_b = rg90 * (mb / 90.0)
        lam_a_s = ra90 * (ms / 90.0); lam_a_b = ra90 * (mb / 90.0)

        p_goal_raw   = ps*(1.0 - np.exp(-lam_g_s)) + (1.0-ps)*pc*(1.0 - np.exp(-lam_g_b))
        p_assist_raw = ps*(1.0 - np.exp(-lam_a_s)) + (1.0-ps)*pc*(1.0 - np.exp(-lam_a_b))
        lam_g_eff = ps*lam_g_s + (1.0-ps)*pc*lam_g_b
        lam_a_eff = ps*lam_a_s + (1.0-ps)*pc*lam_a_b
    else:
        lam_g_eff = rg90 * (out["pred_minutes"].to_numpy() / 90.0)
        lam_a_eff = ra90 * (out["pred_minutes"].to_numpy() / 90.0)
        p_goal_raw   = 1.0 - np.exp(-lam_g_eff)
        p_assist_raw = 1.0 - np.exp(-lam_a_eff)
        lam_g_s = lam_g_b = lam_a_s = lam_a_b = np.full(len(out), np.nan)

    # ---- Optional calibration (TRAIN tail, actual minutes) ----
    if args.calibrate_poisson:
        _, calib_idx = _tail_index(train_df, frac=0.15)
        calib_df = train_df.loc[calib_idx].copy()
        Xc = X.loc[calib_idx]
        pos_c = calib_df["pos"].astype(str).str.upper()

        g_c_mean = np.clip(_predict_per_pos(g_pos_models, g_global, Xc, pos_c, is_glm=False, med=None), 0, None)
        a_c_mean = np.clip(_predict_per_pos(a_pos_models, a_global, Xc, pos_c, is_glm=False, med=None), 0, None)
        if args.poisson_heads:
            g_c = np.clip(_predict_per_pos(g_pos_pois, g_global_pois, Xc, pos_c, is_glm=True, med=med_train), 0, None)
            a_c = np.clip(_predict_per_pos(a_pos_pois, a_global_pois, Xc, pos_c, is_glm=True, med=med_train), 0, None)
        else:
            g_c = g_c_mean; a_c = a_c_mean

        lam_g_c = g_c * (np.clip(calib_df["minutes"].to_numpy(), 0, None) / 90.0)
        lam_a_c = a_c * (np.clip(calib_df["minutes"].to_numpy(), 0, None) / 90.0)
        p_goal_c   = 1.0 - np.exp(-lam_g_c)
        p_assist_c = 1.0 - np.exp(-lam_a_c)
        y_goal_c   = (calib_df["gls"].fillna(0) > 0).astype(int).to_numpy()
        y_ass_c    = (calib_df["ast"].fillna(0) > 0).astype(int).to_numpy()

        goal_calibs   = _fit_isotonic_per_pos(p_goal_c, y_goal_c, pos_c)
        assist_calibs = _fit_isotonic_per_pos(p_assist_c, y_ass_c, pos_c)

        p_goal = _apply_isotonic_per_pos(p_goal_raw, out["pos"], goal_calibs)
        p_assist = _apply_isotonic_per_pos(p_assist_raw, out["pos"], assist_calibs)

        for target in (latest_dir, version_dir):
            (target / "artifacts").mkdir(parents=True, exist_ok=True)
            joblib.dump(goal_calibs, target / "artifacts" / "p_goal_isotonic_per_pos.joblib")
            joblib.dump(assist_calibs, target / "artifacts" / "p_assist_isotonic_per_pos.joblib")
    else:
        p_goal = np.clip(p_goal_raw, 0, 1)
        p_assist = np.clip(p_assist_raw, 0, 1)

    p_return_any = 1.0 - (1.0 - p_goal) * (1.0 - p_assist)

    # ---- Metrics (eval-only) ----
    truth_idxed = test_df.set_index("rid")
    y_g_all = truth_idxed.loc[rid_keep, "gls"].to_numpy()
    y_a_all = truth_idxed.loc[rid_keep, "ast"].to_numpy()
    valid = np.isfinite(y_g_all) & np.isfinite(y_a_all)
    if args.skip_gk:
        valid = valid & ~out["pos"].eq("GK").to_numpy()
    n_drop = int((~valid).sum())
    if n_drop:
        logging.warning("Ignoring %d rows with NaN labels in TEST when computing metrics.", n_drop)

    y_g = y_g_all[valid]; y_a = y_a_all[valid]
    pred_goals_mean_eval   = pred_goals_mean[valid]
    pred_assists_mean_eval = pred_assists_mean[valid]
    pred_goals_pois_eval   = pred_goals_pois[valid]
    pred_assists_pois_eval = pred_assists_pois[valid]

    m90_eval = (truth_idxed.loc[rid_keep, "minutes"].to_numpy() / 90.0)[valid]
    with np.errstate(divide="ignore", invalid="ignore"):
        y_g_p90_eval = y_g / m90_eval
        y_a_p90_eval = y_a / m90_eval
    p90_mask = np.isfinite(y_g_p90_eval) & np.isfinite(y_a_p90_eval)

    pos_eval = out["pos"].iloc[valid.nonzero()[0]]
    p_goal_eval = p_goal[valid]; p_assist_eval = p_assist[valid]
    y_goal_eval = (y_g > 0).astype(int); y_ass_eval = (y_a > 0).astype(int)

    goal_stats, goal_rel = _compute_reliability(p_goal_eval, y_goal_eval, pos_eval, bins=args.reliability_bins)
    ass_stats,  ass_rel  = _compute_reliability(p_assist_eval, y_ass_eval, pos_eval, bins=args.reliability_bins)

    for target in (latest_dir, version_dir):
        (target / "artifacts").mkdir(parents=True, exist_ok=True)
        goal_rel.to_csv(target / "artifacts" / "reliability_goal.csv", index=False)
        ass_rel.to_csv(target / "artifacts" / "reliability_assist.csv", index=False)

    # Reflect actual per-pos model availability
    per_pos_models_lgbm = {tag: (tag in g_pos_models) and (tag in a_pos_models) for tag in ["GK","DEF","MID","FWD"]}
    if args.poisson_heads:
        per_pos_models_poisson = {tag: (tag in g_pos_pois) and (tag in a_pos_pois) for tag in ["GK","DEF","MID","FWD"]}
    else:
        per_pos_models_poisson = None

    metrics = {
        "code_version": CODE_VERSION,
        "n_train": int(len(Xtr)),
        "n_test_rows_written": int(len(out)),
        "require_pred_minutes": bool(args.require_pred_minutes),
        "poisson_heads": bool(args.poisson_heads),
        "calibrated": bool(args.calibrate_poisson),
        "skip_gk": bool(args.skip_gk),
        "goals_mae_mean":    (round(float(mean_absolute_error(y_g, pred_goals_mean_eval)), 5) if len(y_g) else None),
        "assists_mae_mean":  (round(float(mean_absolute_error(y_a, pred_assists_mean_eval)), 5) if len(y_a) else None),
        "goals_mae_poisson": (round(float(mean_absolute_error(y_g, pred_goals_pois_eval)), 5) if (len(y_g) and not np.isnan(pred_goals_pois_eval).all()) else None),
        "assists_mae_poisson": (round(float(mean_absolute_error(y_a, pred_assists_pois_eval)), 5) if (len(y_a) and not np.isnan(pred_assists_pois_eval).all()) else None),
        "goals_p90_mae_mean": (
            round(float(mean_absolute_error(y_g_p90_eval[p90_mask], g_p90_mean_v[valid][p90_mask])), 5) if p90_mask.any() else None
        ),
        "assists_p90_mae_mean": (
            round(float(mean_absolute_error(y_a_p90_eval[p90_mask], a_p90_mean_v[valid][p90_mask])), 5) if p90_mask.any() else None
        ),
        "goals_p90_mae_poisson": (
            round(float(mean_absolute_error(y_g_p90_eval[p90_mask], g_p90_pois_v[valid][p90_mask])), 5)
            if (p90_mask.any() and not np.isnan(g_p90_pois_v).all()) else None
        ),
        "assists_p90_mae_poisson": (
            round(float(mean_absolute_error(y_a_p90_eval[p90_mask], a_p90_pois_v[valid][p90_mask])), 5)
            if (p90_mask.any() and not np.isnan(a_p90_pois_v).all()) else None
        ),
        "brier_goal": round(goal_stats.get("brier", np.nan), 6),
        "ece_goal": round(goal_stats.get("ece", np.nan), 6),
        "brier_assist": round(ass_stats.get("brier", np.nan), 6),
        "ece_assist": round(ass_stats.get("ece", np.nan), 6),
        "test_season": test_season,
        "first_test_gw": int(args.first_test_gw),
        "per_pos_models": per_pos_models_lgbm,
        "per_pos_models_poisson": per_pos_models_poisson,
    }

    # ---- Assemble output (add true labels) ----
    out["goals_true"]   = truth_idxed.loc[rid_keep, "gls"].to_numpy()
    out["assists_true"] = truth_idxed.loc[rid_keep, "ast"].to_numpy()

    out["pred_goals_p90_mean"]      = g_p90_mean_v
    out["pred_assists_p90_mean"]    = a_p90_mean_v
    out["pred_goals_mean"]          = pred_goals_mean
    out["pred_assists_mean"]        = pred_assists_mean
    out["pred_goals_p90_poisson"]   = g_p90_pois_v
    out["pred_assists_p90_poisson"] = a_p90_pois_v
    out["pred_goals_poisson"]       = pred_goals_pois
    out["pred_assists_poisson"]     = pred_assists_pois
    out["p_goal"]                   = p_goal
    out["p_assist"]                 = p_assist
    out["p_return_any"]             = p_return_any

    # Zero GK outputs if skipping GK for decisions
    if args.skip_gk and out["pos"].eq("GK").any():
        m = out["pos"].eq("GK")
        zero_cols = [
            "pred_goals_p90_mean","pred_assists_p90_mean",
            "pred_goals_mean","pred_assists_mean",
            "pred_goals_p90_poisson","pred_assists_p90_poisson",
            "pred_goals_poisson","pred_assists_poisson",
            "p_goal","p_assist","p_return_any"
        ]
        for c in zero_cols:
            out.loc[m, c] = 0.0

    # Optional lambda dumps
    cols = [
        "season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes","pred_minutes",
        "goals_true","assists_true",
        "team_att_z_venue","opp_def_z_venue",
        "pred_goals_p90_mean","pred_assists_p90_mean",
        "pred_goals_mean","pred_assists_mean",
        "pred_goals_p90_poisson","pred_assists_p90_poisson",
        "pred_goals_poisson","pred_assists_poisson",
        "p_goal","p_assist","p_return_any",
    ]

    if args.dump_lambdas:
        out["lambda_goal"]  = lam_g_eff
        out["lambda_assist"]= lam_a_eff
        out["lambda_goal_start"]   = lam_g_s
        out["lambda_goal_bench"]   = lam_g_b
        out["lambda_assist_start"] = lam_a_s
        out["lambda_assist_bench"] = lam_a_b
        cols += ["lambda_goal","lambda_assist","lambda_goal_start","lambda_goal_bench","lambda_assist_start","lambda_assist_bench"]

    out = out[cols].copy()

    # ===================== AUDITS (row-level & feature-level) =====================
    # Row-level NaN audit for features used by the models (aligned to written rows)
    Xte_mask = Xte.isna().copy()
    Xte_mask.index = test_df["rid"].to_numpy()
    Xte_mask = Xte_mask.reindex(rid_keep)

    feat_non_na = (~Xte_mask).sum(axis=1).astype(int)
    feat_total  = int(Xte_mask.shape[1])
    feat_ratio  = (feat_non_na / float(feat_total))

    missing_features_str = Xte_mask.apply(
        lambda r: "|".join([col for col, is_na in zip(Xte_mask.columns, r.values) if is_na]),
        axis=1
    )

    na_rows = out[["season","gw_orig","date_played","player_id","team_id","player","pos"]].copy()
    na_rows["feat_non_na"]      = feat_non_na.values
    na_rows["feat_total"]       = feat_total
    na_rows["feat_ratio"]       = feat_ratio.values
    na_rows["missing_features"] = missing_features_str.values

    # Feature-level NaN summary: TRAIN vs final written TEST rows
    na_train = Xtr.isna().mean().rename("train_na_rate")
    Xte_final = Xte.copy()
    Xte_final.index = test_df["rid"].to_numpy()
    Xte_final = Xte_final.reindex(rid_keep)
    na_test  = Xte_final.isna().mean().rename("test_na_rate")
    na_sum = pd.concat([na_train, na_test], axis=1)
    na_sum["delta_pp"] = (na_sum["test_na_rate"] - na_sum["train_na_rate"]) * 100.0

    for target in (latest_dir, version_dir):
        (target / "artifacts").mkdir(parents=True, exist_ok=True)
        na_rows.to_csv(target / "artifacts" / "missing_features_by_row.csv", index=False)
        na_sum.sort_values("delta_pp", ascending=False).to_csv(target / "artifacts" / "feature_na_summary.csv")

    # ---- Persist ----
    for target in (latest_dir, version_dir):
        (target / "artifacts").mkdir(parents=True, exist_ok=True)
        out.to_csv(target / "goals_assists_predictions.csv", index=False)
        (target / "artifacts" / "features.json").write_text(json.dumps(list(X.columns), indent=2), encoding="utf-8")
        (target / "artifacts" / "ewm_features_used.txt").write_text("\n".join(ewm_used), encoding="utf-8")

    # Feature importances
    try:
        fi_g = pd.DataFrame({"feature": list(X.columns), "importance": g_global.feature_importances_() if callable(getattr(g_global, "feature_importances_", None)) else g_global.feature_importances_})
        fi_a = pd.DataFrame({"feature": list(X.columns), "importance": a_global.feature_importances_() if callable(getattr(a_global, "feature_importances_", None)) else a_global.feature_importances_})
        for target in (latest_dir, version_dir):
            fi_g.to_csv(target / "artifacts" / "goals_feature_importances.csv", index=False)
            fi_a.to_csv(target / "artifacts" / "assists_feature_importances.csv", index=False)
    except Exception:
        pass

    # Save models
    def _save_lgb(name: str, model: lgb.LGBMRegressor):
        try:
            model.booster_.save_model(str(latest_dir / f"{name}.txt"))
            model.booster_.save_model(str(version_dir / f"{name}.txt"))
        except Exception:
            pass

    _save_lgb("goals_global_lgbm", g_global)
    _save_lgb("assists_global_lgbm", a_global)
    for tag, mdl in g_pos_models.items(): _save_lgb(f"goals_{tag}_lgbm", mdl)
    for tag, mdl in a_pos_models.items(): _save_lgb(f"assists_{tag}_lgbm", mdl)

    if args.poisson_heads:
        if g_global_pois is not None:
            joblib.dump(g_global_pois, latest_dir / "goals_global_poisson.joblib")
            joblib.dump(g_global_pois, version_dir / "goals_global_poisson.joblib")
        if a_global_pois is not None:
            joblib.dump(a_global_pois, latest_dir / "assists_global_poisson.joblib")
            joblib.dump(a_global_pois, version_dir / "assists_global_poisson.joblib")
        for tag, mdl in g_pos_pois.items():
            joblib.dump(mdl, latest_dir / f"goals_{tag}_poisson.joblib")
            joblib.dump(mdl, version_dir / f"goals_{tag}_poisson.joblib")
        for tag, mdl in a_pos_pois.items():
            joblib.dump(mdl, latest_dir / f"assists_{tag}_poisson.joblib")
            joblib.dump(mdl, version_dir / f"assists_{tag}_poisson.joblib")

    (latest_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_meta(latest_dir, args)
    _write_meta(version_dir, args)

    logging.info(json.dumps(metrics, indent=2))
    logging.info("Artifacts written to %s (latest) and %s (versioned: %s).", latest_dir, version_dir, version_name)

if __name__ == "__main__":
    main()
