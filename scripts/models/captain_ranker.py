#!/usr/bin/env python3
"""
captain_ranker.py — v2.2
Rank the best captain picks per gameweek (top-5) with Learning-to-Rank; optional binary baseline.

Key features
------------
• Auto-versioning under --models-out (e.g., v1, v2, …).
• Multi-season training via --train-seasons (comma list), plus strict anti-leak split:
    TRAIN  = all seasons != test_season  OR  (test_season & gw < test_from_gw)
    TEST   = (test_season & test_from_gw .. test_from_gw+test_horizon-1)
• Works with single-season or multi-season actuals:
    - if the CSV lacks a 'season' column, it's treated as the test season only.
• Labels are clipped to be non-negative for LightGBM ranker (required).
• Outputs:
    <models-out>/<version>/captain_preds_rank.csv
    <models-out>/<version>/captain_top5_by_gw.csv
    <models-out>/<version>/metrics.json
    (and *_clf.* if --mode=clf)
• Saves model:
    <models-out>/<version>/ranker_lgbm.txt (or classifier_lgbm.txt)

CLI (example)
-------------
py -m scripts.models.captain_ranker \
  --xp-csv data/models/expected_points/v1/expected_points.csv \
  --actual-csv data/processed/fpl/2024-2025/gws/merged_gws.csv \
  --season 2024-2025 \
  --train-seasons 2022-2023,2023-2024,2024-2025 \
  --test-from-gw 30 --test-horizon 6 \
  --mode rank \
  --models-out data/models/captain_ranker \
  --auto-version
"""

from __future__ import annotations
import argparse, json, re, sys, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMClassifier
from sklearn.metrics import ndcg_score

# ───────────────────── helpers & constants ─────────────────────

KEY = ["season","gw_orig","player_id"]

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "season" in df.columns:
        df["season"] = df["season"].astype(str)
    if "gw_orig" in df.columns:
        df["gw_orig"] = pd.to_numeric(df["gw_orig"], errors="coerce").astype("Int64")
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(str)
    if "team_id" in df.columns:
        df["team_id"] = df["team_id"].astype(str)
    return df

def _resolve_version(base_dir: Path, requested: Optional[str], auto: bool) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    if auto or (not requested) or (requested.lower() == "auto"):
        existing = [p.name for p in base_dir.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
        nxt = (max(int(s[1:]) for s in existing) + 1) if existing else 1
        ver = f"v{nxt}"
        print(f"[info] auto-version -> {ver}")
        return ver
    if not re.fullmatch(r"v\d+", requested):
        if requested.isdigit():
            return f"v{requested}"
        raise ValueError(f"--model-version must be like v3 or a number; got {requested}")
    return requested

def _coalesce_col(df: pd.DataFrame, candidates: List[str], target: str) -> pd.DataFrame:
    for c in candidates:
        if c in df.columns:
            if c != target:
                df = df.rename(columns={c: target})
            return df
    return df

def _parse_train_seasons(s: Optional[str], test_season: str) -> List[str]:
    if not s:
        return [test_season]
    items = [x.strip() for x in s.split(",") if x.strip()]
    if test_season not in items:
        items.append(test_season)
    # ensure sortable 'YYYY-YYYY' order
    return sorted(items)

# ───────────────────── data loaders ─────────────────────

def load_xp(fp: Path, seasons_needed: List[str]) -> pd.DataFrame:
    df = pd.read_csv(fp, parse_dates=["date_played"], low_memory=False)
    df = _norm_cols(df)

    need = {"season","gw_orig","player_id","team_id","pos","player","exp_points_total"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"[XP] missing columns: {miss}")

    df = df[df["season"].isin([str(s) for s in seasons_needed])].copy()

    # Collapse duplicates (DGWs) to 1 row per player-GW at feature level
    agg_cols = [c for c in [
        "exp_points_total","p60","p1",
        "xp_goals","xp_assists","xp_clean_sheets","xp_saves_points","xp_concede_penalty"
    ] if c in df.columns]
    grp_cols = ["season","gw_orig","player_id","team_id","pos","player"]
    xp = df.groupby(grp_cols, as_index=False)[agg_cols].sum()
    return xp

def load_actual_any(fp: Path, seasons_needed: List[str], test_season: str) -> pd.DataFrame:
    """
    Accepts:
      • multi-season CSV with 'season' column
      • single-season CSV *without* 'season' → assumed to be 'test_season' only
    Returns rows aggregated to (season, gw_orig, player_id) with column 'actual_points'.
    """
    df = pd.read_csv(fp, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # GW column -> gw_orig
    df = _coalesce_col(df, ["gw_orig","gw","event","round","deadline_id"], "gw_orig")
    if "gw_orig" not in df.columns:
        raise ValueError("[ACTUAL] needs a gameweek column (gw_orig/gw/event/round/deadline_id)")
    df["gw_orig"] = pd.to_numeric(df["gw_orig"], errors="coerce").astype("Int64")

    # Player id column
    if "player_id" not in df.columns:
        raise ValueError("[ACTUAL] must contain 'player_id' that matches XP's player_id")

    # Points column
    if "total_points" not in df.columns:
        # Some exports use 'points' — map if present
        if "points" in df.columns:
            df = df.rename(columns={"points":"total_points"})
        else:
            raise ValueError("[ACTUAL] must contain 'total_points'")

    # Season handling
    if "season" not in df.columns:
        # Treat as single-season file → test season only
        df["season"] = str(test_season)
    else:
        df["season"] = df["season"].astype(str)

    df = df[df["season"].isin([str(s) for s in seasons_needed])].copy()
    if df.empty:
        # let caller handle; training may proceed with available seasons only
        return pd.DataFrame(columns=["season","gw_orig","player_id","actual_points"])

    act = (df.groupby(["season","gw_orig","player_id"], as_index=False)["total_points"]
             .sum()
             .rename(columns={"total_points":"actual_points"}))
    act = _norm_cols(act)
    return act

def load_prices_json(fp: Path, season_gws: List[int]) -> pd.DataFrame:
    """Return per-GW prices: player_id, gw_orig, now_cost (tenths)."""
    reg = json.loads(Path(fp).read_text("utf-8"))
    players = reg.get("players", {})
    rows = []
    def _pick(gw_dict: dict, gw: int):
        sgw = str(gw)
        if sgw in gw_dict: return gw_dict[sgw]
        keys = sorted(int(k) for k in gw_dict.keys() if k.isdigit() and int(k) <= gw)
        return gw_dict[str(keys[-1])] if keys else None

    for pid, pdata in players.items():
        gwd = pdata.get("gw", {})
        for gw in season_gws:
            ent = _pick(gwd, int(gw))
            if not ent: 
                continue
            price = ent.get("price", None)
            if price is None: 
                continue
            rows.append({
                "player_id": str(pid),
                "gw_orig": int(gw),
                "now_cost": int(round(float(price)*10)),
            })
    return pd.DataFrame(rows)

def add_prices(df: pd.DataFrame, prices: Optional[pd.DataFrame]) -> pd.DataFrame:
    if prices is None or prices.empty:
        df = df.copy()
        df["now_cost"] = np.nan
        return df
    out = df.merge(prices[["player_id","gw_orig","now_cost"]], on=["player_id","gw_orig"], how="left")
    out = out.sort_values(["player_id","gw_orig"])
    out["now_cost"] = (out.groupby("player_id")["now_cost"].ffill()
                         .groupby(out["player_id"]).bfill())
    return out

# ───────────────────── dataset & metrics ─────────────────────

def ensure_nonnegative_labels(df: pd.DataFrame, mode: str = "clip0") -> pd.DataFrame:
    df = df.copy()
    if "actual_points" not in df.columns:
        df["actual_points"] = 0.0
        return df
    if mode == "clip0":
        df["actual_points"] = pd.to_numeric(df["actual_points"], errors="coerce").fillna(0.0).clip(lower=0.0)
    elif mode == "shift_gw":
        mins = df.groupby(["season","gw_orig"])["actual_points"].transform("min")
        df["actual_points"] = pd.to_numeric(df["actual_points"], errors="coerce").fillna(0.0) - np.minimum(mins, 0.0)
    elif mode == "shift_all":
        gmin = float(pd.to_numeric(df["actual_points"], errors="coerce").fillna(0.0).min())
        df["actual_points"] = pd.to_numeric(df["actual_points"], errors="coerce").fillna(0.0) - min(gmin, 0.0)
    else:
        raise ValueError(f"Unknown label transform: {mode}")
    return df

def make_dataset(
    xp: pd.DataFrame,
    act: pd.DataFrame,
    prices: Optional[pd.DataFrame],
    test_season: str,
    test_from_gw: int,
    test_horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    # merge labels
    df = xp.merge(act, on=["season","gw_orig","player_id"], how="left")
    df["actual_points"] = df["actual_points"].fillna(0.0)

    # non-negative labels (LightGBM ranker requirement)
    df = ensure_nonnegative_labels(df, mode="clip0")

    # price feature
    all_gws = sorted(df.loc[df["season"] == str(test_season), "gw_orig"].dropna().unique().astype(int).tolist())
    df = add_prices(df, prices)

    # base features
    feat_cols = [c for c in [
        "exp_points_total","p60","p1",
        "xp_goals","xp_assists","xp_clean_sheets","xp_saves_points","xp_concede_penalty",
        "now_cost"
    ] if c in df.columns]

    # position one-hot (from XP)
    df["pos"] = df["pos"].astype(str).str.upper()
    # normalize common variants
    df["pos"] = df["pos"].replace({"GKP":"GK","G":"GK","D":"DEF","M":"MID","F":"FWD"})
    for pos in ["GK","DEF","MID","FWD"]:
        df[f"pos_{pos}"] = (df["pos"] == pos).astype(int)
    feat_cols += [f"pos_{p}" for p in ["GK","DEF","MID","FWD"]]

    # strict chrono split (anti-leak)
    te_start = int(test_from_gw)
    te_end   = te_start + int(test_horizon) - 1
    is_test_season = df["season"].astype(str).eq(str(test_season))

    test  = df[ is_test_season & df["gw_orig"].between(te_start, te_end) ].copy()
    train = df[ (~is_test_season) | (df["gw_orig"] < te_start)            ].copy()

    # columns to return
    cols = feat_cols + KEY + ["team_id","player","actual_points","pos"]
    # Keep available subset
    cols = [c for c in cols if c in df.columns]

    return train[cols].copy(), test[cols].copy(), \
           sorted(train["gw_orig"].dropna().unique().astype(int).tolist()), \
           list(range(te_start, te_end+1))

def make_groups(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """Return X, y, group_sizes, and feature_names for LightGBM ranker/cls."""
    df = df.sort_values(["season","gw_orig","player_id"])
    feat_cols = [c for c in df.columns if c not in (KEY + ["team_id","player","actual_points","pos"])]
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["actual_points"].to_numpy(dtype=float)
    groups = df.groupby(["season","gw_orig"]).size().tolist()
    return X, y, groups, feat_cols

def ndcg_at_k_macro(test_df: pd.DataFrame, preds: pd.Series, k: int = 5) -> float:
    scores = []
    for (_, _), idx in test_df.groupby(["season","gw_orig"]).groups.items():
        y_true = test_df.loc[idx, "actual_points"].to_numpy().reshape(1, -1)
        y_pred = preds.loc[idx].to_numpy().reshape(1, -1)
        # Handle degenerate all-zeros truth gracefully
        if np.all(y_true == 0.0):
            scores.append(0.0)
        else:
            scores.append(ndcg_score(y_true, y_pred, k=k))
    return float(np.mean(scores)) if scores else 0.0

def precision_hit_at5(test_df: pd.DataFrame, pred: pd.Series) -> Tuple[float,float]:
    P5, H5, n = 0.0, 0.0, 0
    for (_, _), g in test_df.groupby(["season","gw_orig"]):
        g = g.copy()
        g["pred"] = pred.loc[g.index]
        pred_top5   = set(g.sort_values("pred", ascending=False).head(5)["player_id"])
        actual_top5 = set(g.sort_values("actual_points", ascending=False).head(5)["player_id"])
        actual_top1 = g.sort_values("actual_points", ascending=False).iloc[0]["player_id"] if len(g) else None
        inter = len(pred_top5 & actual_top5)
        P5 += inter / 5.0
        H5 += 1.0 if (actual_top1 in pred_top5) else 0.0
        n  += 1
    return (P5 / n if n else 0.0, H5 / n if n else 0.0)

# ───────────────────── training / inference ─────────────────────

def run_ranker(train: pd.DataFrame, test: pd.DataFrame, out_dir: Path) -> None:
    Xtr, ytr, gtr, feat_names = make_groups(train)
    Xte, yte, gte, _         = make_groups(test)

    # safety: LightGBM requires non-negative labels
    if np.any(ytr < 0) or np.any(yte < 0):
        raise ValueError("Negative labels present after clipping — check input preprocessing.")

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42
    )
    # Note: LightGBM sklearn ranker wants groups as list of group sizes
    model.fit(Xtr, ytr, group=gtr, eval_set=[(Xte, yte)], eval_group=[gte], eval_at=[5])

    # Predict on test
    test_feat = test[feat_names]
    test_pred = pd.Series(model.predict(test_feat), index=test.index, name="pred_rank")

    # Metrics
    ndcg5 = ndcg_at_k_macro(test, test_pred, k=5)
    p5, h5 = precision_hit_at5(test, test_pred)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pred_df = test.copy()
    pred_df["pred_rank"] = test_pred
    pred_df.to_csv(out_dir / "captain_preds_rank.csv", index=False)

    # Top-5 per GW
    top5_rows = []
    for (s,gw), g in pred_df.groupby(["season","gw_orig"]):
        top = g.sort_values("pred_rank", ascending=False).head(5)
        for r in top.itertuples():
            top5_rows.append({
                "season": s, "gw_orig": int(gw),
                "player_id": r.player_id, "player": r.player, "team_id": r.team_id, "pos": r.pos,
                "pred_rank": float(r.pred_rank),
                "xp": float(getattr(r, "exp_points_total", np.nan)),
                "actual_points": float(r.actual_points),
            })
    pd.DataFrame(top5_rows).to_csv(out_dir / "captain_top5_by_gw.csv", index=False)

    # Save metrics + model
    (out_dir / "metrics.json").write_text(json.dumps({
        "mode": "rank",
        "ndcg_at_5": ndcg5,
        "precision_at_5": p5,
        "hit_at_5": h5
    }, indent=2))
    # booster text model
    model.booster_.save_model(out_dir / "ranker_lgbm.txt")

    print(f"[RANK] NDCG@5={ndcg5:.3f}  P@5={p5:.3f}  Hit@5={h5:.3f}")

def run_classifier(train: pd.DataFrame, test: pd.DataFrame, out_dir: Path) -> None:
    # binary label = is_top5_actual per GW
    def label_top5(df: pd.DataFrame) -> pd.Series:
        lab = pd.Series(0, index=df.index, dtype=int)
        for (_, _), g in df.groupby(["season","gw_orig"]):
            top5_idx = g.sort_values("actual_points", ascending=False).head(5).index
            lab.loc[top5_idx] = 1
        return lab

    ytr = label_top5(train)
    yte = label_top5(test)

    feat_cols = [c for c in train.columns if c not in (KEY + ["team_id","player","actual_points","pos"])]
    Xtr = train[feat_cols].to_numpy(dtype=float)
    Xte = test[feat_cols].to_numpy(dtype=float)

    pos_rate = max(1e-6, float(ytr.mean()))
    spw = (1.0 - pos_rate) / pos_rate

    clf = LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=25,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        random_state=42
    )
    clf.fit(Xtr, ytr)

    proba = pd.Series(clf.predict_proba(Xte)[:,1], index=test.index, name="pred_clf")

    ndcg5 = ndcg_at_k_macro(test, proba, k=5)
    p5, h5 = precision_hit_at5(test, proba)

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = test.copy()
    pred_df["pred_clf"] = proba
    pred_df.to_csv(out_dir / "captain_preds_clf.csv", index=False)

    top5_rows = []
    for (s,gw), g in pred_df.groupby(["season","gw_orig"]):
        top = g.sort_values("pred_clf", ascending=False).head(5)
        for r in top.itertuples():
            top5_rows.append({
                "season": s, "gw_orig": int(gw),
                "player_id": r.player_id, "player": r.player, "team_id": r.team_id, "pos": r.pos,
                "pred_clf": float(r.pred_clf),
                "xp": float(getattr(r, "exp_points_total", np.nan)),
                "actual_points": float(r.actual_points),
            })
    pd.DataFrame(top5_rows).to_csv(out_dir / "captain_top5_by_gw_clf.csv", index=False)

    (out_dir / "metrics_clf.json").write_text(json.dumps({
        "mode": "clf",
        "ndcg_at_5": ndcg5,
        "precision_at_5": p5,
        "hit_at_5": h5
    }, indent=2))

    # Save model
    clf.booster_.save_model(out_dir / "classifier_lgbm.txt")

    print(f"[CLF ] NDCG@5={ndcg5:.3f}  P@5={p5:.3f}  Hit@5={h5:.3f}")

# ───────────────────── CLI ─────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xp-csv", type=Path, required=True)
    ap.add_argument("--actual-csv", type=Path, required=True,
                    help="merged_gws.csv (single-season without 'season' column is OK) or a multi-season CSV with 'season'")
    ap.add_argument("--prices-json", type=Path, default=None)
    ap.add_argument("--season", type=str, required=True, help="test season, e.g. 2024-2025")
    ap.add_argument("--train-seasons", type=str, default=None,
                    help="comma list, e.g. 2022-2023,2023-2024,2024-2025. If omitted, uses only --season")
    ap.add_argument("--test-from-gw", type=int, required=True)
    ap.add_argument("--test-horizon", type=int, default=8)
    ap.add_argument("--mode", type=str, choices=["rank","clf"], default="rank")
    ap.add_argument("--models-out", type=Path, required=True, help="models root where versioned folder will be created")
    ap.add_argument("--model-version", type=str, default=None, help="version folder name (e.g. v3). Overridden by --auto-version.")
    ap.add_argument("--auto-version", action="store_true", help="pick next vN under --models-out automatically")
    args = ap.parse_args()

    test_season = str(args.season)
    seasons_needed = _parse_train_seasons(args.train_seasons, test_season)

    # Resolve versioned output directory
    version = _resolve_version(args.models_out, args.model_version, args.auto_version)
    out_dir = args.models_out / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load XP for all needed seasons
    xp = load_xp(args.xp_csv, seasons_needed)

    # Load ACTUALS (may be single-season file)
    act_all = load_actual_any(args.actual_csv, seasons_needed, test_season)
    # Warn if some requested seasons have no labels
    missing = [s for s in seasons_needed if s not in set(act_all["season"].unique())]
    if missing and len(set(seasons_needed)) > 1:
        print(f"WARNING: No labels found for seasons: {', '.join(missing)} (training will use available seasons only)")

    # Optional prices
    prices = None
    if args.prices_json:
        season_gws = sorted(xp.loc[xp["season"] == test_season, "gw_orig"].dropna().unique().astype(int).tolist())
        prices = load_prices_json(args.prices_json, season_gws)

    # Build datasets (strict anti-leak)
    train, test, tr_gws, te_gws = make_dataset(
        xp=xp, act=act_all, prices=prices,
        test_season=test_season, test_from_gw=args.test_from_gw, test_horizon=args.test_horizon
    )

    print(f"Seasons (train): {sorted(train['season'].unique().tolist())} | Test: {test_season}")
    print(f"Train rows: {len(train)} | Test rows: {len(test)} | Features: {len(train.columns) - 7}")
    print(f"Test GWs: {te_gws}")

    # Save fold info for reproducibility
    (out_dir / "fold_info.json").write_text(json.dumps({
        "season": test_season,
        "train_gws_unique": tr_gws,
        "test_gws": te_gws,
        "mode": args.mode,
        "train_seasons": seasons_needed,
        "version": version,
    }, indent=2))

    # Train & predict
    if args.mode == "rank":
        run_ranker(train, test, out_dir)
    else:
        run_classifier(train, test, out_dir)

if __name__ == "__main__":
    main()
