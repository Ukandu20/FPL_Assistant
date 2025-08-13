#!/usr/bin/env python3
"""
expected_points_aggregator.py – v2 (fixed/complete)

Combine model outputs into expected FPL points per player-fixture.

Inputs (CSV paths):
  --minutes        minutes_predictions.csv
                   (needs: pred_exp_minutes; optional: prob_played1_cal/_raw, prob_played60_cal/_raw, pos)
  --goals-assists  goals_assists_predictions.csv
                   (needs: pred_goals_mean, pred_assists_mean; provides pos)
  --saves          saves_predictions.csv
                   (GK only; optional: exp_save_points_mean or pred_saves_mean)
  --defense        defense_predictions.csv
                   (optional: cs probability under cs_prob_* or prob_cs_* or p_cs;
                              optional: exp_dcp_points or pred_ga_mean for concede penalty)
  --out-dir        directory to write results (e.g., data/models/expected_points)
  --version        subfolder under out-dir (e.g., v1)
  --log-level      INFO/DEBUG/WARN

Key used to align rows:
  (season, gw_orig, date_played, player_id, team_id)

Output:
  <out-dir>/<version>/expected_points.csv with per-component breakdown and total.

Notes:
- Appearance points are E = p1 + p60 (1 for appearance + 1 for 60+ minutes).
- Clean sheet points include the 60+ requirement: GK/DEF 4 pts, MID 1 pt, FWD 0.
- Goals: GK/DEF 6, MID 5, FWD 4. Assists: all 3.
- If defense file provides exp_dcp_points, we use it; else, if pred_ga_mean is available,
  we approximate concede penalty as -0.5 * pred_ga_mean for GK/DEF (≈ -1 per 2 conceded).
- Saves: if exp_save_points_mean exists, use it; else if pred_saves_mean exists, use pred_saves_mean/3.
- All merges are left-joins onto the GA base after dropping duplicate keys.
"""

from __future__ import annotations
import argparse, logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

KEY: List[str] = ["season", "gw_orig", "date_played", "player_id", "team_id"]

# ───────────────────────── helpers ─────────────────────────

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names / dtypes and keep KEY types consistent."""
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    # Ensure datetime for date_played
    if "date_played" in df.columns and not np.issubdtype(df["date_played"].dtype, np.datetime64):
        try:
            df["date_played"] = pd.to_datetime(df["date_played"])
        except Exception:
            pass
    # Coerce season/gw to str/int sensible types
    if "season" in df.columns:
        df["season"] = df["season"].astype(str)
    if "gw_orig" in df.columns:
        try:
            df["gw_orig"] = pd.to_numeric(df["gw_orig"], errors="coerce").astype("Int64")
        except Exception:
            pass
    return df

def _read_csv(path: Path, need: List[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date_played"], low_memory=False)
    df = _norm(df)
    if need:
        miss = [c for c in need if c not in df.columns]
        if miss:
            logging.warning("File %s missing expected columns: %s", path, miss)
    return df

def _drop_dupes(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    if not set(KEY).issubset(df.columns):
        return df
    before = len(df)
    df = df.drop_duplicates(subset=KEY, keep="last").copy()
    d = before - len(df)
    if d:
        logging.warning("Dropped %d duplicate rows on %s for %s", d, KEY, tag)
    return df

def _select_prob(df: pd.DataFrame, cal: str, raw: str, fallback: np.ndarray) -> np.ndarray:
    if cal in df.columns:
        return df[cal].astype(float).to_numpy()
    if raw in df.columns:
        return df[raw].astype(float).to_numpy()
    return fallback

def _choose_first(df: pd.DataFrame, candidates: List[str], default: float | None = None) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(default, index=df.index, dtype="float64")

def _pos_points_vector(pos: pd.Series, kind: str) -> np.ndarray:
    """Return per-row points for goals (kind='goal') or CS (kind='cs') based on position."""
    # Normalize position codes
    p = pos.fillna("").str.upper().str[:3]
    if kind == "goal":
        # GK/DEF=6, MID=5, FWD=4
        return np.where(p.isin(["GKP","GK"]), 6,
               np.where(p.isin(["DEF"]), 6,
               np.where(p.isin(["MID"]), 5,
               np.where(p.isin(["FWD"]), 4, 0)))).astype(float)
    elif kind == "cs":
        # GK/DEF=4, MID=1, FWD=0
        return np.where(p.isin(["GKP","GK","DEF"]), 4,
               np.where(p.isin(["MID"]), 1, 0)).astype(float)
    else:
        raise ValueError("kind must be 'goal' or 'cs'")

def _is_def_or_gk(pos: pd.Series) -> np.ndarray:
    p = pos.fillna("").str.upper().str[:3]
    return p.isin(["GKP","GK","DEF"]).to_numpy()

# ───────────────────────── core ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Aggregate expected FPL points from component models.")
    ap.add_argument("--minutes", required=True, type=Path, help="CSV from minutes model")
    ap.add_argument("--goals-assists", required=True, type=Path, help="CSV from GA model")
    ap.add_argument("--saves", type=Path, default=None, help="CSV from saves model (GK)")
    ap.add_argument("--defense", type=Path, default=None, help="CSV from defense model")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory (base)")
    ap.add_argument("--version", required=True, type=str, help="Subfolder under out-dir")
    ap.add_argument("--log-level", default="INFO", type=str)
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    # Read inputs
    ga_need = ["pred_goals_mean", "pred_assists_mean"]
    ga_df = _drop_dupes(_read_csv(args.goals_assists, need=ga_need), "GA")
    if not set(KEY).issubset(ga_df.columns):
        raise ValueError("GA file missing required key columns")

    min_need = ["pred_exp_minutes"]
    min_df = _drop_dupes(_read_csv(args.minutes, need=min_need), "MIN")

    # Attach appearance probs on minutes df (aligned to its index)
    m_exp = min_df["pred_exp_minutes"].fillna(0.0).astype(float).to_numpy()
    fallback_p1  = (m_exp > 0).astype(float)
    fallback_p60 = np.clip(m_exp / 90.0, 0.0, 1.0)

    min_df["__p1__"]  = _select_prob(min_df, "prob_played1_cal",  "prob_played1_raw",  fallback_p1)
    min_df["__p60__"] = _select_prob(min_df, "prob_played60_cal", "prob_played60_raw", fallback_p60)

    # Base frame: GA predictions (has pos, venue, etc.)
    base = ga_df.copy()

    # Merge minutes: bring expected minutes and p1/p60
    base = base.merge(min_df[KEY + ["pred_exp_minutes","__p1__","__p60__"]], on=KEY, how="left")

    # Choose pos column (prefer GA, else minutes if present)
    if "pos" not in base.columns:
        sources = []
        for src in [ga_df, min_df]:
            if "pos" in src.columns:
                sources.append(src[KEY + ["pos"]])
        if sources:
            pos_df = pd.concat(sources).drop_duplicates(subset=KEY, keep="last")
            base = base.merge(pos_df, on=KEY, how="left")
        else:
            base["pos"] = np.nan

    # Defense merges
    if args.defense:
        def_df = _drop_dupes(_read_csv(args.defense), "DEF")
        # Select CS probability (prefer cs_prob_* names first)
        cs_candidates = [
            "cs_prob_cal","cs_prob_raw",
            "prob_cs_cal","prob_cs_raw",
            "p_cs","prob_cs"
        ]
        def_df["__p_cs__"] = _choose_first(def_df, cs_candidates, default=np.nan)
        keep_cols = [c for c in ["__p_cs__", "exp_dcp_points", "pred_ga_mean"] if (c in def_df.columns) or c.startswith("__")]
        base = base.merge(def_df[KEY + keep_cols], on=KEY, how="left")
    else:
        base["__p_cs__"] = np.nan

    # Saves merges
    if args.saves:
        sv_df = _drop_dupes(_read_csv(args.saves), "SAV")
        if "exp_save_points_mean" in sv_df.columns:
            sv_df["__xp_saves__"] = sv_df["exp_save_points_mean"].astype(float)
        elif "pred_saves_mean" in sv_df.columns:
            sv_df["__xp_saves__"] = sv_df["pred_saves_mean"].astype(float) / 3.0
        else:
            sv_df["__xp_saves__"] = np.nan
        base = base.merge(sv_df[KEY + ["__xp_saves__"]], on=KEY, how="left")
    else:
        base["__xp_saves__"] = np.nan

    # Fill NaNs for appearance probs
    base["p1"]  = pd.Series(base["__p1__"]).astype(float).fillna(0.0)
    base["p60"] = pd.Series(base["__p60__"]).astype(float).fillna(0.0)

    # Appearance points
    base["xp_appearance"] = base["p1"] + base["p60"]

    # Goals & assists points
    goal_mean_candidates   = ["pred_goals_mean", "pred_goals_poisson"]
    assist_mean_candidates = ["pred_assists_mean", "pred_assists_poisson"]

    base["exp_goals"]   = _choose_first(base, goal_mean_candidates, default=0.0).astype(float).clip(lower=0.0)
    base["exp_assists"] = _choose_first(base, assist_mean_candidates, default=0.0).astype(float).clip(lower=0.0)

    goal_pts_vec = _pos_points_vector(base.get("pos", pd.Series(index=base.index)), kind="goal")
    base["xp_goals"]   = base["exp_goals"]   * goal_pts_vec
    base["xp_assists"] = base["exp_assists"] * 3.0

    # Clean sheet points
    cs_pts_vec = _pos_points_vector(base.get("pos", pd.Series(index=base.index)), kind="cs")
    p_cs = pd.to_numeric(base.get("__p_cs__"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    base["xp_clean_sheets"] = base["p60"] * p_cs * cs_pts_vec

    # Goals conceded penalty (GK/DEF only)
    is_def_gk = _is_def_or_gk(base.get("pos", pd.Series(index=base.index)))
    if "exp_dcp_points" in base.columns:
        base["xp_concede_penalty"] = np.where(is_def_gk, base["exp_dcp_points"].astype(float), 0.0)
    elif "pred_ga_mean" in base.columns:
        base["xp_concede_penalty"] = np.where(is_def_gk, -0.5 * pd.to_numeric(base["pred_ga_mean"], errors="coerce").fillna(0.0), 0.0)
    else:
        base["xp_concede_penalty"] = 0.0

    # Saves points (GK only)
    base["xp_saves_points"] = np.where(
        is_def_gk & base.get("pos", pd.Series(index=base.index)).fillna("").str.upper().str.startswith("GK"),
        pd.to_numeric(base["__xp_saves__"], errors="coerce").fillna(0.0),
        0.0
    )

    # Optional extras if present (bonus, cards, pens etc.)
    extras: Dict[str, float] = {}
    for col in ["exp_bonus_points_mean", "exp_card_points_mean", "exp_penalty_points_mean"]:
        if col in base.columns:
            extras[col] = base[col].astype(float).fillna(0.0)
        else:
            base[col] = 0.0

    comp_cols = [
        "xp_appearance",
        "xp_goals",
        "xp_assists",
        "xp_clean_sheets",
        "xp_concede_penalty",
        "xp_saves_points",
        *extras.keys()
    ]

    base["exp_points_total"] = base[comp_cols].sum(axis=1)

    # Order & write
    out_cols = list(dict.fromkeys([
        *KEY, "player", "pos", "venue",
        "pred_exp_minutes", "p1", "p60",
        "exp_goals","exp_assists","__p_cs__","pred_ga_mean",
        *comp_cols, "exp_points_total"
    ]))
    out_cols = [c for c in out_cols if c in base.columns]

    out = base[out_cols].copy().sort_values(["season","gw_orig","date_played","team_id","player_id"])

    out_dir = args.out_dir / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "expected_points.csv"
    out.to_csv(fp, index=False)
    logging.info("Wrote expected points to %s", fp.resolve())

if __name__ == "__main__":
    main()
