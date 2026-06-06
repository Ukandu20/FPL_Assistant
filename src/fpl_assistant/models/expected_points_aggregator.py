#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
expected_points_aggregator.py — v4.2.0 (probability-first, versioned IO, pandas-safe)

What this does
--------------
Combines component model outputs (minutes, goals/assists, defense, saves) into
expected FPL points per player–fixture (per GW key), using a probability-first policy.
Optionally attaches ACTUALS from a player fixture calendar (fbref_id + total_points)
for side-by-side comparisons.

Locked policies
---------------
1) Appearance points (minutes):
   xp_appearance = p_play + p60
     • p_play := P(plays ≥1')
     • p60    := P(plays ≥60')
   Fallbacks if missing:
     • p_play := p_start + p_cameo (clipped to [0,1])
     • p60    := clip(pred_minutes/90, 0, 1)
   Sanity: p60 ≤ p_play

2) Goals / Assists from "at least one" probabilities:
   λ_goal   = -ln(1 - p_goal)
   λ_assist = -ln(1 - p_assist)
   xp_goals_points   = λ_goal   * points_per_goal(position)
   xp_assists_points = 3 * λ_assist
   Fallbacks: use pred_goals_mean / pred_assists_mean if p_* missing.

3) Clean sheets:
   xp_clean_sheets = p60 * prob_cs * cs_points(position)
   (prob_cs pulled from any of: cs_prob_cal, cs_prob_raw, prob_cs_cal, prob_cs_raw, p_cs, prob_cs)

4) DCP (2-pt bonus; outfield only):
   xp_dcp_bonus = 2 * prob_dcp

5) Saves (GK only):
   xp_saves_points = p_play * E[floor(S/3)], S ~ Poisson(λ = pred_saves_poisson)

6) Concede penalty (GK/DEF only):
   If λ_GA available (pred_ga_mean or lambda_ga columns), use:
   xp_concede_penalty_points = - E[floor(GA/2)] * exposure
     • exposure = clip(pred_minutes/90, 0, 1), else fallback to p_play

ACTUALS attach (optional)
-------------------------
If --actuals is provided (e.g.,
  data/processed/registry/fixtures/2024-2025/player_fixture_calendar.csv),
we read that file and aggregate per KEY=["season","gw_orig","player_id","team_id"]:
  • total_points := sum across fixtures in a DGW (so it’s comparable to per-GW xPts)
  • fbref_id     := first non-null

If the CSV lacks a 'season' column, we try to parse it from the path (e.g., "2024-2025").
If still absent, we fallback to joining on ["gw_orig","player_id","team_id"].

Keys & Output
-------------
• Join KEY: ["season","gw_orig","player_id","team_id"]
• Output CSV written to: <out-dir>/<version>/expected_points.csv
• If --write-latest is set, updates <out-dir>/latest -> <version> pointer.

CLI
---
python expected_points_aggregator.py \
  --minutes data/models/minutes/versions/v1/expected_minutes.csv \
  --goals-assists data/models/goals_assists/versions/v1/goals_assists_predictions.csv \
  --defense data/models/defense/v1/predictions/expected_defense.csv \
  --saves data/models/saves/versions/v1/saves_predictions.csv \
  --actuals data/processed/registry/fixtures/2024-2025/player_fixture_calendar.csv \
  --out-dir data/models/expected_points \
  --version v1 --write-latest --log-level INFO
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = "xp.v4.2.0"
KEY: list[str] = ["season","gw_orig","player_id","team_id"]

GOAL_POINTS = {
    "GK": 6, "GKP": 6,
    "DEF": 6, "D": 6, "DF": 6,
    "MID": 5, "M": 5,
    "FWD": 4, "FW": 4, "F": 4,
}
CS_POINTS = {
    "GK": 4, "GKP": 4,
    "DEF": 4, "D": 4, "DF": 4,
    "MID": 1, "M": 1,
    "FWD": 0, "FW": 0, "F": 0,
}

# ───────────────────────── versioning & pointers ─────────────────────────

def _resolve_version(base_dir: Path, requested: Optional[str], auto: bool) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    if auto or not requested or requested.lower() == "auto":
        existing = [p.name for p in base_dir.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
        nxt = (max(int(s[1:]) for s in existing) + 1) if existing else 1
        ver = f"v{nxt}"
        logging.info("[version] auto -> %s", ver)
        return ver
    if re.fullmatch(r"v\d+", requested):
        return requested
    if requested.isdigit():
        return f"v{requested}"
    raise ValueError("--version must look like v3 (or use --auto-version)")

def _write_latest_pointer(root: Path, version: str) -> None:
    latest = root / "latest"
    target = root / version
    try:
        if latest.exists() or latest.is_symlink():
            try:
                latest.unlink()
            except Exception:
                pass
        os.symlink(target.name, latest, target_is_directory=True)
        logging.info("[latest] symlink -> %s", version)
    except (OSError, NotImplementedError):
        (root / "LATEST_VERSION.txt").write_text(version, encoding="utf-8")
        logging.info("[latest] wrote LATEST_VERSION.txt -> %s", version)

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)

# ───────────────────────── small helpers ─────────────────────────

def _read_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        logging.warning("Input not found: %s", p)
        return None
    head = pd.read_csv(p, nrows=0)
    parse_dates = [c for c in ("date_played","date_sched","kickoff_time") if c in head.columns]
    return pd.read_csv(p, low_memory=False, parse_dates=parse_dates)

def _read_csv_or_empty(path: Optional[Path]) -> pd.DataFrame:
    df = _read_csv(path)
    return df if df is not None else pd.DataFrame()

def _norm_key_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in KEY:
        if c not in df.columns:
            continue
        if c == "gw_orig":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        else:
            df[c] = df[c].astype(str)
    return df

def _pos_series_to_goal_points(pos: pd.Series) -> pd.Series:
    pos = pos.fillna("").astype(str).str.upper()
    return pos.map(GOAL_POINTS).fillna(0.0)

def _pos_series_to_cs_points(pos: pd.Series) -> pd.Series:
    pos = pos.fillna("").astype(str).str.upper()
    return pos.map(CS_POINTS).fillna(0.0)

def _lambda_from_p(p: pd.Series | np.ndarray) -> np.ndarray:
    p = pd.to_numeric(p, errors="coerce").to_numpy(dtype=float)
    p = np.clip(p, 0.0, 1.0)
    eps = 1e-12
    return -np.log(np.clip(1.0 - p, eps, 1.0))

def _expected_floor_div_k_poisson(lam: float, k: int) -> float:
    """E[floor(N/k)] for N~Poisson(lam) via tail-sum ∑ P(N ≥ k*n)."""
    if not np.isfinite(lam) or lam <= 0:
        return 0.0
    tol = 1e-12
    max_n = int(max(30, math.ceil((lam + 10.0 * math.sqrt(lam)) / k)))
    K = k * max_n
    p0 = math.exp(-lam)
    cdf = p0
    pmf_prev = p0
    E = 0.0
    next_mult = k
    for j in range(1, K+1):
        pmf_j = pmf_prev * (lam / j)
        cdf += pmf_j
        pmf_prev = pmf_j
        if j == next_mult:
            tail_k = 1.0 - (cdf - pmf_j)  # P(N ≥ k*n)
            E += tail_k
            next_mult += k
            if tail_k < tol and j > lam:
                pass
    return float(E)

_vec_E_floor_div2 = np.frompyfunc(lambda lam: _expected_floor_div_k_poisson(float(lam), 2), 1, 1)
_vec_E_floor_div3 = np.frompyfunc(lambda lam: _expected_floor_div_k_poisson(float(lam), 3), 1, 1)

def _choose_first(df: pd.DataFrame, cols: Sequence[str], default=np.nan) -> pd.Series:
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series(default, index=df.index, dtype=float)

def _safe_merge(left: pd.DataFrame, right: Optional[pd.DataFrame], on: Sequence[str]) -> pd.DataFrame:
    if right is None or right.empty:
        return left
    right = right.drop_duplicates(subset=on, keep="last")
    return left.merge(right, on=list(on), how="left")

def _extract_season_from_path(p: Path) -> Optional[str]:
    m = re.search(r"(\d{4}-\d{4})", str(p))
    return m.group(1) if m else None

# ───────────────────────── core aggregation ─────────────────────────

def aggregate_points(minutes_csv: Path,
                     ga_csv: Path,
                     defense_csv: Optional[Path] = None,
                     saves_csv: Optional[Path] = None,
                     actuals_csv: Optional[Path] = None) -> pd.DataFrame:
    # read safely
    min_df = _read_csv_or_empty(minutes_csv)
    ga_df  = _read_csv_or_empty(ga_csv)
    def_df = _read_csv(defense_csv) if defense_csv else None
    sv_df  = _read_csv(saves_csv) if saves_csv else None

    # normalize keys
    min_df = _norm_key_types(min_df)
    ga_df  = _norm_key_types(ga_df)
    if def_df is not None: def_df = _norm_key_types(def_df)
    if sv_df  is not None: sv_df  = _norm_key_types(sv_df)

    # base from minutes; include any GA rows missing from minutes
    if not all(c in min_df.columns for c in KEY):
        raise ValueError(f"MINUTES CSV missing key columns for required KEY: {KEY}")

    base = min_df[KEY].drop_duplicates().copy()
    if not ga_df.empty:
        missing = ga_df[KEY].drop_duplicates().merge(base, on=KEY, how="left", indicator=True)
        missing = missing.loc[missing["_merge"]=="left_only", KEY]
        if not missing.empty:
            base = pd.concat([base, missing], ignore_index=True).drop_duplicates(KEY)

    # attach identity (prefer GA, then minutes, then defense/saves if present)
    # NOTE: do NOT include any *_id fields here except what's already in KEY
    identity_cols = ["player","pos","date_played","gw_played","fdr"]
    for src in (ga_df, min_df, def_df, sv_df):
        if src is None or src.empty:
            continue
        keep = [c for c in identity_cols if c in src.columns]
        if keep:
            select_cols = KEY + [c for c in keep if c not in KEY]
            base = base.merge(src[select_cols], on=KEY, how="left", suffixes=("","_r"))

    # safer duplicate collapse: only known merge suffixes; never touch *_id keys
    for c in identity_cols:
        pattern = re.compile(rf"^{re.escape(c)}(_r|_[xy])$")
        dup_cols = [col for col in base.columns if col == c or pattern.match(col)]
        if len(dup_cols) > 1:
            base[c] = base[dup_cols].bfill(axis=1).iloc[:, 0]
            base = base.drop(columns=[col for col in dup_cols if col != c])

    # ensure KEY still present
    missing_keys = [k for k in KEY if k not in base.columns]
    if missing_keys:
        raise KeyError(
            f"Internal error: base lost key columns {missing_keys} after identity collapse. "
            f"Columns now: {list(base.columns)[:30]}..."
        )

    # minutes → p_play, p60, pred_minutes
    if "p_play" in min_df.columns:
        p_play = pd.to_numeric(min_df["p_play"], errors="coerce")
    else:
        ps = pd.to_numeric(min_df.get("p_start", 0.0), errors="coerce").fillna(0.0)
        pc = pd.to_numeric(min_df.get("p_cameo", 0.0), errors="coerce").fillna(0.0)
        p_play = np.clip(ps + pc, 0.0, 1.0)

    if "p60" in min_df.columns:
        p60 = pd.to_numeric(min_df["p60"], errors="coerce")
    else:
        m = pd.to_numeric(min_df.get("pred_minutes", 0.0), errors="coerce").fillna(0.0)
        p60 = np.clip(m / 90.0, 0.0, 1.0)

    p60 = np.minimum(p60, p_play)

    min_probs = min_df[KEY].copy()
    min_probs["p_play"] = p_play.values
    min_probs["p60"] = p60.values
    if "pred_minutes" in min_df.columns:
        min_probs["pred_minutes"] = pd.to_numeric(min_df["pred_minutes"], errors="coerce")
    else:
        min_probs["pred_minutes"] = np.nan

    base = _safe_merge(base, min_probs, KEY)

    # appearance points
    base["xp_appearance"] = base["p_play"].fillna(0.0) + base["p60"].fillna(0.0)
    if "exp_minutes_points" in base.columns:
        base["exp_minutes_points"] = base["xp_appearance"]

    # GA probs → implied λ; fallback to means
    if not ga_df.empty:
        ga_tmp = ga_df.copy()
        ga_tmp["__p_goal__"] = _choose_first(ga_tmp, ["p_goal","prob_goal","p_goal_cal","prob_goal_cal"], default=np.nan)
        ga_tmp["__p_assist__"] = _choose_first(ga_tmp, ["p_assist","prob_assist","p_assist_cal","prob_assist_cal"], default=np.nan)
        ga_tmp["__xg_mean__"] = pd.to_numeric(ga_tmp.get("pred_goals_mean", np.nan), errors="coerce")
        ga_tmp["__xa_mean__"] = pd.to_numeric(ga_tmp.get("pred_assists_mean", np.nan), errors="coerce")
        base = _safe_merge(base, ga_tmp[KEY + ["__p_goal__","__p_assist__","__xg_mean__","__xa_mean__"]], KEY)
    else:
        base["__p_goal__"] = np.nan
        base["__p_assist__"] = np.nan
        base["__xg_mean__"] = np.nan
        base["__xa_mean__"] = np.nan

    lam_goal = _lambda_from_p(base["__p_goal__"])
    lam_ass  = _lambda_from_p(base["__p_assist__"])
    lam_goal = np.where(np.isfinite(lam_goal) & (lam_goal > 0),
                        lam_goal,
                        pd.to_numeric(base["__xg_mean__"], errors="coerce").to_numpy())
    lam_ass  = np.where(np.isfinite(lam_ass) & (lam_ass > 0),
                        lam_ass,
                        pd.to_numeric(base["__xa_mean__"], errors="coerce").to_numpy())

    goal_pts_vec = _pos_series_to_goal_points(base.get("pos", pd.Series(index=base.index)))
    base["xp_goals_points"]   = goal_pts_vec.to_numpy(dtype=float) * np.nan_to_num(lam_goal, nan=0.0)
    base["xp_assists_points"] = 3.0 * np.nan_to_num(lam_ass, nan=0.0)

    # Clean sheets + DCP
    if def_df is not None and not def_df.empty:
        def_tmp = def_df.copy()
        def_tmp["prob_cs"] = _choose_first(def_tmp,
            ["cs_prob_cal","cs_prob_raw","prob_cs_cal","prob_cs_raw","p_cs","prob_cs"], default=np.nan)
        keep = [c for c in ["prob_cs","prob_dcp","pred_ga_mean","lambda_goals_against","lambda_ga","ga_lambda","lambda_concede"] if c in def_tmp.columns]
        base = _safe_merge(base, def_tmp[KEY + keep], KEY)
    if "prob_cs" not in base.columns: base["prob_cs"] = np.nan
    if "prob_dcp" not in base.columns: base["prob_dcp"] = np.nan

    cs_pts_vec = _pos_series_to_cs_points(base.get("pos", pd.Series(index=base.index)))
    base["xp_clean_sheets"] = cs_pts_vec.to_numpy(dtype=float) * base["p60"].fillna(0.0) * pd.to_numeric(base["prob_cs"], errors="coerce").fillna(0.0)

    is_gk = base.get("pos", pd.Series(index=base.index)).fillna("").astype(str).str.upper().str.startswith(("GK","GKP"))
    base["xp_dcp_bonus"] = np.where(is_gk, 0.0,
                                    2.0 * pd.to_numeric(base.get("prob_dcp", np.nan), errors="coerce").fillna(0.0))

    # Saves (GK)
    if sv_df is not None and not sv_df.empty:
        sv_tmp = sv_df.copy()
        lam_saves = None
        if "pred_saves_poisson" in sv_tmp.columns:
            lam_saves = pd.to_numeric(sv_tmp["pred_saves_poisson"], errors="coerce")
        elif "pred_saves_mean" in sv_tmp.columns:
            lam_saves = pd.to_numeric(sv_tmp["pred_saves_mean"], errors="coerce")
        if lam_saves is not None:
            sv_keep = sv_tmp[KEY].copy()
            sv_keep["__lam_saves__"] = lam_saves
            base = _safe_merge(base, sv_keep, KEY)
    if "__lam_saves__" in base.columns:
        lamS = pd.to_numeric(base["__lam_saves__"], errors="coerce").to_numpy()
        E_floorS3 = _vec_E_floor_div3(lamS).astype(float)
        base["xp_saves_points"] = np.where(is_gk, base["p_play"].fillna(0.0).to_numpy() * E_floorS3, 0.0)
    else:
        base["xp_saves_points"] = 0.0

    # Goals conceded penalty (GK/DEF)
    lam_ga_col = None
    for name in ("pred_ga_mean","lambda_goals_against","lambda_ga","ga_lambda","lambda_concede"):
        if name in base.columns:
            lam_ga_col = name
            break
    if lam_ga_col:
        lamGA = pd.to_numeric(base[lam_ga_col], errors="coerce").to_numpy()
        E_floorGA2 = _vec_E_floor_div2(lamGA).astype(float)
        exposure = pd.to_numeric(base.get("pred_minutes", np.nan), errors="coerce").to_numpy()
        exposure = np.clip(exposure / 90.0, 0.0, 1.0)
        exposure = np.where(np.isnan(exposure), base["p_play"].fillna(0.0).to_numpy(), exposure)
        is_def = base.get("pos", pd.Series(index=base.index)).fillna("").astype(str).str.upper().str.startswith(("GK","GKP","D","DF","DEF"))
        base["xp_concede_penalty_points"] = np.where(is_def, - E_floorGA2 * exposure, 0.0)
    else:
        base["xp_concede_penalty_points"] = 0.0

    # Optional: attach actuals (fbref_id + total_points per KEY); handle missing 'season' in the CSV
    if actuals_csv is not None:
        act = _read_csv(actuals_csv)
        if act is not None and not act.empty:
            # add season if absent
            if "season" not in act.columns:
                guess = _extract_season_from_path(Path(actuals_csv))
                if guess:
                    act = act.copy()
                    act["season"] = guess
                    logging.info("[actuals] inferred season='%s' from path for %s", guess, actuals_csv)
                else:
                    logging.warning("[actuals] no 'season' column and could not infer from path; will join on 3-col key")

            # normalize key types; build 4-col or 3-col grouping
            act = _norm_key_types(act)
            join_key = [k for k in KEY if k in act.columns]
            if len(join_key) < 3 or "player_id" not in join_key or "team_id" not in join_key or "gw_orig" not in join_key:
                logging.warning("[actuals] missing required key cols to merge actuals; skipping attach")
            else:
                # Aggregate per KEY (sum total_points across DGWs), take first fbref_id
                agg = (act.groupby(join_key, dropna=False)
                           .agg(total_points=("total_points","sum"),
                                fbref_id=("fbref_id","first"))
                           .reset_index())
                before = base["total_points"].notna().sum() if "total_points" in base.columns else 0
                base = _safe_merge(base, agg[join_key + ["total_points","fbref_id"]], join_key)
                after = base["total_points"].notna().sum()
                logging.info("[actuals] attached total_points/fbref_id for %d rows (prev had %d)", after, before)
        else:
            logging.warning("[actuals] provided path has no rows: %s", actuals_csv)

    # Total & formatting
    components = [
        "xp_appearance",
        "xp_goals_points",
        "xp_assists_points",
        "xp_clean_sheets",
        "xp_saves_points",
        "xp_dcp_bonus",
        "xp_concede_penalty_points",
    ]
    for c in components:
        if c not in base.columns:
            base[c] = 0.0
    base["xPts"] = base[components].sum(axis=1)

    # column order
    meta_cols = [c for c in ["season","gw_orig","gw_played","date_played","player_id","player","team_id","pos","fdr","fbref_id","total_points"] if c in base.columns]
    prob_cols = [c for c in ["p_play","p60","prob_cs","prob_dcp","__p_goal__","__p_assist__"] if c in base.columns]
    model_cols = [c for c in ["pred_minutes","__xg_mean__","__xa_mean__","__lam_saves__"] if c in base.columns]
    key_first = [c for c in KEY if c in base.columns]
    out_cols = list(dict.fromkeys(key_first + meta_cols + components + ["xPts"] + prob_cols + model_cols))
    base = base[out_cols].copy()

    # rounding (xPts to 1dp, others 2dp)
    for c in base.columns:
        if pd.api.types.is_float_dtype(base[c]):
            base[c] = base[c].round(1 if c == "xPts" else 2)

    base.attrs["schema_version"] = SCHEMA_VERSION
    return base

# ───────────────────────── CLI ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Aggregate expected FPL points (probability-first).")
    ap.add_argument("--minutes", required=True, type=Path)
    ap.add_argument("--goals-assists", required=True, type=Path)
    ap.add_argument("--defense", type=Path, default=None)
    ap.add_argument("--saves", type=Path, default=None)
    ap.add_argument("--actuals", type=Path, default=None,
                    help="Optional path to player_fixture_calendar.csv to attach fbref_id and total_points (per KEY).")
    # Support both --out-dir (preferred) and legacy --out
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="(legacy) same as --out-dir")
    ap.add_argument("--version", type=str, default=None)
    ap.add_argument("--auto-version", action="store_true")
    ap.add_argument("--write-latest", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    out_dir = args.out_dir or args.out
    if out_dir is None:
        ap.error("Provide --out-dir (or legacy --out).")

    version = _resolve_version(out_dir, args.version, args.auto_version)
    version_dir = out_dir / version
    out_path = version_dir / "expected_points.csv"

    df = aggregate_points(args.minutes, args.goals_assists, args.defense, args.saves, args.actuals)

    _atomic_write_csv(df, out_path)
    logging.info("Wrote %d rows to %s", len(df), out_path)

    if args.write_latest:
        _write_latest_pointer(out_dir, version)

if __name__ == "__main__":
    main()
