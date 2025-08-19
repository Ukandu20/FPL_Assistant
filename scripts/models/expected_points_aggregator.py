#!/usr/bin/env python3
"""
expected_points_aggregator.py — v3.1 (production)

Combine per-component model outputs into expected FPL points per player–fixture.

Inputs (CSV paths)
------------------
--minutes        minutes_predictions.csv
                 (needs: pred_exp_minutes; optional: prob_played1_cal/_raw, prob_played60_cal/_raw, pos, exp_appearance_points)
--goals-assists  goals_assists_predictions.csv
                 (needs: pred_goals_mean, pred_assists_mean; typically provides pos/player/venue)
--saves          saves_predictions.csv  (GK only; optional: exp_save_points_mean or pred_saves_mean)
--defense        defense_predictions.csv
                 (optional: clean-sheet prob as one of [cs_prob_cal, cs_prob_raw, prob_cs_cal, prob_cs_raw, p_cs, prob_cs];
                            optional: exp_dcp_points (expected goals-conceded penalty points),
                                      or pred_ga_mean to approximate concede penalty)

Key used to align rows (must be present in GA file, others will be aligned to it):
    (season, gw_orig, date_played, player_id, team_id)

Outputs
-------
<out-dir>/<version>/expected_points.csv
<out-dir>/<version>/expected_points.meta.json

Scoring notes
-------------
• Appearance points: E = p1 + p60 (1 for any appearance + 1 for ≥60’). If minutes file provides
  'exp_appearance_points', that value is used; else falls back to p1+p60.
• Clean sheet points include the 60’ requirement: GK/DEF 4, MID 1, FWD 0 → xp_clean_sheets = p60 * p_cs * cs_pts.
• Goal points by position: GK/DEF 6, MID 5, FWD 4. Assist = 3 for all.
• Concede penalty (GK/DEF only): prefer 'exp_dcp_points'; else if 'pred_ga_mean' present, use -0.5 * pred_ga_mean.
• Saves: prefer 'exp_save_points_mean'; else if 'pred_saves_mean' present, use pred_saves_mean/3.

Robustness
----------
• Auto-versioning via --auto-version (creates next vN under --out-dir).
• Optional --write-latest to update `<out-dir>/latest` (symlink or LATEST_VERSION.txt).
• Safe CSV parsing, duplicate key drops, schema normalization, coverage logging.
"""

from __future__ import annotations
import argparse, logging, json, os, re, datetime as dt
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

SCHEMA_VERSION = "v3.1"
KEY: List[str] = ["season", "gw_orig", "date_played", "player_id", "team_id"]

# ───────────────────────── auto-versioning helpers ─────────────────────────

def _resolve_version(base_dir: Path, requested: Optional[str], auto: bool) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    if auto or (not requested) or (requested.lower() == "auto"):
        existing = [p.name for p in base_dir.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
        nxt = (max(int(s[1:]) for s in existing) + 1) if existing else 1
        ver = f"v{nxt}"
        logging.info("[info] auto-version -> %s", ver)
        return ver
    if not re.fullmatch(r"v\d+", requested):
        if requested.isdigit():
            return f"v{requested}"
        raise ValueError("--version must look like v3 (or pass --auto-version)")
    return requested

def _write_latest_pointer(root: Path, version: str) -> None:
    latest = root / "latest"
    target = root / version
    try:
        if latest.exists() or latest.is_symlink():
            try: latest.unlink()
            except Exception: pass
        os.symlink(target.name, latest, target_is_directory=True)
        logging.info("Updated 'latest' symlink -> %s", version)
    except (OSError, NotImplementedError):
        (root / "LATEST_VERSION.txt").write_text(version, encoding="utf-8")
        logging.info("Wrote LATEST_VERSION.txt -> %s", version)

# ───────────────────────── CSV & schema helpers ─────────────────────────

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names / dtypes and keep KEY types consistent."""
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # Ensure datetime for date_played (if present)
    if "date_played" in df.columns:
        df["date_played"] = pd.to_datetime(df["date_played"], errors="coerce")
    # Coerce season/gw types
    if "season" in df.columns:
        df["season"] = df["season"].astype(str)
    if "gw_orig" in df.columns:
        df["gw_orig"] = pd.to_numeric(df["gw_orig"], errors="coerce").astype("Int64")
    # IDs as string
    for c in ["player_id", "team_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def _read_csv(path: Path, need: List[str] | None = None) -> pd.DataFrame:
    # robust parse_dates only if column exists
    try:
        # Try optimistic parse with date_played if present in header
        head = pd.read_csv(path, nrows=0)
        parse_dates = ["date_played"] if "date_played" in head.columns else None
        df = pd.read_csv(path, parse_dates=parse_dates, low_memory=False)
    except Exception:
        df = pd.read_csv(path, low_memory=False)
    df = _norm(df)
    if need:
        miss = [c for c in need if c not in df.columns]
        if miss:
            logging.warning("%s • missing expected columns: %s", path, miss)
    return df

def _drop_dupes(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Drop duplicate rows on KEY (keep last), with a warning."""
    if not set(KEY).issubset(df.columns):
        return df
    before = len(df)
    df2 = df.drop_duplicates(subset=KEY, keep="last").copy()
    d = before - len(df2)
    if d:
        logging.warning("%s • dropped %d duplicate rows on %s", tag, d, KEY)
    return df2

def _coverage(df: pd.DataFrame, cols: List[str], tag: str) -> None:
    present = [c for c in cols if c in df.columns]
    if not present:
        logging.info("%s • coverage: (no requested cols present)", tag)
        return
    stats = {c: float(df[c].notna().mean()) for c in present}
    logging.info("%s • coverage: %s", tag, ", ".join(f"{k}={v:.1%}" for k,v in stats.items()))

def _select_prob(df: pd.DataFrame, cal: str, raw: str, fallback: np.ndarray) -> np.ndarray:
    if cal in df.columns:
        return pd.to_numeric(df[cal], errors="coerce").fillna(0.0).to_numpy(float)
    if raw in df.columns:
        return pd.to_numeric(df[raw], errors="coerce").fillna(0.0).to_numpy(float)
    return fallback

def _choose_first(df: pd.DataFrame, candidates: List[str], default: float | None = None) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")

def _pos_points_vector(pos: pd.Series, kind: str) -> np.ndarray:
    """Return per-row points for goals (kind='goal') or CS (kind='cs') based on position."""
    p = pos.fillna("").astype(str).str.upper().str[:3]
    if kind == "goal":
        return np.where(p.isin(["GKP","GK"]), 6,
               np.where(p.isin(["DEF"]), 6,
               np.where(p.isin(["MID"]), 5,
               np.where(p.isin(["FWD"]), 4, 0)))).astype(float)
    if kind == "cs":
        return np.where(p.isin(["GKP","GK","DEF"]), 4,
               np.where(p.isin(["MID"]), 1, 0)).astype(float)
    raise ValueError("kind must be 'goal' or 'cs'")

def _is_def_or_gk(pos: pd.Series) -> np.ndarray:
    p = pos.fillna("").astype(str).str.upper().str[:3]
    return p.isin(["GKP","GK","DEF"]).to_numpy()

def _ensure_base_identity(base: pd.DataFrame, sources: list[pd.DataFrame], cols: List[str]) -> pd.DataFrame:
    """Ensure columns like player/pos/venue exist on base by pulling from the first source that has them."""
    want = [c for c in cols if c not in base.columns]
    if not want:
        return base
    add_frames = []
    for src in sources:
        if src is None:
            continue
        has = [c for c in want if c in src.columns]
        if not has:
            continue
        add_frames.append(src[KEY + has])
        # remove supplied ones from want
        want = [c for c in want if c not in has]
        if not want:
            break
    if add_frames:
        extra = pd.concat(add_frames).drop_duplicates(subset=KEY, keep="last")
        base = base.merge(extra, on=KEY, how="left")
    # default if still missing
    for c in cols:
        if c not in base.columns:
            base[c] = np.nan
    return base

# ───────────────────────── core ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Aggregate expected FPL points from component models.")
    ap.add_argument("--minutes", required=True, type=Path, help="CSV from minutes model")
    ap.add_argument("--goals-assists", required=True, type=Path, help="CSV from GA model")
    ap.add_argument("--saves", type=Path, default=None, help="CSV from saves model (GK)")
    ap.add_argument("--defense", type=Path, default=None, help="CSV from defense model")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory (root)")
    ap.add_argument("--version", type=str, default=None, help="Subfolder name (e.g., v3). Use --auto-version to pick next.")
    ap.add_argument("--auto-version", action="store_true", help="Pick the next vN under --out-dir automatically")
    ap.add_argument("--write-latest", action="store_true", help="Update <out-dir>/latest pointer")
    ap.add_argument("--log-level", default="INFO", type=str)
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    # Resolve versioned output directory
    version = _resolve_version(args.out_dir, args.version, args.auto_version)
    out_dir = args.out_dir / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read inputs
    ga_need = ["pred_goals_mean", "pred_assists_mean"]
    ga_df = _drop_dupes(_read_csv(args.goals_assists, need=ga_need), "GA")
    if not set(KEY).issubset(ga_df.columns):
        raise ValueError("GA file missing required key columns: " + ", ".join(KEY))
    _coverage(ga_df, KEY + ["pred_goals_mean","pred_assists_mean","pos","player","venue"], "GA")

    min_need = ["pred_exp_minutes"]
    min_df = _drop_dupes(_read_csv(args.minutes, need=min_need), "MIN")
    _coverage(min_df, KEY + ["pred_exp_minutes","prob_played1_cal","prob_played60_cal","prob_played1_raw","prob_played60_raw","exp_appearance_points","pos"], "MIN")

    def_df = None
    if args.defense:
        def_df = _drop_dupes(_read_csv(args.defense), "DEF")
        _coverage(def_df, KEY + ["cs_prob_cal","cs_prob_raw","prob_cs_cal","prob_cs_raw","p_cs","prob_cs","exp_dcp_points","pred_ga_mean"], "DEF")

    sv_df = None
    if args.saves:
        sv_df = _drop_dupes(_read_csv(args.saves), "SAV")
        _coverage(sv_df, KEY + ["exp_save_points_mean","pred_saves_mean"], "SAV")

    # Base frame: GA is authoritative for row set and identity fields
    base = ga_df.copy()

    # Merge minutes: bring expected minutes and p1/p60
    m_exp = pd.to_numeric(min_df.get("pred_exp_minutes", pd.Series(np.nan, index=min_df.index)), errors="coerce").fillna(0.0).to_numpy(float)
    fallback_p1  = (m_exp > 0).astype(float)
    fallback_p60 = np.clip(m_exp / 90.0, 0.0, 1.0)

    min_df["__p1__"]  = _select_prob(min_df, "prob_played1_cal",  "prob_played1_raw",  fallback_p1)
    min_df["__p60__"] = _select_prob(min_df, "prob_played60_cal", "prob_played60_raw", fallback_p60)

    base = base.merge(min_df[KEY + [c for c in ["pred_exp_minutes","__p1__","__p60__","exp_appearance_points","pos"] if c in min_df.columns]],
                      on=KEY, how="left", validate="one_to_one")

    # Ensure identity columns exist (player, pos, venue), preferring GA → MIN
    base = _ensure_base_identity(base, [ga_df, min_df, def_df, sv_df], cols=["player","pos","venue"])

    # Clean sheet probability
    if def_df is not None:
        def_tmp = def_df.copy()
        cs_candidates = ["cs_prob_cal","cs_prob_raw","prob_cs_cal","prob_cs_raw","p_cs","prob_cs"]
        def_tmp["__p_cs__"] = _choose_first(def_tmp, cs_candidates, default=np.nan)
        keep_cols = [c for c in ["__p_cs__", "exp_dcp_points", "pred_ga_mean"] if c in def_tmp.columns or c == "__p_cs__"]
        base = base.merge(def_tmp[KEY + keep_cols], on=KEY, how="left", validate="one_to_one")
    else:
        base["__p_cs__"] = np.nan

    # Saves points
    if sv_df is not None:
        tmp = sv_df.copy()
        if "exp_save_points_mean" in tmp.columns:
            tmp["__xp_saves__"] = pd.to_numeric(tmp["exp_save_points_mean"], errors="coerce")
        elif "pred_saves_mean" in tmp.columns:
            tmp["__xp_saves__"] = pd.to_numeric(tmp["pred_saves_mean"], errors="coerce") / 3.0
        else:
            tmp["__xp_saves__"] = np.nan
        base = base.merge(tmp[KEY + ["__xp_saves__"]], on=KEY, how="left", validate="one_to_one")
    else:
        base["__xp_saves__"] = np.nan

    # Appearance probabilities and points
    base["p1"]  = pd.to_numeric(base.get("__p1__"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    base["p60"] = pd.to_numeric(base.get("__p60__"), errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # If upstream minutes provided exp_appearance_points, prefer it; otherwise p1+p60
    if "exp_appearance_points" in base.columns:
        base["xp_appearance"] = pd.to_numeric(base["exp_appearance_points"], errors="coerce").fillna(base["p1"] + base["p60"])
    else:
        base["xp_appearance"] = base["p1"] + base["p60"]

    # Goals & assists expectations
    goal_mean_candidates   = ["pred_goals_mean", "pred_goals_poisson", "exp_goals_mean"]
    assist_mean_candidates = ["pred_assists_mean", "pred_assists_poisson", "exp_assists_mean"]

    base["exp_goals"]   = _choose_first(base, goal_mean_candidates, default=0.0).clip(lower=0.0)
    base["exp_assists"] = _choose_first(base, assist_mean_candidates, default=0.0).clip(lower=0.0)

    goal_pts_vec = _pos_points_vector(base["pos"], kind="goal")
    base["xp_goals"]   = base["exp_goals"]   * goal_pts_vec
    base["xp_assists"] = base["exp_assists"] * 3.0

    # Clean sheet points (require 60+)
    cs_pts_vec = _pos_points_vector(base["pos"], kind="cs")
    p_cs = pd.to_numeric(base.get("__p_cs__"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    base["xp_clean_sheets"] = base["p60"] * p_cs * cs_pts_vec

    # Concede penalty (GK/DEF only)
    is_def_gk = _is_def_or_gk(base["pos"])
    if "exp_dcp_points" in base.columns:
        base["xp_concede_penalty"] = np.where(is_def_gk, pd.to_numeric(base["exp_dcp_points"], errors="coerce").fillna(0.0), 0.0)
    elif "pred_ga_mean" in base.columns:
        base["xp_concede_penalty"] = np.where(is_def_gk, -0.5 * pd.to_numeric(base["pred_ga_mean"], errors="coerce").fillna(0.0), 0.0)
    else:
        base["xp_concede_penalty"] = 0.0

    # Saves points (GK only)
    is_gk = base["pos"].fillna("").astype(str).str.upper().str.startswith("GK")
    base["xp_saves_points"] = np.where(is_gk, pd.to_numeric(base["__xp_saves__"], errors="coerce").fillna(0.0), 0.0)

    # Optional extras if already computed upstream (bonus/cards/pens)
    for col in ["exp_bonus_points_mean", "exp_card_points_mean", "exp_penalty_points_mean"]:
        if col not in base.columns:
            base[col] = 0.0
        else:
            base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)

    # Sum components
    comp_cols = [
        "xp_appearance",
        "xp_goals",
        "xp_assists",
        "xp_clean_sheets",
        "xp_concede_penalty",
        "xp_saves_points",
        "exp_bonus_points_mean",
        "exp_card_points_mean",
        "exp_penalty_points_mean",
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

    fp = out_dir / "expected_points.csv"
    out.to_csv(fp, index=False)
    logging.info("Wrote expected points to %s  (rows=%d)", fp.resolve(), len(out))

    # Meta file (useful lineage for downstream reproducibility)
    meta = {
        "schema": SCHEMA_VERSION,
        "build_ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "minutes": str(args.minutes),
            "goals_assists": str(args.goals_assists),
            "saves": str(args.saves) if args.saves else None,
            "defense": str(args.defense) if args.defense else None,
        },
        "version": version,
        "features_used": {
            "goals": goal_mean_candidates,
            "assists": assist_mean_candidates,
            "clean_sheet_prob_candidates": ["cs_prob_cal","cs_prob_raw","prob_cs_cal","prob_cs_raw","p_cs","prob_cs"],
            "saves": ["exp_save_points_mean", "pred_saves_mean/3"],
            "concede_penalty": ["exp_dcp_points", "pred_ga_mean → -0.5 * ga"],
            "appearance": ["exp_appearance_points", "p1+p60 (fallback)"]
        },
        "component_columns": comp_cols,
        "key": KEY
    }
    (out_dir / "expected_points.meta.json").write_text(json.dumps(meta, indent=2))

    if args.write_latest:
        _write_latest_pointer(args.out_dir, version)

if __name__ == "__main__":
    main()
