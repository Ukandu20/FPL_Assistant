#!/usr/bin/env python3
r"""team_form_builder.py  – schema v2.8 (prefer flags for sides, diagnostics, FDR 2–5, hybrid bucketing, EWMA)

Inputs (per season):
  data/processed/fixtures/<SEASON>/fixture_calendar.csv
  Required (base) columns (values may contain NaNs for future fixtures):
    fpl_id, fbref_id, team_id, team, gw_orig, date_played, date_sched,
    gf, ga, xg, xga, poss, result, home_id, away_id
  Optional identity columns:
    opponent_id, is_home, is_away, venue

Changes (v2.8):
• Prefer using explicit is_home flags to split sides; fall back to id equality if flags unusable.
• Rich diagnostics when 0 fixtures form (counts of home/away rows).
• FDR difficulty buckets are 2..5 (5 = hardest). Hybrid bucketing (GW<cutoff uses fixed z; else season pct).
• Stable join keys; relaxed merge fallback to (season,fpl_id) if unique.
• EWMA option for rolling stats (past-only; no leakage). Priors for early weeks.

Outputs (per season):
  data/processed/features/<version>/<SEASON>/team_form.csv
  data/processed/features/<version>/<SEASON>/team_form.meta.json
"""

from __future__ import annotations
import argparse, json, logging, datetime as dt, hashlib, os, re, shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from functools import reduce

import numpy as np
import pandas as pd

SCHEMA_VERSION = "v2.8"

# ───────────────────────── config ─────────────────────────

METRICS = {
    "att": {"raw": ("gf", "xg")},
    "def": {"raw": ("ga", "xga")},
    "pos": {"raw": ("poss",)},
}

REQUIRED_BASE = {
    "fpl_id", "fbref_id", "team_id", "team", "gw_orig",
    "home_id", "away_id", "date_played", "date_sched",
    "gf", "ga", "xg", "xga", "poss", "result",
}

OUTPUT_FILE = "team_form.csv"
META_FILE = "team_form.meta.json"

LEGACY_FDR = [
    "fdr_home","fdr_away",
    "fdr_att_home","fdr_def_home","fdr_att_away","fdr_def_away",
    "fdr_home_cont","fdr_away_cont",
    "fdr_att_home_cont","fdr_def_home_cont","fdr_att_away_cont","fdr_def_away_cont",
]

DEFAULT_MEANS = {"gf": 1.40, "xg": 1.45, "ga": 1.40, "xga": 1.45, "poss": 50.0}

# ───────────────────────── version helpers ─────────────────────────

def _resolve_version(features_root: Path, requested: Optional[str], auto: bool) -> str:
    if auto or (not requested) or (requested.lower() == "auto"):
        existing = [p.name for p in features_root.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
        nxt = (max(int(p[1:]) for p in existing) + 1) if existing else 1
        ver = f"v{nxt}"
        logging.info("Auto-version resolved to %s", ver)
        return ver
    if not re.fullmatch(r"v\d+", requested):
        if requested and requested.isdigit():
            return f"v{requested}"
        raise ValueError(f"--version must be like v9 or a number; got {requested}")
    return requested

def _write_latest_pointer(features_root: Path, version: str) -> None:
    latest = features_root / "latest"
    target = features_root / version
    try:
        if latest.exists() or latest.is_symlink():
            try:
                latest.unlink()
            except Exception:
                pass
        os.symlink(target.name, latest, target_is_directory=True)
        logging.info("Updated 'latest' symlink -> %s", version)
    except (OSError, NotImplementedError):
        (features_root / "LATEST_VERSION.txt").write_text(version, encoding="utf-8")
        logging.info("Wrote LATEST_VERSION.txt -> %s", version)

def _copy_to_latest_dir(features_root: Path, version: str, season: str) -> None:
    src = features_root / version / season
    dst = features_root / "latest" / season
    dst.mkdir(parents=True, exist_ok=True)
    for fname in (OUTPUT_FILE, META_FILE):
        fp = src / fname
        if fp.exists():
            shutil.copy2(fp, dst / fname)

# ───────────────────────── utils ─────────────────────────

def load_json(p: Path) -> dict:
    return json.loads(p.read_text("utf-8")) if p.is_file() else {}

def save_json(obj: dict, p: Path) -> None:
    p.write_text(json.dumps(obj, indent=2))

def _meta_hash(df: pd.DataFrame) -> str:
    cols = sorted([c for c in df.columns if c.startswith(("att_","def_","fdr_","pos_"))])
    base = ["season","fpl_id","team_id","opponent_id"] if "opponent_id" in df.columns else ["season","fpl_id","team_id"]
    sample = df[base + cols].head(500).to_json(orient="split", index=False)
    return hashlib.sha256(sample.encode("utf-8")).hexdigest()[:12]

def _seasons_before(all_seasons: List[str], current: str, k: int = 3) -> List[str]:
    prior = [s for s in all_seasons if s < current]
    prior.sort()
    return prior[-k:]

def _safe_means(frame: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    out = {}
    for c in cols:
        v = pd.to_numeric(frame[c], errors="coerce")
        m = float(v.mean()) if v.notna().any() else DEFAULT_MEANS.get(c, 0.0)
        out[c] = m
    return out

def build_team_priors(cal_all: pd.DataFrame, season: str, all_seasons: List[str], cols: List[str], beta: float = 0.7) -> pd.DataFrame:
    prev_seasons = _seasons_before(all_seasons, season, k=3)
    cur_ids = cal_all.loc[cal_all["season"] == season, "team_id"].dropna().unique()

    if not prev_seasons:
        base = pd.DataFrame([DEFAULT_MEANS], index=[0])[cols]
        base = base.reindex(cur_ids).ffill().bfill()
        base.index = cur_ids
        return base

    stacks = []
    for i, s in enumerate(prev_seasons):
        w = beta ** (len(prev_seasons) - 1 - i)
        hist = cal_all[cal_all["season"] == s]
        if hist.empty:
            continue
        g = hist.groupby("team_id")[cols].mean().mul(w)
        stacks.append(g)

    if not stacks:
        base = pd.DataFrame([DEFAULT_MEANS], index=[0])[cols]
        base = base.reindex(cur_ids).ffill().bfill()
        base.index = cur_ids
        return base

    prior = reduce(lambda a, b: a.add(b, fill_value=0.0), stacks).reindex(cur_ids)

    league_prev = pd.concat([cal_all[cal_all["season"] == s] for s in prev_seasons], ignore_index=True)
    league_means = _safe_means(league_prev, cols)
    prior = prior.fillna(pd.Series(league_means))
    for c in cols:
        prior[c] = prior[c].fillna(DEFAULT_MEANS.get(c, 0.0))
    return prior

# ───────────────────────── schema normalization ─────────────────────────

def _is_blank_series(s: pd.Series) -> pd.Series:
    if s.dtype == 'O' or pd.api.types.is_string_dtype(s):
        return s.isna() | s.str.strip().eq("") | s.str.lower().isin({"nan", "none", "<na>"})
    return s.isna()

def _coerce_poss(df: pd.DataFrame) -> pd.DataFrame:
    df["poss"] = pd.to_numeric(df["poss"], errors="coerce")
    if df["poss"].max(skipna=True) is not None and df["poss"].max(skipna=True) <= 1.5:
        df["poss"] = df["poss"] * 100.0
    df["poss"] = df["poss"].fillna(50.0)
    return df

def _require_base(df: pd.DataFrame, season: str) -> None:
    missing_base = REQUIRED_BASE - set(df.columns)
    if missing_base:
        raise KeyError(f"{season}: fixture_calendar missing base columns: {missing_base}")

def _validate_or_infer_from_ids(frame: pd.DataFrame, season: str, strict: bool) -> pd.DataFrame:
    """
    Prefer PROVIDED flags (is_home/is_away/venue). Fall back to id equality only when flags are missing.
    Never rewrite given home_id/away_id. Ensure opponent_id and venue.
    """
    df = frame.copy()

    # Normalize id types
    for c in ["team_id","opponent_id","home_id","away_id","fpl_id","fbref_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()

    # Provided flags (highest priority)
    if "is_home" in df.columns:
        provided = pd.to_numeric(df["is_home"], errors="coerce").astype("Float64")
    elif "is_away" in df.columns:
        provided = (pd.to_numeric(df["is_away"], errors="coerce") == 0).astype("Float64")
    elif "venue" in df.columns:
        provided = df["venue"].astype(str).str.strip().str.lower().map({"home":1.0,"away":0.0}).astype("Float64")
    else:
        provided = pd.Series(pd.NA, index=df.index, dtype="Float64")

    # Fallback: infer from id equality only where provided is NA
    eq_home = df["team_id"] == df["home_id"]
    eq_away = df["team_id"] == df["away_id"]
    inferred = pd.Series(
        np.where(eq_home & ~eq_away, 1, np.where(eq_away & ~eq_home, 0, np.nan)),
        index=df.index, dtype="Float64"
    )

    is_home = provided.where(provided.notna(), inferred)

    # If still NaN, warn (or error strictly) and default to Home
    n_amb = int(is_home.isna().sum())
    if n_amb:
        msg = f"{season}: {n_amb} rows ambiguous is_home (flags/ids missing)."
        if strict:
            raise AssertionError(msg)
        logging.warning(msg + " Setting venue='Home' by default for ambiguous rows.")
        is_home = is_home.fillna(1.0)

    is_home = is_home.astype("Int8")
    df["is_home"] = is_home
    df["venue"] = np.where(is_home == 1, "Home", "Away")

    # Validate consistency where comparable (non-fatal)
    bad_home = (is_home == 1) & (df["team_id"] != df["home_id"])
    bad_away = (is_home == 0) & (df["team_id"] != df["away_id"])
    n_bad = int(bad_home.sum() + bad_away.sum())
    if n_bad:
        msg = f"{season}: {n_bad} rows where provided is_home disagrees with team_id vs home/away ids."
        if strict:
            raise AssertionError(msg)
        logging.warning(msg + " Keeping given home_id/away_id; trusting provided flags for venue.")

    # Ensure opponent_id present
    if "opponent_id" not in df.columns or _is_blank_series(df["opponent_id"]).any():
        mask = _is_blank_series(df.get("opponent_id", pd.Series(index=df.index, dtype=object)))
        df.loc[mask, "opponent_id"] = np.where(df.loc[mask, "is_home"] == 1,
                                               df.loc[mask, "away_id"], df.loc[mask, "home_id"]).astype(str)
    return df

    if "is_home" in df.columns:
        provided = pd.to_numeric(df["is_home"], errors="coerce")
    elif "is_away" in df.columns:
        provided = (pd.to_numeric(df["is_away"], errors="coerce") == 0).astype(float)
    elif "venue" in df.columns:
        provided = df["venue"].astype(str).str.lower().map({"home":1.0,"away":0.0})
    else:
        provided = pd.Series(np.nan, index=df.index, dtype=float)

    is_home = pd.Series(inferred_is_home, dtype="Float64").where(~pd.isna(inferred_is_home), provided)
    n_amb = int(is_home.isna().sum())
    if n_amb:
        msg = f"{season}: {n_amb} rows ambiguous is_home (ids/flags missing)."
        if strict:
            raise AssertionError(msg)
        logging.warning(msg + " Setting venue='Home' by default for ambiguous rows.")
        is_home = is_home.fillna(1.0)

    is_home = is_home.astype("Int8")
    df["is_home"] = is_home
    df["venue"] = np.where(is_home == 1, "Home", "Away")

    bad_home = (is_home == 1) & (df["team_id"] != df["home_id"])
    bad_away = (is_home == 0) & (df["team_id"] != df["away_id"])
    n_bad = int(bad_home.sum() + bad_away.sum())
    if n_bad:
        msg = f"{season}: {n_bad} rows where is_home disagrees with team_id vs home/away ids."
        if strict:
            raise AssertionError(msg)
        logging.warning(msg + " Keeping given home_id/away_id; only fixing is_home/venue.")

    if "opponent_id" not in df.columns or _is_blank_series(df["opponent_id"]).any():
        mask = _is_blank_series(df.get("opponent_id", pd.Series(index=df.index, dtype=object)))
        df.loc[mask, "opponent_id"] = np.where(df.loc[mask, "is_home"] == 1,
                                               df.loc[mask, "away_id"], df.loc[mask, "home_id"]).astype(str)
    return df

def _load_all(fixtures_root: Path, seasons: List[str], strict_ids: bool) -> pd.DataFrame:
    frames = []
    for s in seasons:
        fp = fixtures_root / s / "fixture_calendar.csv"
        if not fp.is_file():
            logging.warning("%s • missing fixture_calendar.csv – skipped", s)
            continue
        df = pd.read_csv(fp, parse_dates=["date_played","date_sched"])

        _require_base(df, s)
        df = _coerce_poss(df)
        df = _validate_or_infer_from_ids(df, s, strict=strict_ids)

        for c in ["team_id","opponent_id","home_id","away_id","fpl_id","fbref_id"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.lower()

        dk = pd.to_datetime(df["date_played"], errors="coerce")
        ds = pd.to_datetime(df["date_sched"], errors="coerce")
        df["date_key"] = dk.fillna(ds)

        frames.append(df.assign(season=s))

    if not frames:
        raise FileNotFoundError("No seasons could be loaded from fixtures_root")

    all_df = pd.concat(frames, ignore_index=True)
    # quick sanity: mismatched namespaces
    try:
        t = all_df.assign(tid=all_df["team_id"].astype(str),
                          hid=all_df["home_id"].astype(str),
                          aid=all_df["away_id"].astype(str))
        ok = ((t["tid"]==t["hid"]) | (t["tid"]==t["aid"]))
        bad = int((~ok).sum()); frac = 100.0*bad/max(1,len(t))
        if bad:
            logging.debug("Identity sanity: %d rows (%.1f%%) where team_id != home_id/away_id.", bad, frac)
    except Exception:
        pass

    return all_df.sort_values(["season","date_key","fpl_id","home_id","away_id"]).reset_index(drop=True)

# ───────────────────────── rolling (classic + EWMA) ─────────────────────────

def rolling_past_only(team_df: pd.DataFrame, val_col: str, window: int, tau: float,
                      prior_val: Optional[float], prior_matches: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v   = pd.to_numeric(team_df[val_col], errors="coerce").to_numpy()
    ven = team_df["venue"].astype(str).to_numpy()
    if np.isnan(v).any():
        v = np.where(np.isnan(v), 50.0 if val_col == "poss" else 0.0, v)
    n = len(v)
    csum = np.r_[0.0, np.cumsum(v)]
    roll  = np.empty(n, dtype=float); roll_h= np.empty(n, dtype=float); roll_a= np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - window)
        cnt = i - lo
        base = 0.0 if cnt == 0 else (csum[i] - csum[lo]) / max(1, cnt)
        if prior_val is not None and prior_matches > 0 and i < prior_matches:
            w = 1.0 - (i / max(1, prior_matches))
            base = (1 - w) * base + w * prior_val
        mask = ven[lo:i] == "Home"; rec  = v[lo:i]
        nh = int(mask.sum()); na = int((~mask).sum())
        lam_h = nh / (nh + tau) if (nh + tau) > 0 else 0.0
        lam_a = na / (na + tau) if (na + tau) > 0 else 0.0
        mean_h = rec[mask].mean() if nh else base
        mean_a = rec[~mask].mean() if na else base
        roll[i]   = base
        roll_h[i] = lam_h * mean_h + (1 - lam_h) * base
        roll_a[i] = lam_a * mean_a + (1 - lam_a) * base
    return roll, roll_h, roll_a

def ewma_past_only(team_df: pd.DataFrame, val_col: str, halflife: float, tau: float,
                   prior_val: Optional[float], prior_matches: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v   = pd.to_numeric(team_df[val_col], errors="coerce").to_numpy()
    ven = team_df["venue"].astype(str).to_numpy()
    n   = len(v)
    v = np.where(np.isnan(v), 50.0 if val_col == "poss" else 0.0, v)
    alpha = 1 - 2 ** (-1.0 / max(1e-9, halflife))
    roll  = np.empty(n, dtype=float); roll[:]  = np.nan
    rollH = np.empty(n, dtype=float); rollH[:] = np.nan
    rollA = np.empty(n, dtype=float); rollA[:] = np.nan
    m_all = prior_val if prior_val is not None else (50.0 if val_col == "poss" else 0.0)
    m_h   = m_all; m_a = m_all
    cnt_all = 0; cnt_h = 0; cnt_a = 0
    for i in range(n):
        lam_h = cnt_h / (cnt_h + tau) if (cnt_h + tau) > 0 else 0.0
        lam_a = cnt_a / (cnt_a + tau) if (cnt_a + tau) > 0 else 0.0
        roll[i]  = m_all
        rollH[i] = lam_h * m_h + (1 - lam_h) * m_all
        rollA[i] = lam_a * m_a + (1 - lam_a) * m_all
        x = v[i]
        m_all = (1 - alpha) * m_all + alpha * x; cnt_all += 1
        if ven[i] == "Home":
            m_h = (1 - alpha) * m_h + alpha * x; cnt_h += 1
        else:
            m_a = (1 - alpha) * m_a + alpha * x; cnt_a += 1
        if prior_val is not None and cnt_all <= prior_matches:
            w = 1.0 - (cnt_all / max(1, prior_matches))
            m_all = (1 - w) * m_all + w * prior_val
            m_h   = (1 - w) * m_h   + w * prior_val
            m_a   = (1 - w) * m_a   + w * prior_val
    return roll, rollH, rollA

# ───────────────────────── FDR difficulty (buckets 2..5) ─────────────────────────

def _bucket_fixed_z(series: pd.Series) -> pd.Series:
    # map standardized to 2..5 using quartiles: [-0.6745, 0, 0.6745]
    bins = np.array([-0.67448975, 0.0, 0.67448975], dtype=float)
    return (np.digitize(series.to_numpy(), bins, right=True) + 2).astype(int)

def _percentile_bucket_by_season(df: pd.DataFrame, col: str) -> pd.Series:
    pct = df.groupby("season")[col].rank(pct=True)
    return (np.ceil(pct * 4).clip(1, 4).astype(int) + 1)

def _parse_is_home_flags(_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, bool]:
    if "is_home" not in _df.columns:
        return pd.Series(False, index=_df.index), pd.Series(False, index=_df.index), False
    raw = _df["is_home"]
    if pd.api.types.is_numeric_dtype(raw):
        h = pd.to_numeric(raw, errors="coerce")
        home_mask = (h == 1)
        away_mask = (h == 0)
    else:
        s = raw.astype(str).str.strip().str.lower()
        home_mask = s.isin({"1","true","t","yes","y","home","h"})
        away_mask = s.isin({"0","false","f","no","n","away","a"})
    usable = (home_mask.any() and away_mask.any())
    return home_mask.fillna(False), away_mask.fillna(False), bool(usable)

def _compute_fixture_fdr_difficulty(
    df: pd.DataFrame,
    season: str,
    w_att: float = 0.5,
    w_def: float = 0.5,
    bucket_mode: str = "hybrid",
    hybrid_cutoff: int = 6
) -> pd.DataFrame:
    need = [
        "season","fpl_id","team_id","home_id","away_id","date_played",
        "att_xg_home_roll_z","att_xg_away_roll_z",
        "def_xga_home_roll_z","def_xga_away_roll_z",
    ]
    missing = set(need) - set(df.columns)
    if missing:
        raise KeyError(f"{season}: missing columns for FDR difficulty: {missing}")

    # prefer flags; fallback to id equality
    home_mask_flag, away_mask_flag, flags_usable = _parse_is_home_flags(df)
    if flags_usable:
        is_home_mask = home_mask_flag
        is_away_mask = away_mask_flag
        logging.debug("%s: side split via flags (home=%d, away=%d).", season, int(is_home_mask.sum()), int(is_away_mask.sum()))
    else:
        tid  = df["team_id"].astype(str); hid  = df["home_id"].astype(str); aid  = df["away_id"].astype(str)
        is_home_mask = (tid == hid)
        is_away_mask = (tid == aid)
        logging.debug("%s: side split via id equality (home=%d, away=%d).", season, int(is_home_mask.sum()), int(is_away_mask.sum()))

    home_rows = df[is_home_mask][[
        "season","fpl_id","home_id","away_id","date_played",
        "att_xg_home_roll_z","def_xga_home_roll_z"
    ]].rename(columns={
        "att_xg_home_roll_z":"home_att_xg_home_z",
        "def_xga_home_roll_z":"home_def_xga_home_z",
    })

    away_rows = df[is_away_mask][[
        "season","fpl_id","home_id","away_id","date_played",
        "att_xg_away_roll_z","def_xga_away_roll_z"
    ]].rename(columns={
        "att_xg_away_roll_z":"away_att_xg_away_z",
        "def_xga_away_roll_z":"away_def_xga_away_z",
    })

    fixtures = pd.merge(
        home_rows, away_rows,
        on=["season","fpl_id","home_id","away_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_home","_away")
    )

    if fixtures.empty:
        n_home = int(len(home_rows)); n_away = int(len(away_rows))
        logging.warning("%s: 0 fixtures formed (home_rows=%d, away_rows=%d). "
                        "Likely namespace mismatch (team_id vs home/away) or only one side present.",
                        season, n_home, n_away)
        return pd.DataFrame(columns=[
            "season","fpl_id","home_id","away_id","date_played",
            "fdr_att_home","fdr_def_home","fdr_home",
            "fdr_att_away","fdr_def_away","fdr_away"
        ])

    # bring gw_orig (for hybrid) if present
    if "gw_orig" in df.columns:
        gw_h = df.loc[is_home_mask, ["fpl_id","gw_orig"]].rename(columns={"gw_orig":"gw_home"})
        gw_a = df.loc[is_away_mask, ["fpl_id","gw_orig"]].rename(columns={"gw_orig":"gw_away"})
        fixtures = fixtures.merge(gw_h, on="fpl_id", how="left").merge(gw_a, on="fpl_id", how="left")
        fixtures["gw_orig"] = fixtures["gw_home"].combine_first(fixtures["gw_away"])
        fixtures.drop(columns=["gw_home","gw_away"], inplace=True)
    else:
        fixtures["gw_orig"] = np.nan

    # choose a single date_played
    fixtures["date_played"] = fixtures["date_played_home"].combine_first(fixtures["date_played_away"])
    fixtures.drop(columns=["date_played_home","date_played_away"], inplace=True)

    # Difficulty components (5 = harder; buckets later converted 2..5)
    home_att_diff = - fixtures["away_def_xga_away_z"]
    home_def_diff = + fixtures["away_att_xg_away_z"]
    fixtures["fdr_att_home_score"] = home_att_diff
    fixtures["fdr_def_home_score"] = home_def_diff
    fixtures["fdr_home_score"]     = w_att * home_att_diff + w_def * home_def_diff

    away_att_diff = - fixtures["home_def_xga_home_z"]
    away_def_diff = + fixtures["home_att_xg_home_z"]
    fixtures["fdr_att_away_score"] = away_att_diff
    fixtures["fdr_def_away_score"] = away_def_diff
    fixtures["fdr_away_score"]     = w_att * away_att_diff + w_def * away_def_diff

    # Bucketing
    cols = ["fdr_att_home_score","fdr_def_home_score","fdr_home_score",
            "fdr_att_away_score","fdr_def_away_score","fdr_away_score"]

    if bucket_mode == "global":
        for c in cols:
            fixtures[c.replace("_score","")] = _bucket_fixed_z(fixtures[c])
    elif bucket_mode == "season":
        for c in cols:
            fixtures[c.replace("_score","")] = _percentile_bucket_by_season(fixtures, c)
    else:  # hybrid
        use_z = fixtures["gw_orig"].fillna(99).astype(float) < float(hybrid_cutoff)
        for c in cols:
            g = _percentile_bucket_by_season(fixtures, c)
            z = _bucket_fixed_z(fixtures[c])
            fixtures[c.replace("_score","")] = np.where(use_z, z, g)

    return fixtures[[
        "season","fpl_id","home_id","away_id","date_played",
        "fdr_att_home","fdr_def_home","fdr_home",
        "fdr_att_away","fdr_def_away","fdr_away"
    ]]

# ───────────────────────── per-season build ─────────────────────────

def build_team_form(
    season: str,
    fixtures_root: Path,
    out_version: str,
    out_dir: Path,
    decay_window: int,
    tau: float,
    force: bool,
    reuse_version: bool,
    all_seasons: List[str],
    prior_matches: int,
    cal_all: pd.DataFrame,
    use_ewma: bool,
    halflife: float,
    bucket_mode: str,
    hybrid_cutoff: int,
    w_att: float,
    w_def: float,
) -> bool:
    cal_fp = fixtures_root / season / "fixture_calendar.csv"
    dst_dir = out_dir / out_version / season
    dst_csv = dst_dir / OUTPUT_FILE
    meta_fp = dst_dir / META_FILE

    if dst_csv.exists() and not (force or reuse_version):
        logging.warning("%s exists; use --reuse-version for minor edits or --force to rebuild.", dst_csv)
        return False
    if not cal_fp.is_file():
        logging.warning("%s • fixture_calendar missing – skipped", season)
        return False

    cal = cal_all[cal_all["season"] == season].copy()
    if cal.empty:
        logging.warning("%s • no rows after load – skipped", season)
        return False

    for c in ["gf","ga","xg","xga","poss"]:
        cal[c] = pd.to_numeric(cal[c], errors="coerce")

    cols_att = list(METRICS["att"]["raw"])
    cols_def = list(METRICS["def"]["raw"])
    cols_pos = list(METRICS["pos"]["raw"])
    priors_att = build_team_priors(cal_all=cal_all, season=season, all_seasons=all_seasons, cols=cols_att, beta=0.7)
    priors_def = build_team_priors(cal_all=cal_all, season=season, all_seasons=all_seasons, cols=cols_def, beta=0.7)
    priors_pos = build_team_priors(cal_all=cal_all, season=season, all_seasons=all_seasons, cols=cols_pos, beta=0.7)

    rows = []
    for tid, grp in cal.groupby("team_id", sort=False):
        tmp = grp.sort_values(["date_key","fpl_id"]).copy()
        roll_fun = (lambda col, p: ewma_past_only(tmp, col, halflife, tau, p, prior_matches)) if use_ewma \
                   else (lambda col, p: rolling_past_only(tmp, col, decay_window, tau, p, prior_matches))
        pgf = float(priors_att.loc[tid, cols_att[0]]) if tid in priors_att.index else DEFAULT_MEANS["gf"]
        pxg = float(priors_att.loc[tid, cols_att[1]]) if tid in priors_att.index else DEFAULT_MEANS["xg"]
        gf_roll, gf_h, gf_a = roll_fun(cols_att[0], pgf)
        xg_roll, xg_h, xg_a = roll_fun(cols_att[1], pxg)
        tmp["att_gf_roll"] = gf_roll; tmp["att_xg_roll"] = xg_roll
        tmp["att_gf_home_roll"] = gf_h; tmp["att_gf_away_roll"] = gf_a
        tmp["att_xg_home_roll"] = xg_h; tmp["att_xg_away_roll"] = xg_a

        pga = float(priors_def.loc[tid, cols_def[0]]) if tid in priors_def.index else DEFAULT_MEANS["ga"]
        pxga= float(priors_def.loc[tid, cols_def[1]]) if tid in priors_def.index else DEFAULT_MEANS["xga"]
        ga_roll, ga_h, ga_a = roll_fun(cols_def[0], pga)
        xga_roll, xga_h, xga_a = roll_fun(cols_def[1], pxga)
        tmp["def_ga_roll"] = ga_roll; tmp["def_xga_roll"] = xga_roll
        tmp["def_ga_home_roll"] = ga_h; tmp["def_ga_away_roll"] = ga_a
        tmp["def_xga_home_roll"] = xga_h; tmp["def_xga_away_roll"] = xga_a

        ppos = float(priors_pos.loc[tid, cols_pos[0]]) if tid in priors_pos.index else DEFAULT_MEANS["poss"]
        poss_roll, poss_h, poss_a = roll_fun(cols_pos[0], ppos)
        tmp["pos_poss_roll"] = poss_roll; tmp["pos_poss_home_roll"] = poss_h; tmp["pos_poss_away_roll"] = poss_a

        rows.append(tmp)

    df = pd.concat(rows, ignore_index=True)

    for c in ["att_xg_home_roll","att_xg_away_roll","def_xga_home_roll","def_xga_away_roll",
              "pos_poss_home_roll","pos_poss_away_roll"]:
        grp = df.groupby(["season","gw_orig"])[c]
        mean = grp.transform("mean"); std = grp.transform("std").replace(0, np.nan)
        df[c + "_z"] = ((df[c] - mean) / std).fillna(0.0)

    df = df.drop(columns=[c for c in LEGACY_FDR if c in df.columns], errors="ignore")

    fixtures_fdr = _compute_fixture_fdr_difficulty(
        df, season, w_att=w_att, w_def=w_def, bucket_mode=bucket_mode, hybrid_cutoff=hybrid_cutoff
    )

    if fixtures_fdr.empty:
        for col in ["fdr_att_home","fdr_def_home","fdr_home",
            "fdr_att_away","fdr_def_away","fdr_away"]:
            if col not in df.columns:
                df[col] = np.nan
        logging.warning("[%s] FDR: formed 0 fixtures (one side missing?) — columns initialized to NaN.", season)
    else:
        fixtures_fdr = fixtures_fdr.drop(columns=["date_played"], errors="ignore")
        merge_keys = ["season","fpl_id","home_id","away_id"]
        gp = fixtures_fdr.groupby(["season","fpl_id"]).size()
        if (gp.max() == 1):
            merge_keys = ["season","fpl_id"]
            logging.debug("[%s] Using relaxed merge keys %s (fixtures_fdr unique by season+fpl_id).", season, merge_keys)
        df = df.merge(fixtures_fdr, on=merge_keys, how="left", validate="many_to_one")

    if ("date_played_x" in df.columns) or ("date_played_y" in df.columns):
        dx = pd.to_datetime(df.get("date_played_x"), errors="coerce")
        dy = pd.to_datetime(df.get("date_played_y"), errors="coerce")
        if "date_played" not in df.columns:
            df["date_played"] = dx.combine_first(dy)
        df = df.drop(columns=[c for c in ["date_played_x","date_played_y"] if c in df.columns])

    cov = df[["fdr_home","fdr_away"]].notna().mean().mean() * 100
    logging.info("[%s] FDR coverage: %.1f%%", season, cov)

    played = pd.to_datetime(df.get("date_played"), errors="coerce").notna()
    if played.any():
        try:
            from scipy.stats import spearmanr
            s_home = pd.to_numeric(df.loc[played, "fdr_home"], errors="coerce")
            t_home = pd.to_numeric(df.loc[played, "def_xga_home_roll_z"], errors="coerce") - \
                     pd.to_numeric(df.loc[played, "att_xg_home_roll_z"], errors="coerce")
            ok = s_home.notna() & t_home.notna()
            if ok.any():
                r, _ = spearmanr(s_home[ok], t_home[ok], nan_policy="omit")
                logging.info("[%s] Quick check (home side): Spearman(FDR, proxy target) = %.3f", season, r)
        except Exception as e:
            logging.debug("Validation skipped: %s", e)

    if "opponent_id" not in df.columns:
        df["opponent_id"] = np.where(df["team_id"] == df["home_id"], df["away_id"], df["home_id"]).astype(str)
    else:
        opp_blank = _is_blank_series(df["opponent_id"])
        df.loc[opp_blank, "opponent_id"] = np.where(df.loc[opp_blank, "team_id"] == df.loc[opp_blank, "home_id"],
                                                    df.loc[opp_blank, "away_id"], df.loc[opp_blank, "home_id"]).astype(str)

    dst_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_csv, index=False)

    existing_meta = load_json(meta_fp)
    build_no = int(existing_meta.get("build_no", 0)) + 1 if (reuse_version and meta_fp.exists()) else 1

    features_list = sorted(
        [c for c in df.columns if c.endswith(("_roll","_roll_z"))] +
        ["fdr_att_home","fdr_def_home","fdr_home","fdr_att_away","fdr_def_away","fdr_away"]
    )
    meta = {
        "schema": SCHEMA_VERSION,
        "build_ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "version": out_version,
        "build_no": build_no,
        "window_matches": decay_window,
        "tau": tau,
        "prior_matches": prior_matches,
        "ewma": use_ewma,
        "halflife": halflife if use_ewma else None,
        "fdr": {"type": "difficulty_v2_q2to5",
                "weights": {"att": w_att, "def": w_def},
                "bucket_mode": bucket_mode, "hybrid_cutoff": hybrid_cutoff},
        "seasons_built": all_seasons,
        "hash": _meta_hash(df),
        "features": features_list,
        "data_fingerprint": {
            "rows": int(len(df)),
            "fixtures_unique": int(fixtures_fdr.shape[0]) if not fixtures_fdr.empty else 0,
            "last_fixture_date": str(pd.to_datetime(df.get("date_played")).max())
        },
        "future_season_ready": True
    }
    save_json(meta, meta_fp)
    logging.info("%s • %s (%d rows) written", season, OUTPUT_FILE, len(df))
    return True

# ───────────────────────── batch / CLI ─────────────────────────

def run_batch(
    seasons: List[str],
    fixtures_root: Path,
    out_version: str,
    out_dir: Path,
    decay_window: int,
    tau: float,
    force: bool,
    reuse_version: bool,
    prior_matches: int,
    strict_ids: bool,
    write_latest: bool,
    use_ewma: bool,
    halflife: float,
    bucket_mode: str,
    hybrid_cutoff: int,
    w_att: float,
    w_def: float,
):
    all_seasons = sorted(seasons)
    cal_all = _load_all(fixtures_root, all_seasons, strict_ids=strict_ids)

    for season in all_seasons:
        try:
            ok = build_team_form(
                season=season,
                fixtures_root=fixtures_root,
                out_version=out_version,
                out_dir=out_dir,
                decay_window=decay_window,
                tau=tau,
                force=force,
                reuse_version=reuse_version,
                all_seasons=all_seasons,
                prior_matches=prior_matches,
                cal_all=cal_all,
                use_ewma=use_ewma,
                halflife=halflife,
                bucket_mode=bucket_mode,
                hybrid_cutoff=hybrid_cutoff,
                w_att=w_att,
                w_def=w_def,
            )
            if ok and write_latest:
                _write_latest_pointer(out_dir, out_version)
                _copy_to_latest_dir(out_dir, out_version, season)
        except Exception:
            logging.exception("%s • unhandled error – skipped", season)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", help="e.g. 2025-2026; omit for batch")
    ap.add_argument("--fixtures-root", type=Path, default=Path("data/processed/fixtures"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/features"))

    # Versioning controls
    ap.add_argument("--version", default=None, help="Version folder (e.g., v9). Use with --auto-version to pick next.")
    ap.add_argument("--auto-version", action="store_true", help="Pick the next vN under out-dir automatically.")
    ap.add_argument("--reuse-version", action="store_true",
                    help="Overwrite in-place for minor/non-logic edits; increments build_no in meta.")
    ap.add_argument("--write-latest", action="store_true",
                    help="Update 'latest' pointer and copy outputs to features/latest/<SEASON>/")

    # Rolling params
    ap.add_argument("--window", type=int, default=5, help="rolling window (matches, past-only) for classic mode")
    ap.add_argument("--tau", type=float, default=2.0, help="venue shrinkage strength")
    ap.add_argument("--prior-matches", type=int, default=6, help="first K matches blend prior → 0")
    ap.add_argument("--ewma", action="store_true", help="use EWMA past-only rolling instead of hard window")
    ap.add_argument("--halflife", type=float, default=3.0, help="EWMA halflife in matches (if --ewma)")

    # FDR params
    ap.add_argument("--bucket-mode", choices=["hybrid","season","global"], default="hybrid",
                    help="hybrid: GW<cutoff uses global z bins; otherwise season percentiles")
    ap.add_argument("--hybrid-cutoff", type=int, default=6, help="GW threshold for hybrid mode")
    ap.add_argument("--w-att", type=float, default=0.5, help="weight for attack difficulty component (0..1)")
    ap.add_argument("--w-def", type=float, default=0.5, help="weight for defense difficulty component (0..1)")

    # Behavior
    ap.add_argument("--strict-ids", action="store_true", default=False,
                    help="Error on inconsistencies between ids and is_home (default: warn & continue)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=args.log_level.upper())

    seasons = [args.season] if args.season else sorted(d.name for d in args.fixtures_root.iterdir() if d.is_dir())
    if not seasons:
        logging.error("No season folders in %s", args.fixtures_root); return

    features_root = args.out_dir
    features_root.mkdir(parents=True, exist_ok=True)
    version = _resolve_version(features_root, args.version, args.auto_version)

    logging.info("Processing seasons: %s", ", ".join(seasons))
    logging.info("Writing to version dir: %s", version)

    run_batch(
        seasons=seasons,
        fixtures_root=args.fixtures_root,
        out_version=version,
        out_dir=args.out_dir,
        decay_window=args.window,
        tau=args.tau,
        force=args.force,
        reuse_version=args.reuse_version,
        prior_matches=args.prior_matches,
        strict_ids=args.strict_ids,
        write_latest=args.write_latest,
        use_ewma=args.ewma,
        halflife=args.halflife,
        bucket_mode=args.bucket_mode,
        hybrid_cutoff=args.hybrid_cutoff,
        w_att=args.w_att,
        w_def=args.w_def,
    )

if __name__ == "__main__":
    main()
