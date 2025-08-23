#!/usr/bin/env python3
r"""team_form_builder.py  – schema v2.4 (trusts precomputed home/away ids)

Inputs (per season):
  data/processed/fixtures/<SEASON>/fixture_calendar.csv
  Required (base) columns (values may contain NaNs for future fixtures):
    fpl_id, fbref_id, team_id, team, gw_orig, date_played, date_sched,
    gf, ga, xg, xga, poss, result, home_id, away_id
  Optional identity columns:
    opponent_id, is_home, is_away, venue

Behavior changes from v2.3:
• We DO NOT compute home_id/away_id here. We validate them and only
  infer is_home/venue/opponent_id if missing.
• If --strict-ids is set, any inconsistency between team_id vs home/away ids aborts.

Outputs (per season):
  data/processed/features/<version>/<SEASON>/team_form.csv
  data/processed/features/<version>/<SEASON>/team_form.meta.json

Key rules (validation only, no reassignment of ids):
• If team_id == home_id → is_home = 1, venue = "Home", opponent_id = away_id
• If team_id == away_id → is_home = 0, venue = "Away", opponent_id = home_id
• If both false but is_home/is_away/venue exists → use them to set venue + opponent_id
• If still ambiguous → warn (or error with --strict-ids)

Rolling stats:
• Past-only means (no leakage), venue-aware shrinkage, multi-season priors
• Symmetric FDR per fixture, merged back to both rows
"""
from __future__ import annotations
import argparse, json, logging, datetime as dt, hashlib, os, re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from functools import reduce

import numpy as np
import pandas as pd

SCHEMA_VERSION = "v2.4"

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

def build_team_priors(
    cal_all: pd.DataFrame,
    season: str,
    all_seasons: List[str],
    cols: List[str],
    beta: float = 0.7
) -> pd.DataFrame:
    prev_seasons = _seasons_before(all_seasons, season, k=3)
    cur_ids = cal_all.loc[cal_all["season"] == season, "team_id"].dropna().unique()

    if not prev_seasons:
        base = pd.DataFrame([DEFAULT_MEANS], index=[0])[cols]
        base = base.reindex(cur_ids).ffill().bfill()
        base.index = cur_ids
        return base

    stacks = []
    for i, s in enumerate(prev_seasons):
        w = beta ** (len(prev_seasons) - 1 - i)  # newer seasons ↑ weight
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

    prior = reduce(lambda a, b: a.add(b, fill_value=0.0), stacks)
    prior = prior.reindex(cur_ids)

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
    Trust given home_id/away_id. Do NOT rewrite them.
    Infer is_home/venue/opponent_id if missing, and validate consistency.
    """
    df = frame.copy()

    # Normalize id types
    for c in ["team_id","opponent_id","home_id","away_id","fpl_id","fbref_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()

    # Try to infer is_home from ids
    eq_home = df["team_id"] == df["home_id"]
    eq_away = df["team_id"] == df["away_id"]

    inferred_is_home = np.where(eq_home & ~eq_away, 1,
                          np.where(eq_away & ~eq_home, 0, np.nan))
    # Fallback to provided flags
    if "is_home" in df.columns:
        provided = pd.to_numeric(df["is_home"], errors="coerce")
    elif "is_away" in df.columns:
        provided = (pd.to_numeric(df["is_away"], errors="coerce") == 0).astype(float)
    elif "venue" in df.columns:
        provided = df["venue"].astype(str).str.lower().map({"home":1.0,"away":0.0})
    else:
        provided = pd.Series(np.nan, index=df.index, dtype=float)

    is_home = pd.Series(inferred_is_home, dtype="Float64").where(~pd.isna(inferred_is_home), provided)
    # If still NaN, log (or error in strict mode)
    n_amb = int(is_home.isna().sum())
    if n_amb:
        msg = f"{season}: {n_amb} rows ambiguous is_home (ids/flags missing)."
        if strict:
            raise AssertionError(msg)
        logging.warning(msg + " Setting venue='Home' by default for ambiguous rows.")
        is_home = is_home.fillna(1.0)  # bias to Home to keep deterministic; change if you prefer 0

    is_home = is_home.astype("Int8")
    df["is_home"] = is_home
    df["venue"] = np.where(is_home == 1, "Home", "Away")

    # Validate consistency where we CAN compare
    bad_home = (is_home == 1) & (~eq_home)
    bad_away = (is_home == 0) & (~eq_away)
    n_bad = int(bad_home.sum() + bad_away.sum())
    if n_bad:
        msg = f"{season}: {n_bad} rows where is_home disagrees with team_id vs home/away ids."
        if strict:
            raise AssertionError(msg)
        logging.warning(msg + " Keeping given home_id/away_id; only fixing is_home/venue.")

    # Ensure opponent_id present
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

        # TRUST provided home_id/away_id; infer only is_home/venue/opponent_id if needed
        df = _validate_or_infer_from_ids(df, s, strict=strict_ids)

        # id hygiene
        for c in ["team_id","opponent_id","home_id","away_id","fpl_id","fbref_id"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.lower()

        # date_key: prefer played date, else scheduled date
        dk = pd.to_datetime(df["date_played"], errors="coerce")
        ds = pd.to_datetime(df["date_sched"], errors="coerce")
        df["date_key"] = dk.fillna(ds)

        frames.append(df.assign(season=s))

    if not frames:
        raise FileNotFoundError("No seasons could be loaded from fixtures_root")

    all_df = pd.concat(frames, ignore_index=True)
    return all_df.sort_values(["season","date_key","fpl_id","home_id","away_id"]).reset_index(drop=True)

# ───────────────────────── rolling & FDR ─────────────────────────

def rolling_past_only(
    team_df: pd.DataFrame,
    val_col: str,
    window: int,
    tau: float,
    prior_val: Optional[float],
    prior_matches: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v   = pd.to_numeric(team_df[val_col], errors="coerce").to_numpy()
    ven = team_df["venue"].astype(str).to_numpy()

    if np.isnan(v).any():
        v = np.where(np.isnan(v), 50.0 if val_col == "poss" else 0.0, v)

    n = len(v)
    csum = np.r_[0.0, np.cumsum(v)]

    roll  = np.empty(n, dtype=float)
    roll_h= np.empty(n, dtype=float)
    roll_a= np.empty(n, dtype=float)

    for i in range(n):
        lo = max(0, i - window)   # EXCLUDE current row i
        cnt = i - lo
        base = 0.0 if cnt == 0 else (csum[i] - csum[lo]) / max(1, cnt)

        if prior_val is not None and prior_matches > 0 and i < prior_matches:
            w = 1.0 - (i / max(1, prior_matches))
            base = (1 - w) * base + w * prior_val

        mask = ven[lo:i] == "Home"
        rec  = v[lo:i]
        nh = int(mask.sum()); na = int((~mask).sum())
        lam_h = nh / (nh + tau) if (nh + tau) > 0 else 0.0
        lam_a = na / (na + tau) if (na + tau) > 0 else 0.0
        mean_h = rec[mask].mean() if nh else base
        mean_a = rec[~mask].mean() if na else base

        roll[i]   = base
        roll_h[i] = lam_h * mean_h + (1 - lam_h) * base
        roll_a[i] = lam_a * mean_a + (1 - lam_a) * base

    return roll, roll_h, roll_a

def _compute_fixture_fdr_symmetric(df: pd.DataFrame, season: str) -> pd.DataFrame:
    need = [
        "season","fpl_id","fbref_id","team_id","home_id","away_id",
        "att_xg_home_roll_z","att_xg_away_roll_z",
        "def_xga_home_roll_z","def_xga_away_roll_z",
    ]
    missing = set(need) - set(df.columns)
    if missing:
        raise KeyError(f"{season}: missing columns for symmetric FDR: {missing}")

    is_home_mask = (pd.to_numeric(df.get("is_home", 0), errors="coerce").fillna(0).astype(int) == 1) \
                   if "is_home" in df.columns else (df["team_id"].astype(str) == df["home_id"].astype(str))

    home_rows = df[is_home_mask][[
        "season","fpl_id","fbref_id","home_id","away_id",
        "att_xg_home_roll_z","def_xga_home_roll_z"
    ]].rename(columns={
        "att_xg_home_roll_z":"home_att_xg_home_z",
        "def_xga_home_roll_z":"home_def_xga_home_z",
    })

    away_rows = df[~is_home_mask][[
        "season","fpl_id","fbref_id","home_id","away_id",
        "att_xg_away_roll_z","def_xga_away_roll_z"
    ]].rename(columns={
        "att_xg_away_roll_z":"away_att_xg_away_z",
        "def_xga_away_roll_z":"away_def_xga_away_z",
    })

    fixtures = pd.merge(
        home_rows, away_rows,
        on=["season","fpl_id","fbref_id","home_id","away_id"],
        how="inner"
    )

    fixtures["fdr_att_home_cont"] = fixtures["away_def_xga_away_z"]
    fixtures["fdr_def_home_cont"] = fixtures["away_att_xg_away_z"]
    fixtures["fdr_att_away_cont"] = fixtures["home_def_xga_home_z"]
    fixtures["fdr_def_away_cont"] = fixtures["home_att_xg_home_z"]

    fixtures["fdr_home_cont"] = (fixtures["fdr_att_home_cont"] + fixtures["fdr_def_home_cont"]) / 2.0
    fixtures["fdr_away_cont"] = (fixtures["fdr_att_away_cont"] + fixtures["fdr_def_away_cont"]) / 2.0

    def _bucket(s: pd.Series) -> pd.Series:
        pct = s.rank(pct=True)
        return np.ceil(pct * 5).astype(int).clip(1, 5)

    for col in ["fdr_att_home_cont","fdr_def_home_cont","fdr_home_cont",
                "fdr_att_away_cont","fdr_def_away_cont","fdr_away_cont"]:
        fixtures[col.replace("_cont","")] = fixtures.groupby("season")[col].transform(_bucket)

    keep = ["season","fpl_id","fbref_id",
            "fdr_att_home","fdr_def_home","fdr_home",
            "fdr_att_away","fdr_def_away","fdr_away",
            "fdr_att_home_cont","fdr_def_home_cont","fdr_home_cont",
            "fdr_att_away_cont","fdr_def_away_cont","fdr_away_cont"]
    return fixtures[keep]

# ───────────────────────── per-season build ─────────────────────────

def build_team_form(
    season: str,
    fixtures_root: Path,
    out_version: str,
    out_dir: Path,
    decay_window: int,
    tau: float,
    force: bool,
    all_seasons: List[str],
    prior_matches: int,
    cal_all: pd.DataFrame,
) -> bool:
    cal_fp = fixtures_root / season / "fixture_calendar.csv"
    dst_dir = out_dir / out_version / season
    dst_csv = dst_dir / OUTPUT_FILE
    meta_fp = dst_dir / "team_form.meta.json"

    if dst_csv.exists() and not force:
        logging.warning("%s exists; re-run with --force for consistent priors", season)
        return False
    if not cal_fp.is_file():
        logging.warning("%s • fixture_calendar missing – skipped", season)
        return False

    cal = cal_all[cal_all["season"] == season].copy()
    if cal.empty:
        logging.warning("%s • no rows after load – skipped", season)
        return False

    # Numeric coercion
    for c in ["gf","ga","xg","xga","poss"]:
        cal[c] = pd.to_numeric(cal[c], errors="coerce")

    # Priors
    cols_att = list(METRICS["att"]["raw"])  # ["gf","xg"]
    cols_def = list(METRICS["def"]["raw"])  # ["ga","xga"]
    cols_pos = list(METRICS["pos"]["raw"])  # ["poss"]
    priors_att = build_team_priors(cal_all=cal_all, season=season, all_seasons=all_seasons, cols=cols_att, beta=0.7)
    priors_def = build_team_priors(cal_all=cal_all, season=season, all_seasons=all_seasons, cols=cols_def, beta=0.7)
    priors_pos = build_team_priors(cal_all=cal_all, season=season, all_seasons=all_seasons, cols=cols_pos, beta=0.7)

    # Rolling per team
    rows = []
    for tid, grp in cal.groupby("team_id", sort=False):
        tmp = grp.sort_values(["date_key","fpl_id"]).copy()

        # ATT
        pgf = float(priors_att.loc[tid, cols_att[0]]) if tid in priors_att.index else DEFAULT_MEANS["gf"]
        pxg = float(priors_att.loc[tid, cols_att[1]]) if tid in priors_att.index else DEFAULT_MEANS["xg"]
        gf_roll, gf_h, gf_a = rolling_past_only(tmp, cols_att[0], decay_window, tau, pgf, prior_matches)
        xg_roll, xg_h, xg_a = rolling_past_only(tmp, cols_att[1], decay_window, tau, pxg, prior_matches)
        tmp["att_gf_roll"] = gf_roll
        tmp["att_xg_roll"] = xg_roll
        tmp["att_gf_home_roll"] = gf_h
        tmp["att_gf_away_roll"] = gf_a
        tmp["att_xg_home_roll"] = xg_h
        tmp["att_xg_away_roll"] = xg_a

        # DEF
        pga = float(priors_def.loc[tid, cols_def[0]]) if tid in priors_def.index else DEFAULT_MEANS["ga"]
        pxga= float(priors_def.loc[tid, cols_def[1]]) if tid in priors_def.index else DEFAULT_MEANS["xga"]
        ga_roll, ga_h, ga_a = rolling_past_only(tmp, cols_def[0], decay_window, tau, pga, prior_matches)
        xga_roll, xga_h, xga_a = rolling_past_only(tmp, cols_def[1], decay_window, tau, pxga, prior_matches)
        tmp["def_ga_roll"] = ga_roll
        tmp["def_xga_roll"] = xga_roll
        tmp["def_ga_home_roll"] = ga_h
        tmp["def_ga_away_roll"] = ga_a
        tmp["def_xga_home_roll"] = xga_h
        tmp["def_xga_away_roll"] = xga_a

        # POS
        ppos = float(priors_pos.loc[tid, cols_pos[0]]) if tid in priors_pos.index else DEFAULT_MEANS["poss"]
        poss_roll, poss_h, poss_a = rolling_past_only(tmp, cols_pos[0], decay_window, tau, ppos, prior_matches)
        tmp["pos_poss_roll"] = poss_roll
        tmp["pos_poss_home_roll"] = poss_h
        tmp["pos_poss_away_roll"] = poss_a

        rows.append(tmp)

    df = pd.concat(rows, ignore_index=True)

    # Z-scores within (season, gw_orig)
    for c in [
        "att_xg_home_roll","att_xg_away_roll",
        "def_xga_home_roll","def_xga_away_roll",
        "pos_poss_home_roll","pos_poss_away_roll",
    ]:
        grp = df.groupby(["season","gw_orig"])[c]
        mean = grp.transform("mean")
        std  = grp.transform("std").replace(0, np.nan)
        df[c + "_z"] = ((df[c] - mean) / std).fillna(0.0)

    # Drop legacy FDR cols before merge (idempotence)
    df = df.drop(columns=[c for c in LEGACY_FDR if c in df.columns], errors="ignore")

    # Symmetric FDR per fixture
    fixtures_fdr = _compute_fixture_fdr_symmetric(df, season)

    # Merge back to team rows
    df = df.merge(
        fixtures_fdr,
        on=["season","fpl_id","fbref_id"],
        how="left",
        validate="many_to_one"
    )

    # Symmetry sanity (non-fatal)
    try:
        chk = (df.groupby(["season","fpl_id"])[["fdr_home","fdr_away"]]
                 .nunique()
                 .rename(columns={"fdr_home":"n_home","fdr_away":"n_away"}))
        bad = chk[(chk["n_home"] != 1) | (chk["n_away"] != 1)]
        if len(bad):
            raise AssertionError(f"[{season}] FDR symmetry issue in {len(bad)} fixtures. Example:\n{bad.head(3)}")
    except Exception as e:
        logging.warning("%s • FDR symmetry check warning: %s", season, e)

    # Numeric hygiene
    num_cols = [
        "att_gf_roll","att_xg_roll","att_gf_home_roll","att_gf_away_roll",
        "att_xg_home_roll","att_xg_away_roll",
        "def_ga_roll","def_xga_roll","def_ga_home_roll","def_ga_away_roll",
        "def_xga_home_roll","def_xga_away_roll",
        "pos_poss_roll","pos_poss_home_roll","pos_poss_away_roll",
        "att_xg_home_roll_z","att_xg_away_roll_z",
        "def_xga_home_roll_z","def_xga_away_roll_z",
        "pos_poss_home_roll_z","pos_poss_away_roll_z",
        "fdr_att_home_cont","fdr_def_home_cont","fdr_home_cont",
        "fdr_att_away_cont","fdr_def_away_cont","fdr_away_cont",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Persist/ensure opponent_id (without touching home/away ids)
    if "opponent_id" not in df.columns:
        df["opponent_id"] = np.where(df["team_id"] == df["home_id"], df["away_id"], df["home_id"]).astype(str)
    else:
        opp_blank = _is_blank_series(df["opponent_id"])
        df.loc[opp_blank, "opponent_id"] = np.where(df.loc[opp_blank, "team_id"] == df.loc[opp_blank, "home_id"],
                                                    df.loc[opp_blank, "away_id"], df.loc[opp_blank, "home_id"]).astype(str)

    # Write
    dst_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_csv, index=False)

    # Meta lineage
    features_list = sorted(
        [c for c in df.columns if c.endswith(("_roll","_roll_z"))] +
        ["fdr_att_home","fdr_def_home","fdr_home",
         "fdr_att_away","fdr_def_away","fdr_away",
         "fdr_att_home_cont","fdr_def_home_cont","fdr_home_cont",
         "fdr_att_away_cont","fdr_def_away_cont","fdr_away_cont"]
    )
    meta = {
        "schema": SCHEMA_VERSION,
        "build_ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "version": out_version,
        "window_matches": decay_window,
        "tau": tau,
        "prior_matches": prior_matches,
        "fdr": {"type": "symmetric_v1", "source_forms": ["att_xg_*_roll_z","def_xga_*_roll_z"]},
        "seasons_built": all_seasons,
        "hash": _meta_hash(df),
        "features": features_list,
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
    prior_matches: int,
    strict_ids: bool,
):
    all_seasons = sorted(seasons)
    cal_all = _load_all(fixtures_root, all_seasons, strict_ids=strict_ids)

    for season in all_seasons:
        try:
            build_team_form(
                season=season,
                fixtures_root=fixtures_root,
                out_version=out_version,
                out_dir=out_dir,
                decay_window=decay_window,
                tau=tau,
                force=force,
                all_seasons=all_seasons,
                prior_matches=prior_matches,
                cal_all=cal_all,
            )
        except Exception:
            logging.exception("%s • unhandled error – skipped", season)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", help="e.g. 2025-2026; omit for batch")
    ap.add_argument("--fixtures-root", type=Path, default=Path("data/processed/fixtures"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--version", default=None, help="Version folder (e.g., v9). Use with --auto-version to pick next.")
    ap.add_argument("--auto-version", action="store_true", help="Pick the next vN under out-dir automatically.")
    ap.add_argument("--write-latest", action="store_true", help="Update 'latest' pointer to the new version.")
    ap.add_argument("--window", type=int, default=5, help="rolling window (matches, past-only)")
    ap.add_argument("--tau", type=float, default=2.0, help="venue shrinkage strength")
    ap.add_argument("--prior-matches", type=int, default=6, help="first K matches blend prior → 0")
    ap.add_argument("--strict-ids", action="store_true", help="Error on inconsistencies between ids and is_home.")
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
        prior_matches=args.prior_matches,
        strict_ids=args.strict_ids,
    )

    if args.write_latest:
        _write_latest_pointer(features_root, version)

if __name__ == "__main__":
    main()
