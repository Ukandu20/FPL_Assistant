#!/usr/bin/env python3
r"""team_form_builder.py  – schema v1.8 (adds possession), causal, venue-aware, symmetric FDR, versioned

Inputs (per season):
  data/processed/fixtures/<SEASON>/fixture_calendar.csv
  Required columns:
    fpl_id, fbref_id, gw_orig, gw_played, date_sched, date_played,
    days_since_last_game, team, team_id, home, away, home_id, away_id,
    status, sched_missing, venue, gf, ga, xga, xg, poss, result,
    is_promoted, is_relegated, fdr_home, fdr_away

Outputs (per season):
  data/processed/features/<version>/<SEASON>/team_form.csv
  data/processed/features/<version>/<SEASON>/team_form.meta.json

Design:
• Past-only rolling means (no current-row leakage).
• Multi-season priors: last ≤3 seasons (newer weighted more).
• Promoted teams (no history): fallback to league mean of previous seasons.
• Venue splits with shrinkage λ = n/(n+τ).
• Z-scores computed within (season, gw_orig); z=0 when std==0.
• FDR computed once per fixture (symmetric): consistent across both rows.
• Versioning & lineage: meta includes features list and a content hash.
"""
from __future__ import annotations
import argparse, json, logging, datetime as dt, hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from functools import reduce

import numpy as np
import pandas as pd

SCHEMA_VERSION = "v1.8"

# Raw metric columns expected in fixture_calendar.csv
METRICS = {
    "att": {"raw": ("gf", "xg")},     # team goals for, team xG (match totals)
    "def": {"raw": ("ga", "xga")},    # team goals against, team xGA (match totals)
    "pos": {"raw": ("poss",)},        # team possession (%) per match row
}

REQUIRED = {
    "fpl_id", "fbref_id", "team_id", "team", "venue", "gw_orig", "date_played",
    "home_id", "away_id",
    "fdr_home", "fdr_away",
    "gf","ga","xg","xga","poss","result",
}

OUTPUT_FILE = "team_form.csv"

# Legacy FDR columns to purge pre-merge (idempotence)
LEGACY_FDR = [
    "fdr_home","fdr_away",
    "fdr_att_home","fdr_def_home","fdr_att_away","fdr_def_away",
    "fdr_home_cont","fdr_away_cont",
    "fdr_att_home_cont","fdr_def_home_cont","fdr_att_away_cont","fdr_def_away_cont",
]

# ─────────────────────────── I/O helpers ───────────────────────────

def load_json(p: Path) -> dict:
    return json.loads(p.read_text("utf-8")) if p.is_file() else {}

def save_json(obj: dict, p: Path) -> None:
    p.write_text(json.dumps(obj, indent=2))

def _meta_hash(df: pd.DataFrame) -> str:
    """Stable short hash over a sample of output columns for lineage."""
    cols = sorted([c for c in df.columns if c.startswith(("att_","def_","fdr_","pos_"))])
    sample = df[["season","fpl_id","team_id"] + cols].head(500).to_json(orient="split", index=False)
    return hashlib.sha256(sample.encode("utf-8")).hexdigest()[:12]

# ─────────────── Multi-season priors (last ≤3 seasons) ─────────────

def _seasons_before(all_seasons: List[str], current: str, k: int = 3) -> List[str]:
    prior = [s for s in all_seasons if s < current]
    prior.sort()
    return prior[-k:]

def build_team_priors(
    cal_all: pd.DataFrame,
    season: str,
    all_seasons: List[str],
    cols: List[str],
    beta: float = 0.7
) -> pd.DataFrame:
    prev_seasons = _seasons_before(all_seasons, season, k=3)
    if not prev_seasons:
        base = cal_all[cal_all["season"] == season].groupby("team_id")[cols].mean()
        return base.fillna(base.mean())

    stacks = []
    for i, s in enumerate(prev_seasons):
        w = beta ** (len(prev_seasons) - 1 - i)  # newer seasons ↑ weight
        g = cal_all[cal_all["season"] == s].groupby("team_id")[cols].mean().mul(w)
        stacks.append(g)

    prior = reduce(lambda a, b: a.add(b, fill_value=0), stacks)

    cur_ids = cal_all.loc[cal_all["season"] == season, "team_id"].unique()
    prior = prior.reindex(cur_ids)

    league_mean_prev = (
        pd.concat([cal_all[cal_all["season"] == s] for s in prev_seasons])
        .groupby("team_id")[cols].mean().mean()
    )
    prior = prior.fillna(league_mean_prev.to_dict())
    return prior

# ───────── Rolling (past-only) with venue shrinkage + taper ─────────

def rolling_past_only(
    team_df: pd.DataFrame,
    val_col: str,
    window: int,
    tau: float,
    prior_val: Optional[float],
    prior_matches: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v   = team_df[val_col].astype(float).to_numpy()
    ven = team_df["venue"].astype(str).to_numpy()

    # guard: no NaNs (possession gaps default to neutral 50)
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

        # prior taper for first K matches
        if prior_val is not None and prior_matches > 0 and i < prior_matches:
            w = 1.0 - (i / prior_matches)  # w∈(1..~0]
            base = (1 - w) * base + w * prior_val

        mask = ven[lo:i] == "Home"
        rec  = v[lo:i]
        nh = int(mask.sum()); na = int((~mask).sum())
        lam_h = nh / (nh + tau); lam_a = na / (na + tau)
        mean_h = rec[mask].mean() if nh else base
        mean_a = rec[~mask].mean() if na else base

        roll[i]   = base
        roll_h[i] = lam_h * mean_h + (1 - lam_h) * base
        roll_a[i] = lam_a * mean_a + (1 - lam_a) * base

    return roll, roll_h, roll_a

# ────────────────────────── FDR (symmetric) ─────────────────────────

def _compute_fixture_fdr_symmetric(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Compute FDR per fixture (home & away) using both teams' venue-specific z's.
    Returns a frame keyed by (season, fpl_id, fbref_id) with fdr_* columns.
    """
    need = [
        "season","fpl_id","fbref_id","home_id","away_id","team_id",
        "att_xg_home_roll_z","att_xg_away_roll_z",
        "def_xga_home_roll_z","def_xga_away_roll_z",
    ]
    missing = set(need) - set(df.columns)
    if missing:
        raise KeyError(f"{season}: missing columns for symmetric FDR: {missing}")

    home_rows = df[df["team_id"] == df["home_id"]][[
        "season","fpl_id","fbref_id","home_id","away_id",
        "att_xg_home_roll_z","def_xga_home_roll_z"
    ]].rename(columns={
        "att_xg_home_roll_z":"home_att_xg_home_z",
        "def_xga_home_roll_z":"home_def_xga_home_z",
    })

    away_rows = df[df["team_id"] == df["away_id"]][[
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

    # Continuous difficulty components (venue-aware):
    fixtures["fdr_att_home_cont"] = fixtures["away_def_xga_away_z"]   # ease to attack at home
    fixtures["fdr_def_home_cont"] = fixtures["away_att_xg_away_z"]    # difficulty to defend at home
    fixtures["fdr_att_away_cont"] = fixtures["home_def_xga_home_z"]   # ease to attack away
    fixtures["fdr_def_away_cont"] = fixtures["home_att_xg_home_z"]    # difficulty to defend away

    fixtures["fdr_home_cont"] = (fixtures["fdr_att_home_cont"] + fixtures["fdr_def_home_cont"]) / 2.0
    fixtures["fdr_away_cont"] = (fixtures["fdr_att_away_cont"] + fixtures["fdr_def_away_cont"]) / 2.0

    def _bucket(s: pd.Series) -> pd.Series:
        pct = s.rank(pct=True)
        return np.ceil(pct * 5).astype(int).clip(1, 5)

    for col in ["fdr_att_home_cont","fdr_def_home_cont","fdr_home_cont",
                "fdr_att_away_cont","fdr_def_away_cont","fdr_away_cont"]:
        fixtures[col.replace("_cont","")] = (
            fixtures.groupby("season")[col].transform(_bucket)
        )

    keep = ["season","fpl_id","fbref_id",
            "fdr_att_home","fdr_def_home","fdr_home",
            "fdr_att_away","fdr_def_away","fdr_away",
            "fdr_att_home_cont","fdr_def_home_cont","fdr_home_cont",
            "fdr_att_away_cont","fdr_def_away_cont","fdr_away_cont"]
    return fixtures[keep]

# ───────────────────── helpers to load all seasons ─────────────────

def _coerce_poss(df: pd.DataFrame) -> pd.DataFrame:
    # Possession may be in 0–1 or 0–100; coerce to 0–100 float; fill gaps with neutral 50
    df["poss"] = pd.to_numeric(df["poss"], errors="coerce")
    if df["poss"].max(skipna=True) is not None and df["poss"].max(skipna=True) <= 1.5:
        df["poss"] = df["poss"] * 100.0
    df["poss"] = df["poss"].fillna(50.0)
    return df

def _load_all(fixtures_root: Path, seasons: List[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        fp = fixtures_root / s / "fixture_calendar.csv"
        if not fp.is_file():
            continue
        df = pd.read_csv(fp, parse_dates=["date_played"])
        missing = REQUIRED - set(df.columns)
        if missing:
            raise KeyError(f"{s}: fixture_calendar missing columns: {missing}")
        df = _coerce_poss(df)
        frames.append(df.assign(season=s))
    if not frames:
        raise FileNotFoundError("No seasons could be loaded from fixtures_root")
    all_df = pd.concat(frames, ignore_index=True)
    return all_df.sort_values(["season","date_played","fpl_id","team_id"]).reset_index(drop=True)

# ───────────────────── per-season builder ──────────────────────────

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
    cal_all: pd.DataFrame,            # (5) cached all seasons passed in
) -> bool:
    cal_fp = fixtures_root / season / "fixture_calendar.csv"
    dst_dir = out_dir / out_version / season
    dst_csv = dst_dir / OUTPUT_FILE
    meta_fp = dst_dir / "team_form.meta.json"

    if dst_csv.exists() and not force:
        logging.warning("%s exists; re-run with --force to ensure consistent priors across seasons", season)
        return False
    if not cal_fp.is_file():
        logging.warning("%s • fixture_calendar missing – skipped", season)
        return False

    cal = cal_all[cal_all["season"] == season].copy()
    cal = cal.sort_values(["date_played","fpl_id","team_id"]).reset_index(drop=True)

    # Priors for this season
    cols_att = list(METRICS["att"]["raw"])  # ["gf","xg"]
    cols_def = list(METRICS["def"]["raw"])  # ["ga","xga"]
    cols_pos = list(METRICS["pos"]["raw"])  # ["poss"]
    priors_att = build_team_priors(cal_all=cal_all, season=season, all_seasons=all_seasons, cols=cols_att, beta=0.7)
    priors_def = build_team_priors(cal_all=cal_all, season=season, all_seasons=all_seasons, cols=cols_def, beta=0.7)
    priors_pos = build_team_priors(cal_all=cal_all, season=season, all_seasons=all_seasons, cols=cols_pos, beta=0.7)

    # Rolling calculations per team
    rows = []
    for tid, grp in cal.groupby("team_id", sort=False):
        tmp = grp.copy()

        # ATT
        pgf = priors_att.loc[tid, cols_att[0]] if tid in priors_att.index else None
        pxg = priors_att.loc[tid, cols_att[1]] if tid in priors_att.index else None
        gf_roll, gf_h, gf_a = rolling_past_only(grp, cols_att[0], decay_window, tau, pgf, prior_matches)
        xg_roll, xg_h, xg_a = rolling_past_only(grp, cols_att[1], decay_window, tau, pxg, prior_matches)
        tmp["att_gf_roll"] = gf_roll
        tmp["att_xg_roll"] = xg_roll
        tmp["att_gf_home_roll"] = gf_h
        tmp["att_gf_away_roll"] = gf_a
        tmp["att_xg_home_roll"] = xg_h
        tmp["att_xg_away_roll"] = xg_a

        # DEF
        pga = priors_def.loc[tid, cols_def[0]] if tid in priors_def.index else None
        pxga= priors_def.loc[tid, cols_def[1]] if tid in priors_def.index else None
        ga_roll, ga_h, ga_a = rolling_past_only(grp, cols_def[0], decay_window, tau, pga, prior_matches)
        xga_roll, xga_h, xga_a = rolling_past_only(grp, cols_def[1], decay_window, tau, pxga, prior_matches)
        tmp["def_ga_roll"] = ga_roll
        tmp["def_xga_roll"] = xga_roll
        tmp["def_ga_home_roll"] = ga_h
        tmp["def_ga_away_roll"] = ga_a
        tmp["def_xga_home_roll"] = xga_h
        tmp["def_xga_away_roll"] = xga_a

        # POSSESSION
        ppos = priors_pos.loc[tid, cols_pos[0]] if tid in priors_pos.index else 50.0
        poss_roll, poss_h, poss_a = rolling_past_only(grp, cols_pos[0], decay_window, tau, ppos, prior_matches)
        tmp["pos_poss_roll"] = poss_roll
        tmp["pos_poss_home_roll"] = poss_h
        tmp["pos_poss_away_roll"] = poss_a

        rows.append(tmp)

    df = pd.concat(rows, ignore_index=True)

    # Z-scores with transform (no GroupBy leak)
    for c in [
        "att_xg_home_roll","att_xg_away_roll",
        "def_xga_home_roll","def_xga_away_roll",
        "pos_poss_home_roll","pos_poss_away_roll",
    ]:
        grp = df.groupby(["season","gw_orig"])[c]
        mean = grp.transform("mean")
        std  = grp.transform("std").replace(0, np.nan)
        df[c + "_z"] = ((df[c] - mean) / std).fillna(0.0)

    # (2) Idempotent: drop any pre-existing FDR columns to avoid _x/_y
    df = df.drop(columns=[c for c in LEGACY_FDR if c in df.columns], errors="ignore")

    # Symmetric, fixture-level FDR (unchanged; still xG/xGA based)
    fixtures_fdr = _compute_fixture_fdr_symmetric(df, season)

    # Merge fixture-level FDR back to team rows (symmetry guaranteed)
    df = df.merge(
        fixtures_fdr,
        on=["season","fpl_id","fbref_id"],
        how="left",
        validate="many_to_one"
    )

    # Symmetry assert (fail fast)
    chk = (df.groupby(["season","fpl_id"])[["fdr_home","fdr_away"]]
             .nunique()
             .rename(columns={"fdr_home":"n_home","fdr_away":"n_away"}))
    bad = chk[(chk["n_home"] != 1) | (chk["n_away"] != 1)]
    if len(bad):
        raise AssertionError(f"[{season}] Symmetry breach in {len(bad)} fixtures. Example:\n{bad.head(3)}")

    # Ensure numeric types clean for model ingestion
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
        "window_matches": decay_window,
        "tau": tau,
        "prior_matches": prior_matches,
        "fdr": {"type": "symmetric_v1", "source_forms": ["att_xg_*_roll_z","def_xga_*_roll_z"]},
        "seasons_built": all_seasons,
        "hash": _meta_hash(df),
        "features": features_list
    }
    save_json(meta, meta_fp)
    logging.info("%s • %s (%d rows) written", season, OUTPUT_FILE, len(df))
    return True

# ─────────────────────── batch driver / CLI ───────────────────────

def run_batch(
    seasons: List[str],
    fixtures_root: Path,
    out_version: str,
    out_dir: Path,
    decay_window: int,
    tau: float,
    force: bool,
    prior_matches: int,
):
    all_seasons = sorted(seasons)
    cal_all = _load_all(fixtures_root, all_seasons)

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
    ap.add_argument("--season", help="e.g. 2024-2025; omit for batch")
    ap.add_argument("--fixtures-root", type=Path, default=Path("data/processed/fixtures"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--window", type=int, default=5, help="rolling window (matches, past-only)")
    ap.add_argument("--tau", type=float, default=2.0, help="venue shrinkage strength")
    ap.add_argument("--prior-matches", type=int, default=6, help="first K matches blend prior → 0")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=args.log_level.upper())

    seasons = [args.season] if args.season else sorted(
        d.name for d in args.fixtures_root.iterdir() if d.is_dir()
    )
    if not seasons:
        logging.error("No season folders in %s", args.fixtures_root)
        return
    seasons = sorted(seasons)

    logging.info("Processing seasons: %s", ", ".join(seasons))
    run_batch(
        seasons=seasons,
        fixtures_root=args.fixtures_root,
        out_version=args.version,
        out_dir=args.out_dir,
        decay_window=args.window,
        tau=args.tau,
        force=args.force,
        prior_matches=args.prior_matches,
    )

if __name__ == "__main__":
    main()
