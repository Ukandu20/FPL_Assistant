#!/usr/bin/env python3
r"""team_form_builder.py – defensive form features  ▸ schema v1.2
──────────────────────────────────────────────────────────────────────────────
Writes one row per team-match to:
    data/processed/features/<version>/<SEASON>/team_form.csv
plus a side-car:
    …/team_form.meta.json      ← holds {"schema": "v1.2", ...}

Key logic (delta vs v1.1)
─────────────────────────
• GA/xGA source unchanged → fixture_calendar.csv.
• Early-season prior:
      - Established clubs → uses their own running mean (as before).
      - **Promoted clubs** → prior is the mean GA/xGA of teams
        that were relegated in the previous season (≈ league-average
        bottom-three defence).
• Meta-file is written but *not read* from fixtures (graceful if absent).

Promoted detection
──────────────────
    promoted = teams_curr  – teams_prev
    relegated = teams_prev – teams_curr
The relegated-club mean is computed from last season’s fixture_calendar
(if present).  If not found (e.g. first season in dataset) we fall back
to the current league mean.

Everything else (window=5, decay=0.8→0.2, τ=2.0, break flag, percentiles)
is unchanged.
"""
from __future__ import annotations
import argparse, json, logging, statistics, datetime as dt
from pathlib import Path
from typing import List, Tuple, Dict, Set

import numpy as np
import pandas as pd

# ───────────────────────────── constants ────────────────────────────────────
SCHEMA_VERSION = "v1.2"

# ────────────────────────── small helpers ───────────────────────────────────
def load_json(p: Path) -> dict:  return json.loads(p.read_text("utf-8"))
def save_json(obj: dict, p: Path) -> None:  p.write_text(json.dumps(obj, indent=2))

def ensure_ga_cols(df: pd.DataFrame) -> pd.DataFrame:
    alias = {"goals_conceded": "ga", "goals_against": "ga", "ga90": "ga", 
             "expected_goals_conceded": "xga", "xga90": "xga"}
    rename = {c: alias[c] for c in df.columns if c in alias}
    if rename:
        df = df.rename(columns=rename)
    missing = {"ga", "xga"} - set(df.columns)
    if missing:
        raise KeyError(f"missing {missing} in fixture_calendar")
    return df

# ───────────────────── rolling calculation logic ───────────────────────────
def rolling_ga_xga(team_df: pd.DataFrame, window: int,
                   decay: Tuple[float], tau: float,
                   promoted_prior: Tuple[float, float]|None) -> pd.DataFrame:
    """Compute rolling GA/xGA (+ home/away splits) for a single team."""
    out = team_df.copy().sort_values("date_played").reset_index(drop=True)
    ga, xga   = out["ga"].to_numpy(), out["xga"].to_numpy()
    venues    = out["venue"].to_numpy()          # 'Home' / 'Away'
    m = len(out)

    roll_ga, roll_xga    = np.empty(m), np.empty(m)
    roll_ga_h, roll_ga_a = np.empty(m), np.empty(m)

    # prior for promoted clubs (may be None)
    prior_ga_prom, prior_xga_prom = promoted_prior if promoted_prior else (None, None)

    for i in range(m):
        lo = max(0, i - window + 1)
        recent_ga  = ga [lo:i+1]
        recent_xga = xga[lo:i+1]

        ga_mean  = recent_ga.mean()
        xga_mean = recent_xga.mean()

        # --- early-season decay -------------------------------------------
        if i < len(decay):
            w = decay[i]
            if prior_ga_prom is not None:        # promoted club
                prior_ga  = prior_ga_prom
                prior_xga = prior_xga_prom
            else:                                # established club
                prior_ga  = statistics.mean(ga [:i+1]) if i else ga_mean
                prior_xga = statistics.mean(xga[:i+1]) if i else xga_mean
            ga_mean  = (1 - w) * ga_mean  + w * prior_ga
            xga_mean = (1 - w) * xga_mean + w * prior_xga

        roll_ga [i], roll_xga[i] = ga_mean, xga_mean

        # --- home / away splits with shrinkage ----------------------------
        recent_h = recent_ga[venues[lo:i+1] == "Home"]
        recent_a = recent_ga[venues[lo:i+1] == "Away"]
        nh, na   = len(recent_h), len(recent_a)
        lam_h, lam_a = nh / (nh + tau), na / (na + tau)
        roll_ga_h[i] = lam_h * (recent_h.mean() if nh else ga_mean) + (1 - lam_h) * ga_mean
        roll_ga_a[i] = lam_a * (recent_a.mean() if na else ga_mean) + (1 - lam_a) * ga_mean

    out["ga90_roll5"], out["xga90_roll5"]          = roll_ga, roll_xga
    out["ga90_home_5"], out["ga90_away_5"] = roll_ga_h, roll_ga_a
    return out

# ───────────────────────── break-week flag ──────────────────────────────────
def add_break_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["team_id", "date_played"]).copy()
    df["gap"] = df.groupby("team_id")["date_played"].diff().dt.days
    league_gap = df.groupby("gw_orig")["date_played"].min().diff().dt.days
    df["league_gap"] = df["gw_orig"].map(league_gap)
    df["is_break"] = np.where((df["gap"] >= 8) & (df["league_gap"] >= 13), 1, 0)
    return df.drop(columns=["gap", "league_gap"])

# ───────────────────── per-season build function ───────────────────────────
def build_team_form(season: str, fixtures_root: Path, out_version: str,
                    out_dir: Path, decay: Tuple[float], tau: float,
                    force: bool) -> bool:

    cal_csv = fixtures_root / season / "fixture_calendar.csv"
    dst_dir = out_dir / out_version / season
    dst_csv = dst_dir / "team_form.csv"
    meta_fp = dst_dir / "team_form.meta.json"

    if dst_csv.exists() and not force:
        prev_schema = (load_json(meta_fp)["schema"]
                       if meta_fp.exists() else "")
        if prev_schema == SCHEMA_VERSION:
            logging.info("%s • team_form (%s) already exists – skipped",
                         season, SCHEMA_VERSION)
            return False
        logging.warning("%s • overwriting file with schema %s → %s "
                        "(use --force to suppress message)",
                        season, prev_schema, SCHEMA_VERSION)

    if not cal_csv.is_file():
        logging.warning("%s • fixture_calendar.csv missing – skipped", season)
        return False

    # ------------- load fixture_calendar & validate ------------------------
    cal = pd.read_csv(cal_csv, parse_dates=["date_played"])
    cal = ensure_ga_cols(cal)


    required = {"team_id", "team", "venue", "gw_orig",
                "date_played", "ga", "xga"}
    if not required.issubset(cal.columns):
        raise KeyError(f"{season}: calendar missing {required - set(cal.columns)}")

    # ------------- promoted / relegated logic ------------------------------
    season_years = season.split("-")
    if len(season_years) != 2 or not season_years[0].isdigit():
        prev_cal = None
    else:
        prev_season = f"{int(season_years[0])-1}-{int(season_years[0])}"
        prev_cal_path = fixtures_root / prev_season / "fixture_calendar.csv"
        prev_cal = (pd.read_csv(prev_cal_path, parse_dates=["date_played"])
                    if prev_cal_path.is_file() else None)

    if prev_cal is not None:
        teams_prev  = set(prev_cal["team_id"].unique())
        teams_curr  = set(cal["team_id"].unique())
        relegated   = teams_prev - teams_curr
        promoted    = teams_curr - teams_prev

        # Mean GA/xGA of relegated clubs last season
        rel_stats = prev_cal[prev_cal["team_id"].isin(relegated)]
        prior_ga_releg = round(rel_stats["ga"].mean(), 3)
        prior_xga_releg = round(rel_stats["xga"].mean(), 3)
        promoted_lookup: Dict[str, Tuple[float, float]] = {
            hx: (prior_ga_releg, prior_xga_releg) for hx in promoted
        }
    else:
        promoted_lookup = {}

    # ------------- rolling calculations per team ---------------------------
    result_rows = []
    for team_id, team_df in cal.groupby("team_id"):
        prom_prior = promoted_lookup.get(team_id)
        result_rows.append(
            rolling_ga_xga(team_df, window=5, decay=decay,
                           tau=tau, promoted_prior=prom_prior)
        )

    form_df = pd.concat(result_rows, ignore_index=True)
    form_df = add_break_flag(form_df)

    # -------- league-wide percentile & z-score transforms ------------------
    for col in ("ga90_roll5", "xga90_roll5", "ga90_home_5", "ga90_away_5"):
        form_df[col + "_pct"] = form_df.groupby("gw_orig")[col].rank(pct=True)
        form_df[col + "_z"]   = (form_df[col] - form_df.groupby("gw_orig")[col].transform("mean")) / form_df.groupby("gw_orig")[col].transform("std")
        form_df["ga90_log"] = np.log1p(form_df.ga90_roll5)
        μ = form_df.groupby("gw_orig")["ga90_log"].transform("mean")
        σ = form_df.groupby("gw_orig")["ga90_log"].transform("std")
        form_df["ga90_log_z"] = (form_df.ga90_log - μ) / σ        # then maybe * -1


    # ------------------- write outputs & meta ------------------------------
    dst_dir.mkdir(parents=True, exist_ok=True)
    form_df.to_csv(dst_csv, index=False)

    meta = {
        "schema": SCHEMA_VERSION,
        "build_ts": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "promoted_prior": "relegated_mean" if promoted_lookup else "league_mean",
        "decay": decay,
        "tau": tau,
    }
    save_json(meta, meta_fp)
    logging.info("%s • team_form.csv (%d rows) written (%s)",
                 season, len(form_df), SCHEMA_VERSION)
    return True

# ─────────────────────── batch driver / CLI ────────────────────────────────
def run_batch(seasons: List[str], fixtures_root: Path,
              out_version: str, out_dir: Path,
              decay: Tuple[float], tau: float, force: bool):
    for season in seasons:
        try:
            build_team_form(season, fixtures_root, out_version,
                            out_dir, decay, tau, force)
        except Exception:
            logging.exception("%s • unhandled error – skipped", season)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", help="e.g. 2024-2025; omit for all")
    ap.add_argument("--fixtures-root", type=Path,
                    default=Path("data/processed/fixtures"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/processed/features"))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--decay",   default="0.8,0.6,0.4,0.2")
    ap.add_argument("--tau",     type=float, default=2.0)
    ap.add_argument("--force",   action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s: %(message)s")

    decay = tuple(float(x) for x in args.decay.split(","))

    seasons = ([args.season] if args.season else
               sorted(d.name for d in args.fixtures_root.iterdir()
                       if d.is_dir()))
    if not seasons:
        logging.error("No seasons found in %s", args.fixtures_root)
        return
    logging.info("Processing seasons: %s", ", ".join(seasons))

    run_batch(seasons, args.fixtures_root,
              args.version, args.out_dir,
              decay, args.tau, args.force)

if __name__ == "__main__":
    main()
