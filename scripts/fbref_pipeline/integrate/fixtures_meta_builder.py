#!/usr/bin/env python3
r"""fixtures_meta_builder.py – Batch-capable builder for **fixture_calendar.csv**
───────────────────────────────────────────────────────────────────────────────
Adds **home** and **away** (three-letter short codes) to the final CSV,
derives hex IDs (`home_id`, `away_id`) **from team_id/opponent_id + is_home/is_away**.

This script now keeps fixtures **pure** (no FDR columns). If you want a
denormalized calendar that includes Fixture Difficulty Ratings (FDR),
use the `--attach-fdr <version|latest>` flag to write a view:
features/<views-subdir>/<SEASON>/fixture_calendar_with_fdr__<version>.csv

Batch rules
• `--season` → single season; omit → loop over every folder in `--fpl-root`.
• `--force`  → overwrite existing outputs.

Output columns (order)
----------------------
fpl_id, fbref_id, gw_orig, gw_played,
date_sched, date_played,
days_since_last_game,
team, team_id, opponent_id,
home, away, home_id, away_id,
status, sched_missing,
venue, gf, ga, xga, xg, result, poss,
is_promoted, is_relegated
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np

# ───────────────────── helpers ──────────────────────────────────────────────

def load_json(p: Path) -> dict:
    return json.loads(p.read_text("utf-8"))

def canon(s: str) -> str:
    return " ".join(str(s).lower().split())

def build_maps(long2hex: Dict[str, str], long2code: Dict[str, str]):
    name2hex = {canon(k): str(v).lower() for k, v in long2hex.items()}
    name2code = {canon(k): str(v).upper() for k, v in long2code.items()}
    code2hex = {name2code[k]: v for k, v in name2hex.items() if k in name2code}
    return name2hex, name2code, code2hex

def normalise_date(series: pd.Series) -> pd.Series:
    if series.dt.tz is not None:
        series = series.dt.tz_convert(None)
    return series.dt.floor("D")

def _to_bool_mask(s: pd.Series) -> pd.Series:
    """Convert various truthy encodings to boolean mask."""
    if s is None:
        # Defensive: should not happen in this script; return empty False mask
        return pd.Series(False, index=pd.RangeIndex(0))
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"1","true","t","yes","y"})

def read_fixture_calendar(out_dir: Path, season: str) -> pd.DataFrame:
    fp = out_dir / season / "fixture_calendar.csv"
    return pd.read_csv(fp, parse_dates=["date_sched", "date_played"])

# ─────────────── FDR view materializer (optional, behind a flag) ────────────

def maybe_write_fdr_view(
    calendar_df: pd.DataFrame,
    season: str,
    features_root: Path,
    team_version: str,
    views_subdir: str = "views"
) -> None:
    """
    Materialize a denormalized view with FDRs:
      features/<views-subdir>/<SEASON>/fixture_calendar_with_fdr__<team_version>.csv

    Join strategy (robust):
      A) If team_form.csv has per-fixture FDR with both IDs:
         required columns: ['date_played','home_id','away_id','fdr_home','fdr_away']
         join on ['date_played','home_id','away_id'].
      B) Else if team_form.csv stores per-team FDR keyed by (fpl_id, team_id):
         required columns: ['fpl_id','team_id','fdr_home','fdr_away']
         do two one-to-one merges:
            home: on ['fpl_id','home_id'] → fdr_home
            away: on ['fpl_id','away_id'] → fdr_away
      If neither schema is found, the function logs an error and returns.
    """
    # Resolve team_form path; allow 'latest' to be treated as a folder
    tf_dir = Path(features_root) / team_version
    tf_path = tf_dir / season / "team_form.csv"
    if not tf_path.exists():
        logging.warning("%s • team_form.csv not found at %s; skip FDR view",
                        season, tf_path)
        return

    tf = pd.read_csv(tf_path, low_memory=False)

    # Normalize date column if present under 'game_date'
    if "game_date" in tf.columns and "date_played" not in tf.columns:
        tf = tf.rename(columns={"game_date": "date_played"})

    # Lowercase IDs for safe equality
    for c in ("home_id", "away_id", "team_id"):
        if c in tf.columns:
            tf[c] = tf[c].astype("string").str.lower()

    for c in ("home_id", "away_id"):
        if c in calendar_df.columns:
            calendar_df[c] = calendar_df[c].astype("string").str.lower()

    # Strategy A: full fixture-level FDR present
    need_A = {"date_played", "home_id", "away_id", "fdr_home", "fdr_away"}
    # Strategy B: per-team keyed by (fpl_id, team_id)
    need_B = {"fpl_id", "team_id", "fdr_home", "fdr_away"}

    merged = None
    if need_A.issubset(set(tf.columns)):
        key = ["date_played", "home_id", "away_id"]
        subset = tf[key + ["fdr_home", "fdr_away"]].drop_duplicates(key)
        try:
            merged = calendar_df.merge(
                subset, on=key, how="left", validate="one_to_one"
            )
            logging.info("%s • FDR view join (A: date+ids) OK; null FDR rows=%d",
                         season, int(merged["fdr_home"].isna().sum()))
        except Exception:
            logging.exception("%s • join (A) failed; falling back to (B) if possible", season)

    if merged is None and need_B.issubset(set(tf.columns)) and "fpl_id" in calendar_df.columns:
        tf_b = tf[["fpl_id", "team_id", "fdr_home", "fdr_away"]].copy()
        tf_b["team_id"] = tf_b["team_id"].astype("string").str.lower()
        # home side
        left = calendar_df.merge(
            tf_b[["fpl_id", "team_id", "fdr_home"]].rename(columns={"team_id": "home_id"}),
            on=["fpl_id", "home_id"], how="left", validate="one_to_one"
        )
        # away side
        merged = left.merge(
            tf_b[["fpl_id", "team_id", "fdr_away"]].rename(columns={"team_id": "away_id"}),
            on=["fpl_id", "away_id"], how="left", validate="one_to_one"
        )
        logging.info("%s • FDR view join (B: fpl_id+team_id) OK; null FDR rows=%d",
                     season, int(merged["fdr_home"].isna().sum()))
    if merged is None:
        logging.error(
            "%s • team_form.csv lacks required columns for join. "
            "Expected either %s or %s.",
            season, sorted(need_A), sorted(need_B)
        )
        return

    out_dir = Path(features_root) / views_subdir / season
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fixture_calendar_with_fdr__{team_version}.csv"
    merged.to_csv(out_path, index=False)
    logging.info("%s • wrote FDR view → %s", season, out_path)

# ─────────────────── single-season builder ─────────────────────────────────

def build_fixture_calendar(
    season: str,
    fpl_csv: Path,
    fb_csv: Path,
    team_map_fp: Path,
    short_map_fp: Path,
    out_dir: Path,
    *,
    attach_fdr: str | None,
    features_root: Path,
    views_subdir: str,
    teams_csv: Path | None = None,
    force: bool = False,
) -> bool:
    dst_dir = out_dir / season
    out_csv = dst_dir / "fixture_calendar.csv"
    if out_csv.exists() and not force:
        logging.info("%s • already done – skip (use --force)", season)
        # Optionally still write the view if requested (in case team_form updated)
        if attach_fdr:
            cal = read_fixture_calendar(out_dir, season)
            maybe_write_fdr_view(cal, season, features_root, attach_fdr, views_subdir)
        return False
    if not (fpl_csv.is_file() and fb_csv.is_file()):
        logging.warning("%s • missing fixture or schedule csv – skipped", season)
        return False

    # -- lookups
    name2hex, name2code, code2hex = build_maps(
        load_json(team_map_fp), load_json(short_map_fp)
    )

    # -- FPL data
    fpl = pd.read_csv(fpl_csv, parse_dates=["kickoff_time"])
    fpl = fpl.rename(
        columns={"id": "fpl_id", "event": "gw_orig", "team_h": "home_id_fpl", "team_a": "away_id_fpl"}
    )
    fpl["status"] = np.where(fpl.get("finished", False), "finished", "scheduled")
    fpl["date_played"] = normalise_date(fpl["kickoff_time"])
    fpl["date_sched"] = fpl["date_played"]
    fpl["sched_missing"] = 1

    if teams_csv is None:
        teams_csv = fpl_csv.with_name("teams.csv")
        if not teams_csv.exists():
            logging.warning("%s • teams.csv missing – skipped", season)
            return False
    teams_df = pd.read_csv(teams_csv, usecols=["id", "name"])
    id2name = dict(zip(teams_df.id, teams_df.name.map(canon)))

    # map FPL numeric ids -> normalized long names -> short codes for merge keys
    for side in ("home", "away"):
        fpl[f"{side}_long"] = fpl[f"{side}_id_fpl"].map(id2name)
        fpl[side] = fpl[f"{side}_long"].map(name2code)

    # -- FBref schedule (has team_id/opponent_id/is_home/is_away)
    fb = pd.read_csv(fb_csv, parse_dates=["game_date"])
    fb["date_played"] = normalise_date(fb["game_date"])
    fb_match = fb[[
        "game_id", "team", "team_id", "opponent_id",
        "home", "away", "date_played",
        "venue", "ga", "gf", "xga", "xg", "result", "poss",
        "is_promoted", "is_relegated", "is_home", "is_away"
    ]].copy()

    # -- Merge FPL + FBref  (two rows per fixture: one for each team)
    cal = fpl.merge(
        fb_match,
        on=["home", "away", "date_played"],
        how="left"
    )
    missing = cal[cal.game_id.isna()].copy()
    cal["gw_played"] = cal["gw_orig"]

    # Ensure team/opponent ids are lowercase strings
    for c in ("team_id", "opponent_id"):
        if c in cal.columns:
            cal[c] = cal[c].astype(str).str.lower()

    # If team_id missing (rare), attempt resolve from short code
    if "team_id" not in cal.columns:
        cal["team_id"] = cal["team"].map(code2hex)
    else:
        cal["team_id"] = cal["team_id"].fillna(cal["team"].map(code2hex))
    # opponent_id may be missing if FBref row absent; keep as-is to surface gaps

    # ── Derive home_id/away_id from team_id/opponent_id + is_home/is_away ──
    if "is_home" in cal.columns and cal["is_home"].notna().any():
        hmask = _to_bool_mask(cal["is_home"].fillna(False))
    elif "is_away" in cal.columns and cal["is_away"].notna().any():
        hmask = ~_to_bool_mask(cal["is_away"].fillna(False))
    else:
        # fallback: infer by short code equality if flags missing
        hmask = (cal["team"].astype(str).str.upper() == cal["home"].astype(str).str.upper())

    cal["home_id"] = np.where(hmask, cal["team_id"], cal["opponent_id"])
    cal["away_id"] = np.where(hmask, cal["opponent_id"], cal["team_id"])

    # enforce lowercase hex strings (keeps <NA> if missing)
    for c in ("home_id", "away_id"):
        cal[c] = cal[c].astype("string").str.lower()

    # days since last match (by short code, which is stable per team/season)
    cal = cal.sort_values(["team", "date_played"]).copy()
    cal["days_since_last_game"] = (
        cal.groupby("team")["date_played"].diff().dt.days.fillna(0).astype(int)
    )

    # ── select & order for output (PURE schedule; no FDR here) ──
    out = cal[[
        "fpl_id", "game_id", "gw_orig", "gw_played",
        "date_sched", "date_played", "days_since_last_game",
        "team", "team_id", "opponent_id",
        "home", "away", "home_id", "away_id",
        "status", "sched_missing", "venue",
        "gf", "ga", "xga", "xg", "result", "poss",
        "is_promoted", "is_relegated",
    ]].rename(columns={"game_id": "fbref_id"})

    # ── write base calendar ──
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_csv = dst_dir / "fixture_calendar.csv"
    out.to_csv(out_csv, index=False)
    logging.info("%s • fixture_calendar.csv (%d rows)", season, len(out))

    # Diagnostics
    if not missing.empty:
        missing.to_csv(dst_dir / "_manual_fbref_match.csv", index=False)
        logging.warning("%s • %d rows lack fbref_id (see _manual_fbref_match.csv)", season, len(missing))
    null_ids = out[out["home_id"].isna() | out["away_id"].isna()]
    if not null_ids.empty:
        null_ids.to_csv(dst_dir / "_missing_home_away_ids.csv", index=False)
        logging.warning("%s • %d rows lack home_id/away_id (see _missing_home_away_ids.csv)",
                        season, len(null_ids))

    # ── optionally write an FDR-attached view ──
    if attach_fdr:
        maybe_write_fdr_view(out, season, features_root, attach_fdr, views_subdir)

    return True

# ───────────────────── batch driver ─────────────────────────────────────────

def run_batch(
    seasons: List[str],
    fpl_root: Path,
    fbref_league: Path,
    team_map: Path,
    short_map: Path,
    out_dir: Path,
    *,
    attach_fdr: str | None,
    features_root: Path,
    views_subdir: str,
    force: bool
):
    for season in seasons:
        fpl_csv = fpl_root / season / "season" / "fixtures.csv"
        fb_csv = fbref_league / season / "team_match" / "schedule.csv"
        teams_csv = fpl_root / season / "teams.csv"
        try:
            build_fixture_calendar(
                season=season,
                fpl_csv=fpl_csv,
                fb_csv=fb_csv,
                team_map_fp=team_map,
                short_map_fp=short_map,
                out_dir=out_dir,
                attach_fdr=attach_fdr,
                features_root=features_root,
                views_subdir=views_subdir,
                teams_csv=teams_csv,
                force=force,
            )
        except Exception:
            logging.exception("%s • unhandled error", season)

# ───────────────────────────── CLI ─────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season")
    ap.add_argument("--fpl-root", type=Path, default=Path("data/raw/fpl"))
    ap.add_argument("--fbref-league-dir", type=Path, default=Path("data/processed/fbref/ENG-Premier League"))
    ap.add_argument("--team-map", type=Path, default=Path("data/processed/registry/_id_lookup_teams.json"))
    ap.add_argument("--short-map", type=Path, default=Path("data/config/teams.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/registry/fixtures"))
    ap.add_argument("--features-root", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--attach-fdr", default=None,
                    help="If set (e.g., 'latest' or 'v7'), also write "
                         "features/<views-subdir>/<SEASON>/fixture_calendar_with_fdr__<version>.csv")
    ap.add_argument("--views-subdir", default="views",
                    help="Subfolder under features/ to store materialized views (default: 'views').")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    if not args.fpl_root.exists():
        logging.error("FPL root not found: %s", args.fpl_root); return
    try:
        season_dirs = [d.name for d in args.fpl_root.iterdir() if d.is_dir()]
    except FileNotFoundError:
        season_dirs = []

    seasons = [args.season] if args.season else sorted(season_dirs)
    if not seasons:
        logging.error("No seasons found"); return

    run_batch(
        seasons=seasons,
        fpl_root=args.fpl_root,
        fbref_league=args.fbref_league_dir,
        team_map=args.team_map,
        short_map=args.short_map,
        out_dir=args.out_dir,
        attach_fdr=args.attach_fdr,
        features_root=args.features_root,
        views_subdir=args.views_subdir,
        force=args.force,
    )

if __name__ == "__main__":
    main()
