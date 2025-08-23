#!/usr/bin/env python3
r"""fixtures_meta_builder.py – Batch-capable builder for **fixture_calendar.csv**
───────────────────────────────────────────────────────────────────────────────
Adds **home** and **away** (three-letter short codes) to the final CSV,
derives hex IDs (`home_id`, `away_id`) **from team_id/opponent_id + is_home/is_away**,
and includes Fixture Difficulty Ratings (FDR) for both home and away teams.

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
is_promoted, is_relegated,
fdr_home, fdr_away
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
        return pd.Series(False, index=s.index)
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"1","true","t","yes","y"})

# ─────────────────── single-season builder ─────────────────────────────────

def build_fixture_calendar(
    season: str,
    fpl_csv: Path,
    fb_csv: Path,
    team_map_fp: Path,
    short_map_fp: Path,
    out_dir: Path,
    features_root: Path,
    features_version: str,
    teams_csv: Path | None = None,
    neutral_fdr: float = 3.0,
    force: bool = False,
) -> bool:
    dst_dir = out_dir / season
    out_csv = dst_dir / "fixture_calendar.csv"
    if out_csv.exists() and not force:
        logging.info("%s • already done – skip (use --force)", season)
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

    # enforce lowercase hex strings (keeps None if missing)
    for c in ("home_id", "away_id"):
        cal[c] = cal[c].astype("string").str.lower()

    # days since last match (by short code, which is stable per team/season)
    cal = cal.sort_values(["team", "date_played"]).copy()
    cal["days_since_last_game"] = (
        cal.groupby("team")["date_played"].diff().dt.days.fillna(0).astype(int)
    )

    # ── Inject FDR from team_form (expects fdr_home/fdr_away per team_id) ──
    tfp = features_root / features_version / season / "team_form.csv"
    if tfp.is_file():
        try:
            tf = pd.read_csv(tfp, usecols=["fpl_id", "team_id", "fdr_home", "fdr_away"]).copy()
            tf["team_id"] = tf["team_id"].astype(str).str.lower()

            # home-side FDR
            tf_home = tf[["fpl_id", "team_id", "fdr_home"]].rename(columns={"team_id": "home_id"})
            cal = cal.merge(tf_home, on=["fpl_id", "home_id"], how="left")

            # away-side FDR
            tf_away = tf[["fpl_id", "team_id", "fdr_away"]].rename(columns={"team_id": "away_id"})
            cal = cal.merge(tf_away, on=["fpl_id", "away_id"], how="left")
        except Exception:
            logging.exception("%s • team_form merge failed; using neutral FDR", season)
            if "fdr_home" not in cal.columns: cal["fdr_home"] = np.nan
            if "fdr_away" not in cal.columns: cal["fdr_away"] = np.nan
    else:
        logging.warning("%s • team_form.csv not found; skipping FDR injection", season)
        cal["fdr_home"] = np.nan
        cal["fdr_away"] = np.nan

    # Neutral fallback for any missing FDRs
    cal["fdr_home"] = pd.to_numeric(cal["fdr_home"], errors="coerce").fillna(neutral_fdr)
    cal["fdr_away"] = pd.to_numeric(cal["fdr_away"], errors="coerce").fillna(neutral_fdr)

    # ── select & order for output ──
    out = cal[[
        "fpl_id", "game_id", "gw_orig", "gw_played",
        "date_sched", "date_played", "days_since_last_game",
        "team", "team_id", "opponent_id",
        "home", "away", "home_id", "away_id",
        "status", "sched_missing", "venue",
        "gf", "ga", "xga", "xg", "result", "poss",
        "is_promoted", "is_relegated",
        "fdr_home", "fdr_away"
    ]].rename(columns={"game_id": "fbref_id"})

    # ── write ──
    dst_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    logging.info("%s • fixture_calendar.csv (%d rows)", season, len(out))

    if not missing.empty:
        missing.to_csv(dst_dir / "_manual_fbref_match.csv", index=False)
        # Also surface any rows where home/away_id could not be derived
        null_ids = out[out["home_id"].isna() | out["away_id"].isna()]
        if not null_ids.empty:
            null_ids.to_csv(dst_dir / "_missing_home_away_ids.csv", index=False)
            logging.warning("%s • %d rows lack home_id/away_id (see _missing_home_away_ids.csv)",
                            season, len(null_ids))
        logging.warning("%s • %d rows lack fbref_id (see _manual_fbref_match.csv)", season, len(missing))
    return True

# ───────────────────── batch driver ─────────────────────────────────────────

def run_batch(
    seasons: List[str],
    fpl_root: Path,
    fbref_league: Path,
    team_map: Path,
    short_map: Path,
    out_dir: Path,
    features_root: Path,
    features_version: str,
    neutral_fdr: float,
    force: bool
):
    for season in seasons:
        fpl_csv = fpl_root / season / "fixtures.csv"
        fb_csv = fbref_league / season / "team_match" / "schedule.csv"
        teams_csv = fpl_root / season / "teams.csv"
        try:
            build_fixture_calendar(
                season,
                fpl_csv,
                fb_csv,
                team_map,
                short_map,
                out_dir,
                features_root,
                features_version,
                teams_csv=teams_csv,
                neutral_fdr=neutral_fdr,
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
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/fixtures"))
    ap.add_argument("--features-root", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--features-version", default="v2")  # align with team_form_builder default
    ap.add_argument("--neutral-fdr", type=float, default=3.0, help="Fallback FDR when team_form is missing")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    seasons = [args.season] if args.season else sorted(d.name for d in args.fpl_root.iterdir() if d.is_dir())
    if not seasons:
        logging.error("No seasons found"); return
    run_batch(
        seasons,
        args.fpl_root,
        args.fbref_league_dir,
        args.team_map,
        args.short_map,
        args.out_dir,
        args.features_root,
        args.features_version,
        args.neutral_fdr,
        args.force,
    )

if __name__ == "__main__":
    main()
