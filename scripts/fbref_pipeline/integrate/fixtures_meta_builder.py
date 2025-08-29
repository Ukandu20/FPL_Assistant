#!/usr/bin/env python3
r"""fixtures_meta_builder.py – Batch-capable builder for **fixture_calendar.csv**
───────────────────────────────────────────────────────────────────────────────
Adds **home** and **away** (three-letter short codes) to the final CSV,
derives hex IDs (`home_id`, `away_id`) **from team_id/opponent_id + is_home/is_away**.

This script keeps fixtures **pure** (no FDR columns). If you want a
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
from pandas.api.types import is_bool_dtype, is_numeric_dtype
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
    """Convert various truthy encodings to boolean mask (handles numeric floats like 1.0/0.0)."""
    if s is None:
        return pd.Series(False, index=pd.RangeIndex(0))
    if is_bool_dtype(s):
        return s.fillna(False)
    if is_numeric_dtype(s):
        # 0, 0.0 -> False; any non-zero -> True
        return s.fillna(0).astype(float).ne(0.0)
    # string-like: accept common truthy tokens, including "1.0"
    tokens_true = {"1","1.0","true","t","yes","y"}
    return s.astype(str).str.strip().str.lower().isin(tokens_true)

def _venue_to_is_home_int8(venue: pd.Series) -> pd.Series:
    """Map FBref venue → is_home:Int8 (1=home, 0=away, <NA>=neutral/unknown)."""
    v = venue.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=venue.index, dtype="Int8")
    out[v.isin({"home","h"})] = 1
    out[v.isin({"away","a"})] = 0
    return out

def _naeq(a: pd.Series, b: pd.Series) -> pd.Series:
    """NA-safe equality: True if equal OR both NA."""
    return (a == b) | (a.isna() & b.isna())

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
    tf_dir = Path(features_root) / team_version
    tf_path = tf_dir / season / "team_form.csv"
    if not tf_path.exists():
        logging.warning("%s • team_form.csv not found at %s; skip FDR view",
                        season, tf_path)
        return

    tf = pd.read_csv(tf_path, low_memory=False)

    if "game_date" in tf.columns and "date_played" not in tf.columns:
        tf = tf.rename(columns={"game_date": "date_played"})

    for c in ("home_id", "away_id", "team_id"):
        if c in tf.columns:
            tf[c] = tf[c].astype("string").str.lower()
    for c in ("home_id", "away_id"):
        if c in calendar_df.columns:
            calendar_df[c] = calendar_df[c].astype("string").str.lower()

    need_A = {"date_played", "home_id", "away_id", "fdr_home", "fdr_away"}
    need_B = {"fpl_id", "team_id", "fdr_home", "fdr_away"}

    merged = None
    if need_A.issubset(set(tf.columns)):
        key = ["date_played", "home_id", "away_id"]
        subset = tf[key + ["fdr_home", "fdr_away"]].drop_duplicates(key)
        try:
            merged = calendar_df.merge(subset, on=key, how="left", validate="one_to_one")
            logging.info("%s • FDR view join (A: date+ids) OK; null FDR rows=%d",
                         season, int(merged["fdr_home"].isna().sum()))
        except Exception:
            logging.exception("%s • join (A) failed; falling back to (B) if possible", season)

    if merged is None and need_B.issubset(set(tf.columns)) and "fpl_id" in calendar_df.columns:
        tf_b = tf[["fpl_id", "team_id", "fdr_home", "fdr_away"]].copy()
        tf_b["team_id"] = tf_b["team_id"].astype("string").str.lower()
        left = calendar_df.merge(
            tf_b[["fpl_id", "team_id", "fdr_home"]].rename(columns={"team_id": "home_id"}),
            on=["fpl_id", "home_id"], how="left", validate="one_to_one"
        )
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
    match_tolerance_days: int = 100,
    force: bool = False,
) -> bool:
    dst_dir = out_dir / season
    out_csv = dst_dir / "fixture_calendar.csv"
    if out_csv.exists() and not force:
        logging.info("%s • already done – skip (use --force)", season)
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

    # -- FPL data (scheduled)
    fpl = pd.read_csv(fpl_csv, parse_dates=["kickoff_time"])
    fpl = fpl.rename(
        columns={"id": "fpl_id", "event": "gw_orig", "team_h": "home_id_fpl", "team_a": "away_id_fpl"}
    )
    fpl["_fpl_row"]   = np.arange(len(fpl), dtype=np.int64)
    fpl["status"]     = np.where(fpl.get("finished", False), "finished", "scheduled")
    fpl["date_sched"] = normalise_date(fpl["kickoff_time"])
    fpl["sched_missing"] = 1  # will be corrected after joining

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

    # -- FBref schedule (actual played)
    fb = pd.read_csv(fb_csv, parse_dates=["game_date"])
    fb["date_played"] = normalise_date(fb["game_date"])
    fb_match = fb[[
        "game_id", "team", "team_id", "opponent_id",
        "home", "away", "date_played",
        "venue", "ga", "gf", "xga", "xg", "result", "poss",
        "is_promoted", "is_relegated", "is_home", "is_away"
    ]].copy()

    # -- First pass: strict date match (home, away, date_sched == date_played)
    cal = fpl.merge(
        fb_match,
        left_on=["home", "away", "date_sched"],
        right_on=["home", "away", "date_played"],
        how="left",
        validate="one_to_many",  # one FPL row -> two FB rows (one per team)
    )
    strict_matched = cal["game_id"].notna().sum()
    logging.info("%s • strict matches: %d", season, strict_matched)

    # -- Second pass: nearest-date recovery for reschedules within tolerance
    missing_mask = cal["game_id"].isna()
    if missing_mask.any():
        miss = cal.loc[missing_mask, ["_fpl_row", "home", "away", "date_sched"]].drop_duplicates("_fpl_row")
        fb_dates = fb_match[["home", "away", "date_played"]].drop_duplicates()

        cand = miss.merge(fb_dates, on=["home", "away"], how="left")
        cand["absdiff"] = (cand["date_played"] - cand["date_sched"]).abs().dt.days
        cand = cand[cand["absdiff"] <= int(match_tolerance_days)]
        # Prefer postponements (date_played >= date_sched) before bring-forwards,
        # then choose smallest absolute delta, then earliest actual date.
        nearest = (
            cand.sort_values(["_fpl_row", "absdiff", "date_played"])
                .drop_duplicates("_fpl_row", keep="first")
        )

        # Expand to both team rows for that nearest date
        fb_rows = nearest[["_fpl_row", "home", "away", "date_played"]].merge(
            fb_match, on=["home", "away", "date_played"], how="left", validate="one_to_many"
        )

        # Bring back FPL columns for these rows
        fpl_base = fpl[["_fpl_row", "fpl_id", "gw_orig", "date_sched", "home", "away", "status"]].copy()
        repair = fpl_base.merge(fb_rows, on=["_fpl_row", "home", "away"], how="right")

        # shape repair like 'cal'
        common_cols = list(set(cal.columns).intersection(set(repair.columns)))
        repair = repair[common_cols].copy()
        repair["gw_played"] = repair.get("gw_orig", np.nan)

        # combine: keep strict matches, drop the single unmatched shell rows, add full repaired rows
        still_unresolved = cal.loc[missing_mask & ~cal["_fpl_row"].isin(nearest["_fpl_row"])]
        cal = pd.concat([cal.loc[~missing_mask], repair, still_unresolved], ignore_index=True)

        logging.info("%s • recovered via nearest-date: %d; unresolved: %d",
                     season, int(repair["game_id"].notna().sum()),
                     int(still_unresolved.shape[0]))

    # Ensure team/opponent ids are lowercase strings
    for c in ("team_id", "opponent_id"):
        if c in cal.columns:
            cal[c] = cal[c].astype(str).str.lower()

    # If team_id missing (rare), attempt resolve from short code
    if "team_id" not in cal.columns:
        cal["team_id"] = cal["team"].map(code2hex)
    else:
        cal["team_id"] = cal["team_id"].fillna(cal["team"].map(code2hex))


    # ── Derive home/away using IDs first, then venue, then legacy flags ──
    # Expected hex from FPL short codes
    cal["home_hex_expected"] = cal["home"].astype(str).map(code2hex).astype("string")
    cal["away_hex_expected"] = cal["away"].astype(str).map(code2hex).astype("string")
    # ID-based truth (most reliable): team row is home iff team_id == home_hex_expected
    id_cmp_valid = cal["team_id"].notna() & cal["home_hex_expected"].notna()
    is_home_by_ids = pd.Series(pd.NA, index=cal.index, dtype="Int8")
    is_home_by_ids[id_cmp_valid] = cal.loc[id_cmp_valid, "team_id"].eq(
        cal.loc[id_cmp_valid, "home_hex_expected"]
    ).astype("Int8")
    # Venue-based fallback
    is_home_by_venue = _venue_to_is_home_int8(cal["venue"])
    # Legacy flag fallback
    if "is_home" in cal.columns:
        is_home_flag = _to_bool_mask(cal["is_home"]).astype("Int8")
    elif "is_away" in cal.columns:
        is_home_flag = (~_to_bool_mask(cal["is_away"])).astype("Int8")
    else:
        is_home_flag = pd.Series(pd.NA, index=cal.index, dtype="Int8")
    # Coalesce: IDs → venue → flag
    cal["is_home"] = is_home_by_ids
    need_fill = cal["is_home"].isna()
    if need_fill.any():
        cal.loc[need_fill, "is_home"] = is_home_by_venue.loc[need_fill]
    need_fill = cal["is_home"].isna()
    if need_fill.any():
        cal.loc[need_fill, "is_home"] = is_home_flag.loc[need_fill]
    cal["is_home"] = cal["is_home"].astype("Int8")
    cal["is_away"] = (1 - cal["is_home"]).astype("Int8")
    # Build mask and assign ids
    hmask = (cal["is_home"] == 1)

    cal["home_id"] = np.where(hmask, cal["team_id"], cal["opponent_id"])
    cal["away_id"] = np.where(hmask, cal["opponent_id"], cal["team_id"])



    # enforce lowercase hex strings (keeps <NA> if missing)
    for c in ("home_id", "away_id"):
        cal[c] = cal[c].astype("string").str.lower()

    # days since last match (by FBref 'team' rows; NaNs will be ignored)
    cal = cal.sort_values(["team", "date_played"]).copy()
    cal["days_since_last_game"] = (
        cal.groupby("team")["date_played"].diff().dt.days.fillna(0).astype(int)
    )

    # After you've built/merged `cal` and before selecting `out`:
    if "gw_played" not in cal.columns:
        cal["gw_played"] = np.nan

    # Default: gw_played := gw_orig; keep any already-computed values
    cal["gw_played"] = cal["gw_played"].fillna(cal["gw_orig"])

    # Use nullable ints so NaNs stay NaN (if any remain)
    


    # final flags
    cal["gw_played"] = cal.get("gw_played", cal.get("gw_orig"))
    cal["sched_missing"] = cal["game_id"].isna().astype("Int8")

    # ── select & order for output (PURE schedule; no FDR here) ──
    out = cal[[
        "fpl_id", "game_id", "gw_orig", "gw_played",
        "date_sched", "date_played", "days_since_last_game",
        "team", "team_id", "opponent_id", "is_home",
        "home", "away", "home_id", "away_id",
        "status", "sched_missing", "venue",
        "gf", "ga", "xga", "xg", "result", "poss",
        "is_promoted", "is_relegated",
    ]].rename(columns={"game_id": "fbref_id"}).copy()

    # guarantee integer 0/1 in CSV
    out["is_home"] = out["is_home"].astype("Int8")
    out["gw_played"] = out["gw_played"].astype("Int8")
    out["sched_missing"] = out["sched_missing"].astype("Int8")

    # ── write base calendar ──
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_csv = dst_dir / "fixture_calendar.csv"
    out.to_csv(out_csv, index=False)
    logging.info("%s • fixture_calendar.csv (%d rows)", season, len(out))

    # Diagnostics
    missing = out[out["fbref_id"].isna()]
    if not missing.empty:
        missing.to_csv(dst_dir / "_manual_fbref_match.csv", index=False)
        logging.warning("%s • %d rows lack fbref_id (see _manual_fbref_match.csv)", season, len(missing))
    null_ids = out[out["home_id"].isna() | out["away_id"].isna()]
    if not null_ids.empty:
        null_ids.to_csv(dst_dir / "_missing_home_away_ids.csv", index=False)
        logging.warning("%s • %d rows lack home_id/away_id (see _missing_home_away_ids.csv)",
                        season, len(null_ids))
    # NA-safe alignment audit (only on rows with all IDs present)
    cal_ids_ok = cal[["team_id","opponent_id","home_id","away_id","is_home"]].dropna().copy()
    bad_align = cal_ids_ok[
        ((cal_ids_ok["is_home"]==1) & (~_naeq(cal_ids_ok["home_id"], cal_ids_ok["team_id"]) |
                                      ~_naeq(cal_ids_ok["away_id"], cal_ids_ok["opponent_id"]))) |
        ((cal_ids_ok["is_home"]==0) & (~_naeq(cal_ids_ok["home_id"], cal_ids_ok["opponent_id"]) |
                                      ~_naeq(cal_ids_ok["away_id"], cal_ids_ok["team_id"])))
    ]
    if not bad_align.empty:
        cols = ["home","away","team","venue","date_played","team_id","opponent_id","home_id","away_id","is_home"]
        (dst_dir / "_home_alignment_audit.csv").write_text("")
        cal.loc[bad_align.index, cols].to_csv(dst_dir / "_home_alignment_audit.csv", index=False)
        logging.error("%s • %d rows fail home/away ID alignment (see _home_alignment_audit.csv)", season, len(bad_align))




    # ── reschedule audit (only where we *did* match) ──
    audit = out.loc[out["fbref_id"].notna() & out["date_sched"].notna() & out["date_played"].notna()].copy()
    audit = audit[audit["date_sched"] != audit["date_played"]]
    if not audit.empty:
        audit["delta_days"] = (audit["date_played"] - audit["date_sched"]).dt.days.astype(int)
        audit["abs_delta_days"] = audit["delta_days"].abs()
        audit_cols = [
            "fpl_id", "fbref_id", "gw_orig", "gw_played",
            "home", "away", "team", "team_id", "opponent_id",
            "date_sched", "date_played", "delta_days", "abs_delta_days", "status"
        ]
        (dst_dir / "_reschedule_audit.csv").write_text("")  # ensure file exists even if to_csv appends header
        audit[audit_cols].to_csv(dst_dir / "_reschedule_audit.csv", index=False)
        logging.info("%s • reschedule audit: %d rows moved (see _reschedule_audit.csv)", season, len(audit))
    else:
        logging.info("%s • no reschedules detected", season)

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
    match_tolerance_days: int,
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
                match_tolerance_days=match_tolerance_days,
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
    ap.add_argument("--match-tolerance-days", type=int, default=21,
                    help="Max days between scheduled (FPL) and played (FBref) for nearest-date recovery.")
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
        match_tolerance_days=args.match_tolerance_days,
        force=args.force,
    )

if __name__ == "__main__":
    main()
