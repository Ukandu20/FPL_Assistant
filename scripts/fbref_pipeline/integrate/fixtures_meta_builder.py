#!/usr/bin/env python3
r"""fixtures_meta_builder.py – Batch-capable builder for **fixture_calendar.csv**
───────────────────────────────────────────────────────────────────────────────
Normal mode:
  • Merge FPL fixtures with FBref schedule to build rich per-team rows.

Fallback mode (no FBref for the season):
  • Use fixture_metadata_per_team_resolved.csv to populate key fields:
    fpl_id, date_sched, team (3-letter code), team_id (from id_lookup),
    venue, was_home, home/away (+ their *_id), gw_orig, fdr_home/fdr_away.

Batch rules
• `--season` → single season; omit → loop over every folder in `--fpl-root`.
• `--force`  → overwrite existing outputs.
• `--create-empty` → if inputs are missing, write a header-only CSV.

Output columns (order)
----------------------
fpl_id, fbref_id, gw_orig, gw_played,
date_sched, date_played,
days_since_last_game,
status, sched_missing,
team, team_id, was_home, venue,
home, away, home_id, away_id,
result, gf, ga, xg, xga, poss,
is_promoted, is_relegated,
fdr_home, fdr_away
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

# ───────────────────── helpers ──────────────────────────────────────────────

# Canonical output columns & order for fixture_calendar.csv
OUT_COLS = [
    # Fixture identity / GW
    "fpl_id", "fbref_id", "gw_orig", "gw_played",
    # Scheduling & status
    "date_sched", "date_played", "days_since_last_game",
    "status", "sched_missing",
    # Row's team perspective
    "team", "team_id", "was_home", "venue",
    # Participants (codes + hex IDs)
    "home", "away", "home_id", "away_id",
    # Result & match context
    "result", "gf", "ga", "xg", "xga", "poss",
    # Team meta
    "is_promoted", "is_relegated",
    # Difficulty ratings
    "fdr_home", "fdr_away",
]

def write_empty_fixture_calendar(dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_csv = dst_dir / "fixture_calendar.csv"
    pd.DataFrame(columns=OUT_COLS).to_csv(out_csv, index=False)
    logging.info("%s • wrote EMPTY fixture_calendar.csv (no season data yet)", dst_dir.name)

def load_json(p: Path) -> dict:
    return json.loads(p.read_text("utf-8"))

def canon(s: str) -> str:
    return " ".join(s.lower().split())

def build_maps(long2hex: Dict[str, str], long2code: Dict[str, str]) -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
    """Return (name2hex, name2code, code2hex) based on long-name keyed maps."""
    name2hex = {canon(k): v.lower() for k, v in long2hex.items()}
    name2code = {canon(k): v.upper() for k, v in long2code.items()}
    code2hex = {name2code[k]: v for k, v in name2hex.items() if k in name2code}
    return name2hex, name2code, code2hex

def normalise_date(series: pd.Series) -> pd.Series:
    if series.dt.tz is not None:
        series = series.dt.tz_convert(None)
    return series.dt.floor("D")

def _safe_int(series: pd.Series) -> pd.Series:
    try:
        return series.astype("Int64")
    except Exception:
        return pd.to_numeric(series, errors="coerce").astype("Int64")

# ─────────────── Fallback builder (from per-team metadata) ──────────────────

def build_from_meta_per_team(
    season: str,
    fpl_csv: Path,
    meta_pt_resolved_csv: Path,
    code2hex: Dict[str, str],
    out_dir: Path,
    force: bool,
) -> bool:
    """
    Build minimal fixture_calendar from fixture_metadata_per_team_resolved.csv
    + (optionally) FPL fixtures.csv to add gw_orig and FDRs.
    """
    dst_dir = out_dir / season
    out_csv = dst_dir / "fixture_calendar.csv"
    if out_csv.exists() and not force:
        logging.info("%s • already done – skip (use --force)", season)
        return False

    if not meta_pt_resolved_csv.is_file():
        logging.warning("%s • missing %s – cannot build fallback", season, meta_pt_resolved_csv)
        return False

    meta = pd.read_csv(meta_pt_resolved_csv)
    required = {"fpl_id", "team", "opp", "venue", "date_sched"}
    missing = required - set(meta.columns)
    if missing:
        logging.error("%s • %s missing columns: %s", season, meta_pt_resolved_csv.name, sorted(missing))
        return False

    # Optional readable codes (team_short/opp_short). If absent, try to derive from teams.csv is out of scope here.
    if "team_short" not in meta.columns or "opp_short" not in meta.columns:
        logging.warning("%s • resolved file lacks team_short/opp_short; will proceed with IDs only", season)
        # We'll keep 'team' column as numeric id, but your requested mapping uses short codes.
        # Better: abort and ask user to regenerate resolved; but we keep going with placeholders.
        meta["team_short"] = meta["team"].astype(str)
        meta["opp_short"] = meta["opp"].astype(str)

    # Parse date (keep date-only string in output but compute diffs with datetime)
    meta["_date"] = pd.to_datetime(meta["date_sched"], errors="coerce", utc=True).dt.tz_convert(None)

    # Determine home/away codes per row using venue
    home_code = np.where(meta["venue"].str.lower().eq("home"), meta["team_short"], meta["opp_short"])
    away_code = np.where(meta["venue"].str.lower().eq("home"), meta["opp_short"], meta["team_short"])

    # Map codes → hex ids
    def map_hex(codes: pd.Series) -> pd.Series:
        return codes.map(lambda x: code2hex.get(str(x).upper(), pd.NA)).astype("string")

    team_code = meta["team_short"].astype(str).str.upper()
    opp_code  = meta["opp_short"].astype(str).str.upper()
    home_hex  = map_hex(pd.Series(home_code))
    away_hex  = map_hex(pd.Series(away_code))
    team_hex  = map_hex(team_code)

    # Notify about missing mappings
    missing_codes = sorted(set(pd.concat([team_code[team_hex.isna()], opp_code[away_hex.isna()]]).unique().tolist()))
    if missing_codes:
        logging.warning("%s • no team_id hex mapping for: %s", season, ", ".join(missing_codes))

    # Pull optional gw_orig and FDRs from FPL fixtures
    gw_df = pd.DataFrame()
    if fpl_csv.is_file():
        _fpl = pd.read_csv(fpl_csv, usecols=[
            "id","event","team_h_difficulty","team_a_difficulty"
        ])
        gw_df = _fpl.rename(columns={
            "id": "fpl_id",
            "event": "gw_orig",
            "team_h_difficulty": "fdr_home",
            "team_a_difficulty": "fdr_away",
        })
    else:
        logging.warning("%s • FPL fixtures.csv missing; gw_orig and FDRs will be NaN", season)

    # Assemble per-team calendar rows
    out = pd.DataFrame({
        "fpl_id":     _safe_int(meta["fpl_id"]),
        "fbref_id":   pd.NA,                      # unknown pre-season
        "gw_orig":    pd.NA,                      # will fill from gw_df merge
        "gw_played":  pd.NA,
        "date_sched": meta["date_sched"],         # keep date string (YYYY-MM-DD)
        "date_played": pd.NaT,                    # not played yet
        "days_since_last_game": 0,                # compute below
        "status":     "scheduled",
        "sched_missing": 1,                       # no fbref, so schedule details are "missing"
        "team":       team_code,                  # <- 3-letter code requested
        "team_id":    team_hex,                   # <- hex (from id_lookup)
        "was_home":   (meta["venue"].str.lower().eq("home")).astype("Int8"),
        "venue":      meta["venue"].str.lower(),
        "home":       pd.Series(home_code).astype(str).str.upper(),
        "away":       pd.Series(away_code).astype(str).str.upper(),
        "home_id":    home_hex,
        "away_id":    away_hex,
        "result":     pd.NA,
        "gf":         pd.NA,
        "ga":         pd.NA,
        "xg":         pd.NA,
        "xga":        pd.NA,
        "poss":       pd.NA,
        "is_promoted": pd.NA,
        "is_relegated": pd.NA,
        "fdr_home":   pd.NA,                      # will fill from gw_df
        "fdr_away":   pd.NA,
    })

    # Merge gw_orig + FDRs if we have them
    if not gw_df.empty:
        out = out.merge(gw_df, on="fpl_id", how="left", suffixes=("", "_gw"))
        for c in ["gw_orig","fdr_home","fdr_away"]:
            out[c] = out[c].fillna(out[f"{c}_gw"])
        out = out.drop(columns=[c for c in out.columns if c.endswith("_gw")])

    # days_since_last_game using date_sched
    tmp = out.copy()
    tmp["_date"] = pd.to_datetime(tmp["date_sched"], errors="coerce")
    tmp = tmp.sort_values(["team", "_date"], kind="mergesort")
    dslg = tmp.groupby("team")["_date"].diff().dt.days.fillna(0).astype("Int64")
    out["days_since_last_game"] = dslg.fillna(0).astype(int)

    # Final column order
    out = out.reindex(columns=OUT_COLS)

    # Write
    dst_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    logging.info("%s • fixture_calendar.csv (fallback from per-team metadata, %d rows)", season, len(out))
    return True

# ─────────────────── primary single-season builder ──────────────────────────

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
    force: bool = False,
    create_empty: bool = False,
) -> bool:
    dst_dir = out_dir / season
    out_csv = dst_dir / "fixture_calendar.csv"
    if out_csv.exists() and not force:
        logging.info("%s • already done – skip (use --force)", season)
        return False

    # ---- inputs presence gate for baseline
    if teams_csv is None:
        teams_csv = fpl_csv.with_name("teams.csv")

    fpl_missing = not fpl_csv.is_file()
    teams_missing = not teams_csv.exists()
    fb_missing = not fb_csv.is_file()

    # Load maps early (for fallback)
    long2hex = load_json(team_map_fp) if team_map_fp.is_file() else {}
    long2code = load_json(short_map_fp) if short_map_fp.is_file() else {}
    _, _, code2hex = build_maps(long2hex, long2code)

    # If FPL or teams are missing
    if fpl_missing or teams_missing:
        msg = f"{season} • missing inputs → " + ", ".join(
            [p for p, m in [(str(fpl_csv), fpl_missing), (str(teams_csv), teams_missing)] if m]
        )
        if create_empty:
            logging.warning(msg + " — creating EMPTY fixture_calendar.csv")
            write_empty_fixture_calendar(dst_dir)
            return True
        else:
            logging.warning(msg + " — skipped (use --create-empty to write empty)")
            return False

    # Try the normal (FBref) path first if fb_csv exists
    used_fallback = False
    if not fb_missing:
        try:
            # -- FPL data
            fpl = pd.read_csv(fpl_csv, parse_dates=["kickoff_time"])
            fpl = fpl.rename(
                columns={"id": "fpl_id", "event": "gw_orig", "team_h": "home_id_fpl", "team_a": "away_id_fpl"}
            )
            fpl["status"] = np.where(fpl.get("finished", False), "finished", "scheduled")
            fpl["date_played"] = normalise_date(fpl["kickoff_time"])
            fpl["date_sched"] = fpl["date_played"]
            fpl["sched_missing"] = 1

            teams_df = pd.read_csv(teams_csv, usecols=["id", "name"])
            id2name = dict(zip(teams_df.id, teams_df.name.map(canon)))

            name2hex, name2code, code2hex_norm = build_maps(long2hex, long2code)
            # prefer code2hex from maps created above (same object), but keep local alias
            code2hex = code2hex_norm

            for side in ("home", "away"):
                fpl[f"{side}_long"] = fpl[f"{side}_id_fpl"].map(id2name)
                fpl[side] = fpl[f"{side}_long"].map(name2code)
                fpl[f"{side}_hex"] = fpl[side].map(code2hex)

            # -- FBref schedule
            fb = pd.read_csv(fb_csv, parse_dates=["game_date"])
            if fb.empty:
                raise ValueError("FBref schedule is empty")

            fb["date_played"] = normalise_date(fb["game_date"])
            fb_match = fb[[
                "game_id","team","team_id","home","away","date_played",
                "venue","ga","gf","xga","xg","result","poss",
                "is_promoted","is_relegated"
            ]].copy()
            fb_match.loc[:, "was_home"] = (fb_match["venue"].eq("Home")).astype("Int8")

            # -- Merge FPL + FBref
            cal = fpl.merge(
                fb_match,
                on=["home", "away", "date_played"],
                how="left"
            )

            # If merge produced no fbref rows, go fallback
            if cal["game_id"].notna().sum() == 0:
                raise ValueError("FBref merge produced no matches; using fallback")

            cal["gw_played"] = cal["gw_orig"]

            # derive team_id (hex)
            if "team_id" not in cal.columns:
                cal["team_id"] = cal["team"].map(code2hex)
            else:
                cal.loc[:, "team_id"].fillna(cal["team"].map(code2hex), inplace=True)

            # days since last match
            cal = cal.sort_values(["team", "date_played"]).copy()
            cal["days_since_last_game"] = (
                cal.groupby("team")["date_played"].diff().dt.days.fillna(0).astype(int)
            )

            # -- inject FDR from team_form (optional)
            tfp = features_root / features_version / season / "team_form.csv"
            if tfp.is_file():
                tf = pd.read_csv(tfp, usecols=["fpl_id", "team_id", "fdr"])
                # home FDR
                cal = cal.merge(
                    tf.rename(columns={"team_id": "home_team_id", "fdr": "fdr_home"}),
                    left_on=["fpl_id", "home_hex"],
                    right_on=["fpl_id", "home_team_id"],
                    how="left"
                ).drop(columns=["home_team_id"])
                # away FDR
                cal = cal.merge(
                    tf.rename(columns={"team_id": "away_team_id", "fdr": "fdr_away"}),
                    left_on=["fpl_id", "away_hex"],
                    right_on=["fpl_id", "away_team_id"],
                    how="left"
                ).drop(columns=["away_team_id"])
            else:
                logging.warning("%s • team_form.csv not found; skipping FDR injection", season)

            # venue/was_home consistency (guard)
            pred = (cal["team_id"].astype(str) == cal["home_hex"].astype(str)).astype("Int8")
            if "was_home" in cal.columns:
                if not (cal["was_home"].astype("Int8") == pred).all():
                    logging.warning("%s • was_home != (team_id==home_id) for some rows", season)

            # -- select & rename for output
            out = cal[[
                "fpl_id", "game_id", "gw_orig", "gw_played",
                "date_sched", "date_played", "days_since_last_game",
                "status", "sched_missing",
                "team", "team_id", "was_home", "venue",
                "home", "away", "home_hex", "away_hex",
                "result", "gf", "ga", "xg", "xga", "poss",
                "is_promoted", "is_relegated",
                "fdr_home", "fdr_away",
            ]].rename(columns={
                "game_id":  "fbref_id",
                "home_hex": "home_id",
                "away_hex": "away_id",
            }).reindex(columns=OUT_COLS)

            # Fix sched_missing: 0 if we matched an FBref row, else 1
            out["sched_missing"] = out["fbref_id"].isna().astype(int)

            # Write
            dst_dir.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_csv, index=False)
            logging.info("%s • fixture_calendar.csv (%d rows)", season, len(out))
            if out["fbref_id"].isna().any():
                missing = cal.loc[cal.game_id.isna()]
                missing.to_csv(dst_dir / "_manual_fbref_match.csv", index=False)
                logging.warning("%s • %d rows lack fbref_id", season, out["fbref_id"].isna().sum())
            return True

        except Exception as e:
            logging.warning("%s • normal path failed (%s) — switching to fallback", season, e)
            used_fallback = True

    # Fallback path (FBref missing/empty/merge failed): use per-team metadata
    meta_pt_resolved_csv = fpl_csv.with_name("fixture_metadata_per_team_resolved.csv")
    ok = build_from_meta_per_team(
        season=season,
        fpl_csv=fpl_csv,
        meta_pt_resolved_csv=meta_pt_resolved_csv,
        code2hex=code2hex,
        out_dir=out_dir,
        force=force,
    )
    if not ok and create_empty:
        write_empty_fixture_calendar(dst_dir)
        return True
    return ok

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
    force: bool,
    create_empty: bool,
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
                force=force,
                create_empty=create_empty,
            )
        except Exception:
            logging.exception("%s • unhandled error", season)

# ───────────────────────────── CLI ─────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season")
    ap.add_argument("--fpl-root", type=Path, default=Path("data/raw/fpl"))
    ap.add_argument("--fbref-league-dir", type=Path, default=Path("data/processed/fbref/ENG-Premier League"))
    ap.add_argument("--team-map", type=Path, default=Path("data/processed/_id_lookup_teams.json"))
    ap.add_argument("--short-map", type=Path, default=Path("data/config/teams.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/fixtures"))
    ap.add_argument("--features-root", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--create-empty", action="store_true",
                help="If season inputs are missing, still write an empty fixture_calendar.csv")
    ap.add_argument("--features-version", default="v2")
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
        args.force,
        create_empty=args.create_empty,
    )

if __name__ == "__main__":
    main()
