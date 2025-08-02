#!/usr/bin/env python3
"""
assign_ids.py – Attach team_id, opponent_id & fbref game_id
                to every FPL merged_gws.csv across seasons.

Post-conditions per season
──────────────────────────
1. <season>/gws/merged_gws.csv           ← overwritten with IDs attached
2. <season>/_manual_review/
       assign_ids_missing_rows.json      ← skipped rows & reasons
"""

from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

import pandas as pd

pd.set_option("mode.chained_assignment", None)     # silence warnings


# ───────────────────────────── utils ────────────────────────────
def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text("utf-8"))
    except FileNotFoundError:
        logging.error("missing %s", path); raise


def normalise_maps(team_map: dict, short_map: dict):
    to_hex = {k.strip().lower(): v.strip().upper() for k, v in team_map.items()}
    to_code = {k.strip().lower(): v.strip().upper() for k, v in short_map.items()}
    return to_hex, to_code


def build_teams(raw_csv: Path, to_hex: dict, to_code: dict) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    df["team"]   = df["name"].str.strip().str.lower().map(to_code)
    df["team_id"] = df["name"].str.strip().str.lower().map(to_hex)
    df["opponent_id"] = df["id"]               # numeric id for look-ups
    return df


def collect_skips(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with any missing critical value + aggregated reason(s)."""
    reasons = []
    for _, row in df.iterrows():
        miss = []
        if pd.isna(row["team_id"]):      miss.append("missing_team_id")
        if pd.isna(row["opponent_id"]):  miss.append("missing_opponent_id")
        if pd.isna(row["game_id"]):      miss.append("missing_game_id")
        if miss:
            rdict = row.to_dict()
            rdict["reason"] = ";".join(miss)
            reasons.append(rdict)
    return pd.DataFrame(reasons)


def seasons_in(root: Path):
    return sorted(p.name for p in root.iterdir() if p.is_dir())


# ────────────────────────── main worker ─────────────────────────
def process_season(season: str, paths, to_hex, to_code):
    gws_csv   = paths["fpl"] / season / "gws" / "merged_gws.csv"
    fb_csv    = paths["fbref"] / season / "player_match" / "misc.csv"
    teams_csv = paths["raw_teams"] / season / "teams.csv"
    review_dir = paths["fpl"] / season / "_manual_review"
    review_fp  = review_dir / "assign_ids_missing_rows.json"

    if not gws_csv.exists():
        logging.warning("skip – %s missing", gws_csv); return

    teams = build_teams(teams_csv, to_hex, to_code)
    id2hex   = teams.set_index("id")["team_id"].to_dict()
    code_lkp = teams.set_index("team_id")["team"]

    # -- load GW data
    gws = pd.read_csv(
        gws_csv,
    )
    gws = gws[gws["player_id"].notna()].copy()

    gws["opponent_id"]   = gws["opponent_team"].map(id2hex)
    gws["opponent_team"] = gws["opponent_id"].map(code_lkp)
    gws["team_id"]       = gws["team"].map(teams.set_index("team")["team_id"])
    gws["game_date"]     = pd.to_datetime(gws["kickoff_time"], utc=True).dt.date.astype(str)

    # drop legacy manager metrics if present
    gws.drop(columns=[c for c in gws.columns if c.startswith("mng_")],
             inplace=True, errors="ignore")

    # -- fbref merge on home/away only
    fb = pd.read_csv(fb_csv)
    fixture_df = (
        gws[gws["was_home"]]
          .rename(columns={"team":"home","opponent_team":"away"})
          .loc[:,["home","away"]]
          .drop_duplicates()
    )
    bridge = fixture_df.merge(
        fb,
        left_on=["home","away"],
        right_on=["is_home","is_away"],
        how="left"
    ).loc[:,["home","away","game_id"]]

    gws = gws.merge(
        bridge,
        left_on=["team","opponent_team"],
        right_on=["home","away"],
        how="left"
    ).drop(columns=["home","away"])

    # -- collect & drop skips
    skips = collect_skips(gws)
    if not skips.empty:
        review_dir.mkdir(parents=True, exist_ok=True)
        review_fp.write_text(skips.to_json(orient="records", indent=2))
        gws = gws.drop(index=skips.index)
        logging.info("▲ %d rows skipped → %s", len(skips), review_fp)

    # -- overwrite original CSV
    gws.to_csv(gws_csv, index=False)
    logging.info("✓ %s (%d rows retained)", gws_csv, len(gws))


# ───────────────────────────── CLI ──────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fpl-root",   required=True, type=Path,
                   help="data/processed/fpl")
    p.add_argument("--fbref-root", required=True, type=Path,
                   help="data/processed/fbref/ENG-Premier League")
    p.add_argument("--raw-teams",  required=True, type=Path,
                   help="data/raw/fpl")
    p.add_argument("--team-map",   required=True, type=Path,
                   help="_id_lookup_teams.json")
    p.add_argument("--short-code", required=True, type=Path,
                   help="teams.json")
    p.add_argument("--seasons", help="2024-25 or 2023-24,2024-25; default = all")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    to_hex, to_code = normalise_maps(
        read_json(args.team_map), read_json(args.short_code)
    )
    paths = {"fpl": args.fpl_root, "fbref": args.fbref_root,
             "raw_teams": args.raw_teams}

    chosen = (
         args.seasons.split(",") if args.seasons
         else seasons_in(args.fpl_root)
     )

    for season in chosen:
        logging.info("── %s ──", season)
        process_season(season, paths, to_hex, to_code)


if __name__ == "__main__":
    main()
