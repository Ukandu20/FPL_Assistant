#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_MANIFEST = Path("data/config/transfermarkt_premier_league_clubs.json")
DEFAULT_CLUBELO_DIR = Path("data/raw/clubelo/team_history")
DEFAULT_FBREF_DIR = Path("data/processed/fbref/ENG-Premier League")

MATCH_COLS = [
    "fbref_match_id",
    "fbref_match_date",
    "fbref_match_season",
    "fbref_match_competition",
    "fbref_match_round",
    "fbref_match_home",
    "fbref_match_away",
    "fbref_match_opponent",
    "fbref_match_result",
    "fbref_match_is_home",
    "fbref_match_delta_days",
]


@dataclass(frozen=True)
class ClubMapping:
    team_code: str
    clubelo_stem: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ClubMapping":
        required = {"team_code", "clubelo_stem"}
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"Manifest entry missing required keys: {', '.join(missing)}")
        return cls(
            team_code=str(raw["team_code"]).upper(),
            clubelo_stem=str(raw["clubelo_stem"]).strip(),
        )


def load_manifest(path: Path) -> list[ClubMapping]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in manifest: {path}")
    return [ClubMapping.from_dict(item) for item in raw]


def load_team_schedule_rows(fbref_dir: Path, team_code: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(fbref_dir.glob("*/team_match/schedule.csv")):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception:
            continue
        if "team" not in df.columns or "date" not in df.columns:
            continue
        team_df = df.loc[df["team"].astype(str).str.upper().eq(team_code)].copy()
        if team_df.empty:
            continue
        if "season" not in team_df.columns:
            team_df["season"] = path.parent.parent.name
        if "league" not in team_df.columns:
            team_df["league"] = "ENG-Premier League"
        parts.append(team_df)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(["date", "game_id"], kind="stable")
    subset = ["game_id"] if "game_id" in out.columns else ["date", "home", "away"]
    out = out.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    return out


def _empty_match_columns(out: pd.DataFrame) -> pd.DataFrame:
    enriched = out.copy()
    for col in MATCH_COLS:
        if col in enriched.columns:
            enriched = enriched.drop(columns=[col])
    for col in MATCH_COLS:
        enriched[col] = ""
    return enriched


def add_fbref_match_columns(
    clubelo_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    *,
    team_code: str,
    date_col: str = "from",
    country_filter: str = "ENG",
    level_filter: str = "1",
    max_offset_days: int = 1,
) -> pd.DataFrame:
    out = _empty_match_columns(clubelo_df)

    if date_col not in out.columns or schedule_df.empty:
        return out

    row_dates = pd.to_datetime(out[date_col], errors="coerce")
    eligible = row_dates.notna()

    if country_filter and "country" in out.columns:
        eligible &= out["country"].astype(str).str.strip().eq(country_filter)
    if level_filter and "level" in out.columns:
        eligible &= out["level"].astype(str).str.strip().eq(str(level_filter))

    if not eligible.any():
        return out

    team_schedule = schedule_df.copy()
    if "team" in team_schedule.columns:
        team_schedule = team_schedule.loc[
            team_schedule["team"].astype(str).str.upper().eq(team_code.upper())
        ].copy()
    if team_schedule.empty or "date" not in team_schedule.columns:
        return out

    team_schedule["date"] = pd.to_datetime(team_schedule["date"], errors="coerce").dt.normalize()
    team_schedule = team_schedule.dropna(subset=["date"]).sort_values(
        ["date", "game_id"], kind="stable"
    )
    if team_schedule.empty:
        return out

    candidates = pd.DataFrame(
        {
            "_row_index": out.index[eligible],
            "_row_day": row_dates.loc[eligible].dt.normalize(),
        }
    ).sort_values(["_row_day", "_row_index"], kind="stable")

    unused = set(candidates["_row_index"].tolist())
    if not unused:
        return out

    for _, match in team_schedule.iterrows():
        match_date = match["date"]
        available = candidates.loc[candidates["_row_index"].isin(unused)].copy()
        if available.empty:
            break
        available["_delta_days"] = (available["_row_day"] - match_date).dt.days
        matched = available.loc[
            (available["_delta_days"] >= 0) & (available["_delta_days"] <= max_offset_days)
        ].sort_values(["_delta_days", "_row_day", "_row_index"], kind="stable")
        if matched.empty:
            continue

        chosen = matched.iloc[0]
        row_index = int(chosen["_row_index"])
        unused.remove(row_index)

        home = str(match.get("home", "")).strip()
        away = str(match.get("away", "")).strip()
        opponent = str(match.get("opponent", "")).strip()
        if not opponent:
            opponent = away if home.upper() == team_code.upper() else home

        if "is_home" in match.index:
            is_home = str(match.get("is_home", "")).strip()
        else:
            is_home = "1" if home.upper() == team_code.upper() else "0"

        out.at[row_index, "fbref_match_id"] = str(match.get("game_id", "")).strip()
        out.at[row_index, "fbref_match_date"] = match_date.date().isoformat()
        out.at[row_index, "fbref_match_season"] = str(match.get("season", "")).strip()
        out.at[row_index, "fbref_match_competition"] = str(
            match.get("league", "ENG-Premier League")
        ).strip()
        out.at[row_index, "fbref_match_round"] = str(match.get("round", "")).strip()
        out.at[row_index, "fbref_match_home"] = home
        out.at[row_index, "fbref_match_away"] = away
        out.at[row_index, "fbref_match_opponent"] = opponent
        out.at[row_index, "fbref_match_result"] = str(match.get("result", "")).strip()
        out.at[row_index, "fbref_match_is_home"] = is_home
        out.at[row_index, "fbref_match_delta_days"] = str(int(chosen["_delta_days"]))

    return out


def process_pair(
    clubelo_path: Path,
    schedule_df: pd.DataFrame,
    *,
    team_code: str,
    date_col: str,
    country_filter: str,
    level_filter: str,
    max_offset_days: int,
) -> tuple[int, int, int]:
    clubelo_df = pd.read_csv(clubelo_path, dtype=str, keep_default_na=False)
    enriched = add_fbref_match_columns(
        clubelo_df,
        schedule_df,
        team_code=team_code,
        date_col=date_col,
        country_filter=country_filter,
        level_filter=level_filter,
        max_offset_days=max_offset_days,
    )
    matched_rows = int((enriched["fbref_match_id"] != "").sum())
    total_rows = len(enriched)
    total_matches = len(schedule_df)
    enriched.to_csv(clubelo_path, index=False)
    return matched_rows, total_rows, total_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tag ClubElo team-history rows with FBref Premier League match identifiers."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--clubelo-dir", type=Path, default=DEFAULT_CLUBELO_DIR)
    parser.add_argument("--fbref-dir", type=Path, default=DEFAULT_FBREF_DIR)
    parser.add_argument("--date-col", default="from")
    parser.add_argument("--country-filter", default="ENG")
    parser.add_argument("--level-filter", default="1")
    parser.add_argument("--max-offset-days", type=int, default=1)
    parser.add_argument("--team-code", default="", help="Optional single team code filter, e.g. MUN")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest = load_manifest(args.manifest)
    team_filter = args.team_code.strip().upper()
    total_files = 0
    total_rows = 0
    total_matched = 0
    total_fixtures = 0

    for item in manifest:
        if team_filter and item.team_code != team_filter:
            continue
        clubelo_path = args.clubelo_dir / f"{item.clubelo_stem}.csv"
        if not clubelo_path.exists():
            logging.info("Skipping %s: ClubElo file missing at %s", item.team_code, clubelo_path)
            continue

        schedule_df = load_team_schedule_rows(args.fbref_dir, item.team_code)
        if schedule_df.empty:
            logging.info("Skipping %s: no FBref team schedule rows found", item.team_code)
            continue

        matched_rows, row_count, fixture_count = process_pair(
            clubelo_path,
            schedule_df,
            team_code=item.team_code,
            date_col=args.date_col,
            country_filter=args.country_filter,
            level_filter=args.level_filter,
            max_offset_days=args.max_offset_days,
        )
        total_files += 1
        total_rows += row_count
        total_matched += matched_rows
        total_fixtures += fixture_count
        logging.info(
            "Updated %s: tagged %s ClubElo rows from %s FBref fixtures",
            clubelo_path.name,
            matched_rows,
            fixture_count,
        )

    logging.info(
        "Finished FBref match enrichment for %s files: tagged %s/%s ClubElo rows from %s fixtures",
        total_files,
        total_matched,
        total_rows,
        total_fixtures,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
