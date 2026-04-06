#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

DEFAULT_CLUBELO_DIR = Path("data/raw/clubelo/team_history")
DEFAULT_SEASONS_CSV = Path("data/raw/pl_seasons.csv")


def load_season_windows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"season", "competition", "start_date", "finish_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Season file missing columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["finish_date"] = pd.to_datetime(df["finish_date"], errors="coerce")
    df = df.dropna(subset=["start_date", "finish_date"]).sort_values("start_date")
    return df[["season", "competition", "start_date", "finish_date"]]


def add_season_band_columns(
    df: pd.DataFrame,
    seasons_df: pd.DataFrame,
    *,
    date_col: str = "from",
    country_filter: str = "ENG",
    level_filter: str = "1",
) -> pd.DataFrame:
    out = df.copy()
    for legacy_col in ("season_band", "season_band_competition"):
        if legacy_col in out.columns:
            out = out.drop(columns=[legacy_col])
    out["season"] = ""
    out["league"] = ""

    if date_col not in out.columns:
        return out

    date_series = pd.to_datetime(out[date_col], errors="coerce")
    eligible = date_series.notna()

    if country_filter and "country" in out.columns:
        eligible &= out["country"].astype(str).str.strip().eq(country_filter)
    if level_filter and "level" in out.columns:
        eligible &= out["level"].astype(str).str.strip().eq(str(level_filter))

    if not eligible.any():
        return out

    lookup_left = pd.DataFrame(
        {
            "_row_index": out.index[eligible],
            date_col: date_series.loc[eligible],
        }
    ).sort_values(date_col)

    lookup = pd.merge_asof(
        lookup_left,
        seasons_df.sort_values("start_date"),
        left_on=date_col,
        right_on="start_date",
        direction="backward",
    )
    valid = lookup["finish_date"].notna() & (lookup[date_col] <= lookup["finish_date"])
    lookup = lookup.loc[valid, ["_row_index", "season"]]

    if lookup.empty:
        return out

    season_map = lookup.set_index("_row_index")["season"]

    out.loc[season_map.index, "season"] = season_map.astype(str)
    out.loc[season_map.index, "league"] = "ENG-Premier League"
    return out


def process_clubelo_file(
    path: Path,
    seasons_df: pd.DataFrame,
    *,
    date_col: str,
    country_filter: str,
    level_filter: str,
) -> tuple[int, int]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    enriched = add_season_band_columns(
        df,
        seasons_df,
        date_col=date_col,
        country_filter=country_filter,
        level_filter=level_filter,
    )
    banded_rows = int((enriched["season"] != "").sum())
    total_rows = len(enriched)
    enriched.to_csv(path, index=False)
    return banded_rows, total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Premier League season-band columns to ClubElo team-history CSVs."
    )
    parser.add_argument("--clubelo-dir", type=Path, default=DEFAULT_CLUBELO_DIR)
    parser.add_argument("--seasons-csv", type=Path, default=DEFAULT_SEASONS_CSV)
    parser.add_argument("--glob", default="*.csv")
    parser.add_argument("--date-col", default="from")
    parser.add_argument("--country-filter", default="ENG")
    parser.add_argument("--level-filter", default="1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    seasons_df = load_season_windows(args.seasons_csv)
    files = sorted(args.clubelo_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No ClubElo CSVs found under {args.clubelo_dir} with glob {args.glob}")

    total_files = 0
    total_rows = 0
    total_banded = 0

    for path in files:
        banded_rows, row_count = process_clubelo_file(
            path,
            seasons_df,
            date_col=args.date_col,
            country_filter=args.country_filter,
            level_filter=args.level_filter,
        )
        total_files += 1
        total_rows += row_count
        total_banded += banded_rows
        logging.info("Updated %s: %s/%s rows banded", path.name, banded_rows, row_count)

    logging.info(
        "Finished season-band enrichment for %s files: %s/%s rows banded",
        total_files,
        total_banded,
        total_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
