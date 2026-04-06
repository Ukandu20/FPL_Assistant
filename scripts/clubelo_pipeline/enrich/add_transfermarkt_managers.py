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
DEFAULT_MANAGERS_DIR = Path("data/raw/transfermarkt/premier_league/managers")


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


def load_managers(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"name", "nationality", "appointment_date", "end_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manager file {path} missing columns: {', '.join(sorted(missing))}")
    out = df.copy()
    out["appointment_date"] = pd.to_datetime(out["appointment_date"], errors="coerce")
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    out = out.dropna(subset=["appointment_date"]).sort_values("appointment_date")
    return out


def add_manager_columns(
    clubelo_df: pd.DataFrame,
    managers_df: pd.DataFrame,
    *,
    date_col: str = "from",
) -> pd.DataFrame:
    out = clubelo_df.copy()
    for col in ("manager", "manager_nationality"):
        if col in out.columns:
            out = out.drop(columns=[col])
    out["manager"] = ""
    out["manager_nationality"] = ""

    if date_col not in out.columns or managers_df.empty:
        return out

    row_dates = pd.to_datetime(out[date_col], errors="coerce")
    left = pd.DataFrame({"_row_index": out.index, date_col: row_dates}).dropna(subset=[date_col])
    if left.empty:
        return out

    merged = pd.merge_asof(
        left.sort_values(date_col),
        managers_df[["appointment_date", "end_date", "name", "nationality"]].sort_values("appointment_date"),
        left_on=date_col,
        right_on="appointment_date",
        direction="backward",
    )
    valid = merged["appointment_date"].notna() & (
        merged["end_date"].isna() | (merged[date_col] <= merged["end_date"])
    )
    merged = merged.loc[valid, ["_row_index", "name", "nationality"]]
    if merged.empty:
        return out

    out.loc[merged["_row_index"], "manager"] = merged["name"].astype(str).values
    out.loc[merged["_row_index"], "manager_nationality"] = merged["nationality"].astype(str).values
    return out


def process_pair(clubelo_path: Path, manager_path: Path, *, date_col: str) -> tuple[int, int]:
    clubelo_df = pd.read_csv(clubelo_path, dtype=str, keep_default_na=False)
    managers_df = load_managers(manager_path)
    enriched = add_manager_columns(clubelo_df, managers_df, date_col=date_col)
    filled_rows = int((enriched["manager"] != "").sum())
    total_rows = len(enriched)
    enriched.to_csv(clubelo_path, index=False)
    return filled_rows, total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Transfermarkt manager names to ClubElo team-history CSVs."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--clubelo-dir", type=Path, default=DEFAULT_CLUBELO_DIR)
    parser.add_argument("--managers-dir", type=Path, default=DEFAULT_MANAGERS_DIR)
    parser.add_argument("--date-col", default="from")
    parser.add_argument("--team-code", default="", help="Optional single team code filter, e.g. MUN")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest = load_manifest(args.manifest)
    team_filter = args.team_code.strip().upper()
    total_files = 0
    total_rows = 0
    total_filled = 0

    for item in manifest:
        if team_filter and item.team_code != team_filter:
            continue
        manager_path = args.managers_dir / f"{item.team_code}.csv"
        clubelo_path = args.clubelo_dir / f"{item.clubelo_stem}.csv"
        if not manager_path.exists():
            logging.info("Skipping %s: manager file missing at %s", item.team_code, manager_path)
            continue
        if not clubelo_path.exists():
            logging.info("Skipping %s: ClubElo file missing at %s", item.team_code, clubelo_path)
            continue
        filled_rows, total = process_pair(clubelo_path, manager_path, date_col=args.date_col)
        total_files += 1
        total_rows += total
        total_filled += filled_rows
        logging.info("Updated %s: %s/%s rows assigned a manager", clubelo_path.name, filled_rows, total)

    logging.info(
        "Finished manager enrichment for %s files: %s/%s rows assigned a manager",
        total_files,
        total_filled,
        total_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
