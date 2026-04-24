# scripts/fbref_utils.py
from __future__ import annotations
import logging, time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from scripts.fbref_pipeline.scrape.fbref_adapter import build_fbref_reader

STAT_MAP: Dict[str, List[str]] = {
    "team_season": [
        "standard", "keeper", "keeper_adv", "shooting", "passing", "passing_types",
        "goal_shot_creation", "defense", "possession", "playing_time", "misc",
    ],
    "team_match": [
        "schedule", "keeper", "shooting", "passing", "passing_types",
        "goal_shot_creation", "defense", "possession", "misc",
    ],
    "player_season": [
        "standard", "shooting", "passing", "passing_types", "goal_shot_creation",
        "defense", "possession", "playing_time", "misc", "keeper", "keeper_adv",
    ],
    "player_match": [
        "summary", "keepers", "passing", "passing_types", "defense",
        "possession", "misc",
    ],
}

def safe_write(df: pd.DataFrame, path: Path) -> None:
    """Write CSV + Snappy Parquet, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=True)
    logging.getLogger("fbref").debug("saved %s", path.with_suffix("").name)

def seasons_from_league(
    league: str,
    *,
    proxy: Optional[str] = None,
    browser_path: Optional[str] = None,
    headless: bool = False,
    headers: Optional[Dict[str, str]] = None,
) -> list[str]:
    fb = build_fbref_reader(
        leagues=league,
        proxy=proxy,
        browser_path=browser_path,
        headless=headless,
        headers=headers,
    )
    try:
        return fb.read_seasons().index.get_level_values("season").unique().tolist()
    finally:
        fb.close()

def init_logger(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

def polite_sleep(delay: float) -> None:
    if delay > 0:
        time.sleep(delay)
