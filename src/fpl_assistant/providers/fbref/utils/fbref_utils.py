# Shared FBref reader utilities
from __future__ import annotations
import logging, time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from fpl_assistant.providers.fbref.capabilities import FBREF_CAPABILITIES

STAT_MAP: Dict[str, List[str]] = {
    level: list(stats)
    for level, stats in FBREF_CAPABILITIES.items()
    if level != "supplementary"
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
    from ..scrape.fbref_adapter import build_fbref_reader

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
