"""FBref provider package.

The capability registry is deliberately importable without the optional
scraping stack.  Scraper objects are loaded lazily so canonical-data jobs do
not require Selenium, lxml, or soccerdata merely to inspect provider coverage.
"""

from importlib import import_module

from .capabilities import (
    FBREF_CAPABILITIES,
    HISTORICAL_UNAVAILABLE_STATS,
    CoverageRecord,
    normalize_stat_type,
    supported_stats,
    validate_requested_stats,
)

__all__ = [
    "FBREF_CAPABILITIES",
    "HISTORICAL_UNAVAILABLE_STATS",
    "CoverageRecord",
    "normalize_stat_type",
    "supported_stats",
    "validate_requested_stats",
    "PatchedFBref",
    "build_fbref_reader",
    "resolve_browser_path",
    "match_stats_scraper",
    "season_stats_scraper",
]


def __getattr__(name: str):
    if name in {"PatchedFBref", "build_fbref_reader", "resolve_browser_path"}:
        module = import_module(".scrape.fbref_adapter", __name__)
        return getattr(module, name)
    if name in {"match_stats_scraper", "season_stats_scraper"}:
        return import_module(f".scrape.{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
