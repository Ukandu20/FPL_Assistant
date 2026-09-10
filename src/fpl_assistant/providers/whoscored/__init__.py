"""WhoScored provider with lazy optional scraper imports."""

from importlib import import_module


def __getattr__(name: str):
    if name == "whoscored_match_stats_scraper":
        return import_module(
            ".scrape.whoscored_match_stats_scraper", __name__
        )
    if name == "whoscored_scraper":
        return import_module(".scrape.whoscored_scraper", __name__)

    native = import_module(".scrape.whoscored_native_backend", __name__)
    try:
        return getattr(native, name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc


__all__ = [
    "whoscored_match_stats_scraper",
    "whoscored_scraper",
    "CompetitionConfig",
    "extract_match_centre_payload",
    "parse_calendar_mask",
    "parse_embedded_tournament_fixtures",
    "parse_missing_players_html",
    "parse_schedule_month_payload",
    "parse_season_options",
]
