import pandas as pd
import pytest

from scripts.understat_pipeline.scrape.understat_stats_scraper import (
    _empty_indexed_frame,
    normalize_understat_season,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2025", "2025"),
        ("2025-2026", "2025"),
        ("2025-26", "2025"),
        ("2025/2026", "2025"),
    ],
)
def test_normalize_understat_season(value, expected):
    assert normalize_understat_season(value) == expected


def test_normalize_understat_season_rejects_nonconsecutive_years():
    with pytest.raises(ValueError, match="consecutive years"):
        normalize_understat_season("2025-2027")


def test_empty_indexed_frame_has_named_multiindex():
    frame = _empty_indexed_frame(["league", "season", "game"])

    assert frame.empty
    assert isinstance(frame.index, pd.MultiIndex)
    assert frame.index.names == ["league", "season", "game"]
