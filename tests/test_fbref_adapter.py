from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd

from fpl_assistant.providers.fbref import (
    PatchedFBref,
    build_fbref_reader,
    match_stats_scraper,
    resolve_browser_path,
    season_stats_scraper,
)
from fpl_assistant.testing.paths import get_test_run_dir, get_test_soccerdata_dir


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "fbref_adapter"
TMP_ROOT = get_test_run_dir("fbref_adapter")
os.environ.setdefault("SOCCERDATA_DIR", str(get_test_soccerdata_dir().resolve()))


def _workspace_tmp(name: str) -> Path:
    path = TMP_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_resolve_browser_path_prefers_first_available(monkeypatch):
    tmp_path = _workspace_tmp("browser_order")
    chrome = tmp_path / "chrome.exe"
    edge = tmp_path / "msedge.exe"
    chrome.write_text("", encoding="utf-8")
    edge.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.fbref_pipeline.scrape.fbref_adapter.DEFAULT_BROWSER_CANDIDATES",
        (chrome, edge),
    )

    assert resolve_browser_path() == chrome


def test_resolve_browser_path_honors_explicit_path():
    tmp_path = _workspace_tmp("browser_explicit")
    browser = tmp_path / "custom-browser.exe"
    browser.write_text("", encoding="utf-8")

    assert resolve_browser_path(browser) == browser


def test_build_fbref_reader_preserves_factory_args():
    tmp_path = _workspace_tmp("builder_args")
    browser = tmp_path / "chrome.exe"
    browser.write_text("", encoding="utf-8")

    reader = build_fbref_reader(
        leagues="ENG-Premier League",
        seasons="2025-2026",
        browser_path=str(browser),
        headless=True,
        headers={"User-Agent": "pytest-agent"},
        no_cache=True,
    )
    try:
        assert isinstance(reader, PatchedFBref)
        assert reader.browser_path == browser
        assert reader.headless is True
        assert reader.no_cache is True
        assert reader.headers["User-Agent"] == "pytest-agent"
        assert "sec-ch-ua" in reader.headers
    finally:
        reader.close()


class _FixtureFBref(PatchedFBref):
    def read_seasons(self, split_up_big5: bool = False) -> pd.DataFrame:
        index = pd.MultiIndex.from_tuples(
            [("ENG-Premier League", "2025-2026")],
            names=["league", "season"],
        )
        return pd.DataFrame(
            {"format": ["round-robin"], "url": ["/en/comps/9/2025-2026/2025-2026-Premier-League-Stats"]},
            index=index,
        )

    def get(self, url, filepath=None, max_age=None, no_cache=False, var=None):
        if "2025-2026-Premier-League-Stats" in url:
            return io.BytesIO((FIXTURE_DIR / "overview.html").read_bytes())
        if "Premier-League-Scores-and-Fixtures" in url:
            return io.BytesIO((FIXTURE_DIR / "schedule.html").read_bytes())
        raise AssertionError(f"Unexpected URL in fixture reader: {url}")


def test_schedule_parsing_from_local_fixture():
    reader = _FixtureFBref(leagues="ENG-Premier League", seasons="2025-2026", no_store=True)
    try:
        df = reader.read_schedule()
    finally:
        reader.close()

    row = df.loc[("ENG-Premier League", "2025-2026", "2025-08-15 Arsenal-Chelsea")]
    assert row["week"] == 1
    assert row["home_team"] == "Arsenal"
    assert row["away_team"] == "Chelsea"
    assert row["game_id"] == "abc123"


class _FakeReader:
    def __init__(self, *, data_dir: Path):
        self.data_dir = data_dir

    def read_leagues(self, split_up_big5: bool = False):
        return pd.DataFrame()

    def close(self):
        return None


def test_match_scraper_uses_shared_builder(monkeypatch):
    calls: list[dict] = []
    tmp_path = _workspace_tmp("match_scraper")
    out_base = tmp_path / "out"
    season_dir = out_base / "ENG-Premier League" / "2025-2026"
    season_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"league": "ENG-Premier League", "season": "2025-2026", "date": "2025-08-15"}]
    ).to_csv(season_dir / "schedule.csv", index=False)

    def _fake_builder(**kwargs):
        calls.append(kwargs)
        return _FakeReader(data_dir=tmp_path / "cache")

    monkeypatch.setattr(match_stats_scraper, "build_fbref_reader", _fake_builder)
    monkeypatch.setattr(match_stats_scraper, "_enable_cached_leagues_fallback", lambda fb: True)
    monkeypatch.setattr(match_stats_scraper, "_ensure_leagues_html_in_data_dir", lambda fb, path: False)

    match_stats_scraper.scrape_one(
        "ENG-Premier League",
        "2025-2026",
        out_base,
        0.0,
        levels="both",
        player_stats=[],
        team_stats=[],
        skip_existing=True,
        skip_schedule=True,
        browser_path=str(tmp_path / "chrome.exe"),
        headless=False,
        headers={"User-Agent": "pytest-agent"},
    )

    assert len(calls) >= 1
    assert calls[0]["browser_path"] == str(tmp_path / "chrome.exe")
    assert calls[0]["headers"]["User-Agent"] == "pytest-agent"


def test_season_scraper_uses_shared_builder(monkeypatch):
    calls: list[dict] = []
    tmp_path = _workspace_tmp("season_scraper")

    def _fake_builder(**kwargs):
        calls.append(kwargs)
        return _FakeReader(data_dir=tmp_path / "cache")

    monkeypatch.setattr(season_stats_scraper, "build_fbref_reader", _fake_builder)

    season_stats_scraper.scrape_one(
        "ENG-Premier League",
        "2025-2026",
        tmp_path / "out",
        0.0,
        levels="both",
        player_stats=[],
        team_stats=[],
        browser_path=str(tmp_path / "chrome.exe"),
        headless=True,
    )

    assert len(calls) >= 1
    assert calls[0]["browser_path"] == str(tmp_path / "chrome.exe")
    assert calls[0]["headless"] is True
