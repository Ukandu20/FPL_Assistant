from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import pandas as pd

from apps.fpl.catalog import current_season, discover_seasons


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def load_page_module(filename: str):
    path = ROOT / "apps" / "fpl" / "pages" / filename
    spec = importlib.util.spec_from_file_location(
        f"fpl_app_test_{path.stem}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_returns_newest_valid_season(tmp_path: Path) -> None:
    league = tmp_path / "Premier League"
    for season in ("2024-2025", "2026-2027", "2025-2026"):
        (league / season / "season").mkdir(parents=True)
    (league / "2024-2025" / "season" / "players.csv").touch()
    (league / "2026-2027" / "season" / "players.csv").touch()

    assert discover_seasons(
        "Premier League", root=tmp_path, required_path="season/players.csv"
    ) == ["2026-2027", "2024-2025"]
    assert current_season(
        "Premier League", root=tmp_path, required_path="season/players.csv"
    ) == "2026-2027"


def test_player_season_duplicates_keep_best_supported_row() -> None:
    player_page = load_page_module("0_main.py")
    players = pd.DataFrame(
        [
            {
                "season": "2025-2026",
                "player_id": "same-id",
                "name": "Player A",
                "team": "AAA",
                "minutes": 90,
                "total_points": 3,
            },
            {
                "season": "2025-2026",
                "player_id": "same-id",
                "name": "Player A",
                "team": "BBB",
                "minutes": 900,
                "total_points": 60,
            },
        ]
    )

    resolved = player_page.deduplicate_player_seasons(players)

    assert len(resolved) == 1
    assert resolved.iloc[0]["team"] == "BBB"


def test_percentiles_reverse_adverse_metrics_and_gate_small_samples() -> None:
    player_page = load_page_module("0_main.py")
    players = pd.DataFrame(
        {
            "season": ["2025-2026"] * 3,
            "player_id": ["a", "b", "c"],
            "name": ["A", "B", "C"],
            "fpl_pos": ["MID"] * 3,
            "minutes": [900, 900, 90],
            "total_points": [100, 50, 20],
            "yellow_cards": [1, 5, 0],
        }
    )

    totals = player_page.add_metric_percentiles(players)
    assert totals.loc[0, "total_points_league_percentile"] == 100
    assert totals.loc[0, "yellow_cards_league_percentile"] > totals.loc[
        1, "yellow_cards_league_percentile"
    ]

    per_90 = player_page.add_metric_percentiles(players, per_90=True)
    assert pd.isna(per_90.loc[2, "total_points_league_percentile"])
    assert pd.isna(per_90.loc[2, "total_points"])


def test_team_player_table_adds_recent_form_and_value() -> None:
    team_page = load_page_module("2_teams.py")
    players = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "name": ["Alpha", "Bravo"],
            "team": ["AAA", "AAA"],
            "fpl_pos": ["MID", "DEF"],
            "now_cost": [50, 60],
            "total_points": [20, 24],
            "minutes": [500, 600],
        }
    )
    gameweeks = pd.DataFrame(
        {
            "player_id": ["a", "a", "b"],
            "team": ["AAA", "AAA", "AAA"],
            "round": [4, 5, 5],
            "total_points": [5, 7, 4],
            "minutes": [90, 90, 90],
        }
    )

    result = team_page.build_team_player_table(
        players, gameweeks, "AAA", recent_rounds=3
    )

    alpha = result.loc[result["Player"].eq("Alpha")].iloc[0]
    assert alpha["Last 3 points"] == 12
    assert alpha["Points/£m"] == 4
