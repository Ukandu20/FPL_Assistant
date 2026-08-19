from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import pandas as pd

from apps.fpl.catalog import current_season, discover_seasons, latest_archetype_snapshot


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def load_page_module(filename: str):
    path = ROOT / "apps" / "fpl" / "pages" / filename
    spec = importlib.util.spec_from_file_location(f"fpl_app_test_{path.stem}", path)
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


def test_catalog_selects_latest_archetype_snapshot_within_season(
    tmp_path: Path,
) -> None:
    older = tmp_path / "model_version=1.0.0" / "snapshot=2026-01-01T00-00-00Z"
    latest = tmp_path / "model_version=1.0.0" / "snapshot=2026-05-01T00-00-00Z"
    outside = tmp_path / "model_version=1.0.0" / "snapshot=2026-08-01T00-00-00Z"
    for path in (older, latest, outside):
        path.mkdir(parents=True)
        (path / "archetypes.jsonl").touch()

    assert latest_archetype_snapshot("2025-2026", root=tmp_path) == latest


def test_overview_archetype_tags_prioritize_summary_usage_and_risk() -> None:
    player_page = load_page_module("0_main.py")
    archetypes = pd.DataFrame(
        [
            {
                "archetype_id": "COMPLETE_FORWARD",
                "family": "Production Composite",
                "display_name": "Complete Forward",
                "score_0_100": 82,
                "active_label": True,
                "confidence_band": "High",
            },
            {
                "archetype_id": "GOAL_THREAT",
                "family": "Production Style",
                "display_name": "Goal Threat",
                "score_0_100": 91,
                "active_label": True,
                "confidence_band": "High",
            },
            {
                "archetype_id": "REGULAR_STARTER",
                "family": "Usage",
                "display_name": "Regular Starter",
                "score_0_100": 88,
                "active_label": True,
                "confidence_band": "Medium",
            },
            {
                "archetype_id": "POINTS_HAZARD",
                "family": "Risk Badge",
                "display_name": "Points Hazard",
                "score_0_100": 72,
                "active_label": True,
                "confidence_band": "Low",
            },
            {
                "archetype_id": "EXPLOSIVE",
                "family": "Return Shape",
                "display_name": "Explosive Returner",
                "score_0_100": 77,
                "active_label": True,
                "confidence_band": "Medium",
            },
        ]
    )

    result = player_page.active_archetype_tags(archetypes, overview=True)

    assert result["archetype_id"].tolist() == [
        "COMPLETE_FORWARD",
        "REGULAR_STARTER",
        "POINTS_HAZARD",
    ]
    markup = player_page.archetype_badge_markdown(result)
    assert ":violet-badge[Complete Forward]" in markup
    assert ":green-badge[Regular Starter]" in markup
    assert ":gray-badge[Points Hazard]" in markup


def test_overview_archetype_tags_fall_back_to_strongest_production_component() -> None:
    player_page = load_page_module("0_main.py")
    archetypes = pd.DataFrame(
        [
            {
                "archetype_id": "CREATOR",
                "family": "Production Style",
                "display_name": "Creator",
                "score_0_100": 71,
                "active_label": True,
            },
            {
                "archetype_id": "GOAL_THREAT",
                "family": "Production Style",
                "display_name": "Goal Threat",
                "score_0_100": 84,
                "active_label": True,
            },
            {
                "archetype_id": "ROTATION_RISK",
                "family": "Usage",
                "display_name": "Rotation Risk",
                "score_0_100": 62,
                "active_label": True,
            },
            {
                "archetype_id": "INACTIVE_RISK",
                "family": "Risk Badge",
                "display_name": "Inactive risk",
                "score_0_100": 99,
                "active_label": False,
            },
        ]
    )

    result = player_page.active_archetype_tags(archetypes, overview=True)

    assert result["archetype_id"].tolist() == ["GOAL_THREAT", "ROTATION_RISK"]


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
    assert alpha.iloc[-1] == 4


def test_fixture_schedule_keeps_facts_without_prediction_artifact(
    tmp_path: Path,
) -> None:
    player_page = load_page_module("0_main.py")
    metadata_path = tmp_path / "fixture_metadata_per_team_resolved.csv"
    fixtures_path = tmp_path / "fixtures.csv"
    pd.DataFrame(
        {
            "fpl_id": [1, 1],
            "team": [1, 7],
            "opp": [7, 1],
            "venue": ["home", "away"],
            "date_sched": ["2026-08-21", "2026-08-21"],
            "team_short": ["ARS", "COV"],
            "opp_short": ["COV", "ARS"],
            "opp_name": ["Coventry City", "Arsenal"],
        }
    ).to_csv(metadata_path, index=False)
    pd.DataFrame(
        {
            "id": [1],
            "event": [1],
            "kickoff_time": ["2026-08-21T19:00:00Z"],
            "finished": [False],
            "team_h_difficulty": [2],
            "team_a_difficulty": [5],
        }
    ).to_csv(fixtures_path, index=False)

    result = player_page.load_fixture_schedule(
        str(metadata_path), str(fixtures_path), None, None
    )

    arsenal = result.loc[result["team"].eq(1)].iloc[0]
    coventry = result.loc[result["team"].eq(7)].iloc[0]
    assert arsenal["opponent"] == "COV"
    assert arsenal["opponent_team_id"] == 7
    assert arsenal["is_home"]
    assert arsenal["fdr"] == 2
    assert coventry["fdr"] == 5
