from __future__ import annotations

import importlib.util
import inspect
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


def test_player_bio_is_independent_of_overview_content() -> None:
    player_page = load_page_module("0_main.py")
    bio_parameters = inspect.signature(player_page.render_player_bio).parameters
    overview_parameters = inspect.signature(
        player_page.render_overview_tab
    ).parameters

    assert "selected_season" in bio_parameters
    assert "archetypes" in bio_parameters
    assert "selected_season" in overview_parameters
    assert "archetypes" not in overview_parameters


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


def test_player_selector_resets_to_all_players_when_filters_change() -> None:
    player_page = load_page_module("0_main.py")
    options = [player_page.ALL_PLAYERS_OPTION, "p1", "p2"]

    assert player_page.player_selector_index(
        options,
        "p1",
        filters_changed=True,
        query_is_new=False,
    ) == 0
    assert player_page.player_selector_index(
        options,
        "p1",
        filters_changed=True,
        query_is_new=True,
    ) == 1
    assert player_page.player_selector_index(
        options,
        None,
        filters_changed=False,
        query_is_new=False,
    ) == 0


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


def test_per_90_history_tables_round_values_to_two_decimal_places() -> None:
    player_page = load_page_module("0_main.py")
    players = pd.DataFrame(
        {
            "season": ["2025-2026"],
            "player_id": ["a"],
            "name": ["Alpha"],
            "team": ["AAA"],
            "fpl_pos": ["MID"],
            "minutes": [901],
            "total_points": [101],
        }
    )

    history = player_page.build_player_history(players, "a", per_90=True)
    percentile_table = player_page.build_metric_percentile_table(history, per_90=True)

    expected = round(101 / 901 * 90, 2)
    assert history.iloc[0]["Points"] == expected
    assert percentile_table.loc[
        percentile_table["Metric"].eq("Points /90"), "Value"
    ].iloc[0] == expected
    display = player_page.format_per_90_table_values(percentile_table)
    assert display.loc[display["Metric"].eq("Points /90"), "Value"].iloc[0] == f"{expected:.2f}"


def test_player_stats_coalesce_gameweek_evidence_and_respect_provider_xg() -> None:
    player_page = load_page_module("0_main.py")
    players = pd.DataFrame(
        {
            "player_id": ["alex", "pedro"],
            "xg": [0.0, pd.NA],
            "xa": [pd.NA, pd.NA],
            "defcon": [10, pd.NA],
        }
    )
    gameweeks = pd.DataFrame(
        {
            "player_id": ["alex", "pedro"],
            "expected_goals": [0.01, 0.63],
            "expected_assists": [0.01, 0.06],
            "defensive_contribution": [11, 3],
        }
    )

    result = player_page.coalesce_gameweek_season_stats(players, gameweeks)

    assert result.loc[0, ["xg", "xa", "defcon"]].tolist() == [0.0, 0.01, 11]
    assert result.loc[1, ["xg", "xa", "defcon"]].tolist() == [0.63, 0.06, 3]


def test_metric_table_omits_unavailable_values_for_each_season() -> None:
    player_page = load_page_module("0_main.py")
    history = pd.DataFrame(
        {
            "Season": ["2026-2027", "2025-2026"],
            "Position": ["FWD", "FWD"],
            "xG": [1.2, 4.5],
            "Blocks": [pd.NA, 3],
        }
    )

    table = player_page.build_metric_percentile_table(history)

    current = table.loc[table["Season"].eq("2026-2027")]
    assert current["Metric"].tolist() == ["xG"]


def test_player_history_uses_position_specific_metric_columns() -> None:
    player_page = load_page_module("0_main.py")
    players = pd.DataFrame(
        {
            "season": ["2026-2027", "2026-2027"],
            "player_id": ["outfield", "keeper"],
            "name": ["Outfield", "Keeper"],
            "team": ["AAA", "BBB"],
            "fpl_pos": ["MID", "GKP"],
            "minutes": [900, 900],
            "total_points": [60, 55],
            "xg": [2.5, 0.0],
            "xa": [1.5, 0.0],
            "defcon": [100, 0],
            "blocks": [8, 0],
            "saves": [0, 45],
            "save_pct": [pd.NA, 75.0],
            "shots_on_target_against": [pd.NA, 60],
        }
    )

    outfield = player_page.build_player_history(players, "outfield")
    keeper = player_page.build_player_history(players, "keeper")

    assert {"xG", "xA", "Def Con", "Blocks"}.issubset(outfield.columns)
    assert not {"Saves", "Save %", "Shots on target against"}.intersection(
        outfield.columns
    )
    assert {"Saves", "Save %", "Shots on target against"}.issubset(keeper.columns)
    assert not {"xG", "xA", "Def Con", "Blocks"}.intersection(keeper.columns)


def test_hub_table_rows_supports_limits_and_all_rows() -> None:
    hub_page = load_page_module("0_home.py")
    frame = pd.DataFrame({"value": range(12)})

    assert len(hub_page._table_rows(frame, 5)) == 5
    assert len(hub_page._table_rows(frame, None)) == 12


def test_discovery_uses_previous_season_evidence_and_position_percentiles() -> None:
    player_page = load_page_module("0_main.py")
    current = pd.DataFrame(
        {
            "season": ["2026-2027", "2026-2027"],
            "player_id": ["a", "b"],
            "name": ["Alpha", "Bravo"],
            "team": ["AAA", "BBB"],
            "fpl_pos": ["MID", "MID"],
            "minutes": [0, 0],
            "total_points": [0, 0],
            "now_cost": [80, 65],
            "selected_by_percent": [12, 4],
            "status": ["a", "a"],
        }
    )
    previous = pd.DataFrame(
        {
            "season": ["2025-2026", "2025-2026"],
            "player_id": ["a", "b"],
            "name": ["Alpha", "Bravo"],
            "team": ["AAA", "BBB"],
            "fpl_pos": ["MID", "MID"],
            "minutes": [1800, 900],
            "total_points": [180, 60],
            "goals_scored": [12, 4],
            "assists": [8, 3],
            "xg": [11.5, 4.2],
            "xa": [7.3, 2.8],
            "defcon": [110, 70],
        }
    )
    players = pd.concat([current, previous], ignore_index=True, sort=False)
    gameweeks = pd.DataFrame(
        {
            "player_id": ["a", "a", "b"],
            "fpl_pos": ["MID", "MID", "MID"],
            "minutes": [90, 90, 90],
            "defensive_contribution": [12, 8, 13],
        }
    )

    evidence_season = player_page.discovery_evidence_season(
        players, "2026-2027", ["2026-2027", "2025-2026"]
    )
    result = player_page.build_player_discovery_table(
        current, players, evidence_season, gameweeks
    ).set_index("player_id")

    assert evidence_season == "2025-2026"
    assert result.loc["a", "Evidence season"] == "2025-2026"
    assert result.loc["a", "Points"] == 180
    assert result.loc["a", "Points Pctl"] == 100
    assert result.loc["a", "DefCon hit rate"] == 50
    assert result.loc["b", "DefCon hit rate Pctl"] == 100


def test_discovery_per_90_metrics_are_calculated_from_evidence_season() -> None:
    player_page = load_page_module("0_main.py")
    players = pd.DataFrame(
        {
            "season": ["2025-2026"],
            "player_id": ["a"],
            "name": ["Alpha"],
            "team": ["AAA"],
            "fpl_pos": ["MID"],
            "minutes": [901],
            "total_points": [101],
            "now_cost": [80],
            "selected_by_percent": [12],
            "status": ["a"],
        }
    )

    result = player_page.build_player_discovery_table(
        players, players, "2025-2026", pd.DataFrame(), per_90=True
    )

    assert result.iloc[0]["Points /90"] == 101 / 901 * 90
    assert result.iloc[0]["Points /90 Pctl"] == 100


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


def test_fixture_schedule_excludes_provisionally_finished_matches(
    tmp_path: Path,
) -> None:
    player_page = load_page_module("0_main.py")
    metadata_path = tmp_path / "fixture_metadata_per_team_resolved.csv"
    fixtures_path = tmp_path / "fixtures.csv"
    pd.DataFrame(
        {
            "fpl_id": [1, 1, 2, 2],
            "team": [1, 7, 1, 7],
            "opp": [7, 1, 7, 1],
            "venue": ["home", "away", "home", "away"],
            "date_sched": ["2026-08-21", "2026-08-21", "2026-08-28", "2026-08-28"],
            "team_short": ["ARS", "COV", "ARS", "COV"],
            "opp_short": ["COV", "ARS", "COV", "ARS"],
            "opp_name": ["Coventry City", "Arsenal", "Coventry City", "Arsenal"],
        }
    ).to_csv(metadata_path, index=False)
    pd.DataFrame(
        {
            "id": [1, 2],
            "event": [1, 2],
            "kickoff_time": ["2026-08-21T19:00:00Z", "2026-08-28T19:00:00Z"],
            "finished": [False, False],
            "finished_provisional": [True, False],
            "started": [True, False],
            "minutes": [90, 0],
            "team_h_difficulty": [2, 3],
            "team_a_difficulty": [5, 4],
        }
    ).to_csv(fixtures_path, index=False)

    result = player_page.load_fixture_schedule(
        str(metadata_path), str(fixtures_path), None, None
    )

    assert set(result["fpl_id"]) == {2}
    assert set(result["gw_orig"]) == {2}
