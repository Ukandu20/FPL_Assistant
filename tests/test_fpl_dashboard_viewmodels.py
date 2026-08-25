from __future__ import annotations

import json

import pandas as pd

from fpl_assistant.apps.viewmodels.dashboard import (
    active_archetype_tags,
    archetype_changes,
    archetype_reason,
    comparison_table,
    enrich_current_players,
    fixture_rows,
    forecast_watchlist,
    player_watchlist,
    gameweek_deadline,
    stat_leaders,
    team_fixture_outlook,
    upcoming_fixture_rows,
)


def _archetypes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "p1", "archetype_id": "COMPLETE_FORWARD",
                "display_name": "Complete Forward", "family": "Production Composite",
                "score_0_100": 82, "active_label": True, "confidence_band": "High",
                "component_scores": '{"goal_threat": 88, "creator": 74}',
            },
            {
                "player_id": "p1", "archetype_id": "GOAL_THREAT",
                "display_name": "Goal Threat", "family": "Production Style",
                "score_0_100": 88, "active_label": True, "confidence_band": "High",
            },
            {
                "player_id": "p1", "archetype_id": "STARTER",
                "display_name": "Regular Starter", "family": "Usage",
                "score_0_100": 90, "active_label": True, "confidence_band": "Medium",
            },
            {
                "player_id": "p2", "archetype_id": "ROTATION",
                "display_name": "Rotation Risk", "family": "Usage",
                "score_0_100": 66, "active_label": True, "confidence_band": "Low",
            },
        ]
    )


def test_active_tags_and_reason_are_decision_ready() -> None:
    tags = active_archetype_tags(_archetypes().loc[lambda frame: frame.player_id.eq("p1")], overview=True)

    assert tags["archetype_id"].tolist() == ["COMPLETE_FORWARD", "STARTER"]
    assert archetype_reason(tags.iloc[0]) == "Goal Threat: 88.0 · Creator: 74.0"


def test_active_tags_retain_the_same_archetype_for_every_player() -> None:
    season = pd.concat(
        [
            _archetypes(),
            pd.DataFrame(
                [
                    {
                        "player_id": "p2",
                        "archetype_id": "COMPLETE_FORWARD",
                        "display_name": "Complete Forward",
                        "family": "Production Composite",
                        "score_0_100": 76,
                        "active_label": True,
                        "confidence_band": "Medium",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    tags = active_archetype_tags(season)
    complete_forwards = tags.loc[tags["archetype_id"].eq("COMPLETE_FORWARD")]

    assert complete_forwards["player_id"].tolist() == ["p1", "p2"]


def test_archetype_reason_ignores_nested_calculation_metadata() -> None:
    row = {
        "component_scores": json.dumps(
            {
                "goal_threat": 88.0,
                "window_weights": {"recent": 0.75, "previous_season": 0.25},
                "required": ["GOAL_THREAT"],
                "creator": "74",
            }
        )
    }

    assert archetype_reason(row) == "Goal Threat: 88.0 · Creator: 74.0"


def test_usage_reason_uses_historical_labels_and_denominators() -> None:
    row = {
        "family": "Usage",
        "component_scores": json.dumps(
            {
                "expected_minutes": 76.075,
                "start_probability": 0.875,
                "cameo_probability": 0.2525,
            }
        ),
    }

    assert archetype_reason(row) == (
        "Avg min / available team match: 76.1 · "
        "Historical start rate: 87.5% · "
        "Historical cameo rate when benched: 25.2%"
    )


def test_current_roster_is_enriched_with_baseline_and_unique_family_columns() -> None:
    current = pd.DataFrame(
        {
            "player_id": ["p1", "p2"], "name": ["Alpha", "Bravo"],
            "team": ["AAA", "BBB"], "fpl_pos": ["FWD", "FWD"],
            "minutes": [0, 0], "total_points": [0, 0], "now_cost": [80, 60],
        }
    )
    previous = pd.DataFrame(
        {
            "player_id": ["p1", "p2"], "total_points": [180, 90],
            "minutes": [2700, 1400], "goals_scored": [20, 8], "assists": [8, 3],
        }
    )

    result = enrich_current_players(current, previous, _archetypes())

    assert result.columns.is_unique
    assert result.loc[result.player_id.eq("p1"), "baseline_total_points"].iloc[0] == 180
    assert result.loc[result.player_id.eq("p1"), "Production profile"].iloc[0] == "Complete Forward"
    assert result.loc[result.player_id.eq("p2"), "Usage"].iloc[0] == "Rotation Risk"


def test_watchlists_distinguish_historical_and_forecast_evidence() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["p1", "p2"], "name": ["Alpha", "Bravo"],
            "team": ["AAA", "BBB"], "fpl_pos": ["FWD", "FWD"],
            "status": ["a", "a"], "minutes": [0, 0], "now_cost": [80, 60],
            "selected_by_percent": [10, 5], "baseline_total_points": [180, 120],
            "baseline_minutes": [2700, 1800],
        }
    )
    baseline = player_watchlist(players)
    assert baseline.iloc[0]["Player" if "Player" in baseline else "name"] == "Alpha"
    assert baseline["Evidence"].eq("Previous-season baseline").all()

    forecasts = pd.DataFrame(
        {
            "player_id": ["p1", "p1", "p2", "p2"], "gw_orig": [1, 2, 1, 2],
            "xPts": [4, 5, 6, 6], "pred_minutes": [80, 80, 90, 90],
        }
    )
    projected = forecast_watchlist(players, forecasts)
    assert projected.iloc[0]["player_id"] == "p2"
    assert projected.iloc[0]["Forecast xPts"] == 12


def test_fixture_outlook_expands_both_teams_and_ranks_easier_runs() -> None:
    fixtures = pd.DataFrame(
        {
            "id": [1, 2], "event": [1, 2], "finished": [False, False],
            "kickoff_time": ["2026-08-20T18:00:00Z", "2026-08-27T18:00:00Z"],
            "team_h": [1, 2], "team_a": [2, 1],
            "team_h_difficulty": [2, 4], "team_a_difficulty": [4, 2],
        }
    )
    teams = pd.DataFrame({"id": [1, 2], "name": ["Alpha", "Bravo"], "short_name": ["AAA", "BBB"]})

    rows = fixture_rows(fixtures, teams)
    outlook = team_fixture_outlook(rows)

    assert len(rows) == 4
    assert outlook.loc[outlook.Team.eq("AAA"), "Average FDR"].iloc[0] == 2
    assert outlook.loc[outlook.Team.eq("BBB"), "Average FDR"].iloc[0] == 4


def test_provisionally_finished_fixture_is_not_upcoming() -> None:
    fixtures = pd.DataFrame(
        {
            "id": [1, 2],
            "event": [1, 2],
            "finished": [False, False],
            "finished_provisional": [True, False],
            "started": [True, False],
            "minutes": [90, 0],
            "kickoff_time": ["2026-08-21T19:00:00Z", "2026-08-28T19:00:00Z"],
            "team_h": [1, 1],
            "team_a": [2, 2],
            "team_h_difficulty": [2, 3],
            "team_a_difficulty": [4, 3],
        }
    )
    teams = pd.DataFrame(
        {"id": [1, 2], "name": ["Alpha", "Bravo"], "short_name": ["AAA", "BBB"]}
    )

    rows = fixture_rows(fixtures, teams)
    upcoming = upcoming_fixture_rows(rows)

    assert rows.loc[rows["GW"].eq(1), "finished"].all()
    assert set(upcoming["GW"]) == {2}


def test_deadline_prefers_official_event_and_labels_fixture_fallback() -> None:
    fixtures = pd.DataFrame(
        {"event": [1, 1], "kickoff_time": ["2026-08-21T19:00:00Z", "2026-08-22T11:30:00Z"]}
    )
    events = pd.DataFrame({"id": [1], "deadline_time": ["2026-08-21T17:30:00Z"]})

    official, source = gameweek_deadline(events, fixtures, 1)
    assert official == pd.Timestamp("2026-08-21T17:30:00Z")
    assert source == "official"

    estimated, source = gameweek_deadline(pd.DataFrame(), fixtures, 1)
    assert estimated == pd.Timestamp("2026-08-21T17:30:00Z")
    assert source == "estimated"


def test_stat_leaders_use_match_rows_for_rates_value_and_defcon_hits() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["def", "mid", "gkp"],
            "name": ["Defender", "Midfielder", "Keeper"],
            "team": ["AAA", "BBB", "CCC"],
            "fpl_pos": ["DEF", "MID", "GKP"],
            "now_cost": [50, 100, 45],
        }
    )
    gameweeks = pd.DataFrame(
        {
            "player_id": ["def", "def", "mid", "mid", "gkp"],
            "fpl_pos": ["DEF", "DEF", "MID", "MID", "GKP"],
            "round": [1, 2, 1, 1, 1],
            "kickoff_time": [
                "2026-08-21T19:00:00Z", "2026-08-29T14:00:00Z",
                "2026-08-22T11:30:00Z", "2026-08-25T19:00:00Z",
                "2026-08-22T14:00:00Z",
            ],
            "minutes": [90, 45, 90, 30, 90],
            "total_points": [6, 2, 8, 1, 4],
            "goals_scored": [0, 0, 1, 0, 0],
            "assists": [0, 1, 0, 0, 0],
            "defensive_contribution": [10, 9, 12, 11, 15],
        }
    )

    value = stat_leaders(players, gameweeks, metric="Points / £m")
    assert value.iloc[0]["Player"] == "Defender"
    assert value.iloc[0]["Value"] == 1.6

    per_match = stat_leaders(players, gameweeks, metric="Points", basis="Per appearance")
    assert per_match.iloc[0]["Player"] == "Midfielder"
    assert per_match.iloc[0]["Value"] == 4.5

    hit_rate = stat_leaders(players, gameweeks, metric="DefCon hit rate")
    assert hit_rate.set_index("Player").loc["Defender", "Value"] == 0.5
    assert hit_rate.set_index("Player").loc["Midfielder", "Value"] == 0.5
    assert "Keeper" not in hit_rate["Player"].tolist()

    unavailable = stat_leaders(
        players, gameweeks.drop(columns="defensive_contribution"), metric="DefCon hit rate"
    )
    assert unavailable.empty


def test_archetype_changes_and_comparison_table_preserve_context() -> None:
    current = _archetypes()
    previous = current.loc[~current.archetype_id.eq("COMPLETE_FORWARD")].copy()
    players = pd.DataFrame({"player_id": ["p1", "p2"], "name": ["Alpha", "Bravo"], "team": ["AAA", "BBB"]})
    changes = archetype_changes(current, previous, players)
    assert changes[["Change", "Archetype", "name"]].to_dict("records") == [
        {"Change": "Activated", "Archetype": "Complete Forward", "name": "Alpha"}
    ]

    enriched = pd.DataFrame(
        {
            "name": ["Alpha"], "team": ["AAA"], "fpl_pos": ["FWD"],
            "now_cost": [80], "minutes": [0], "selected_by_percent": [10],
            "baseline_total_points": [180], "baseline_minutes": [2700],
            "baseline_goals_scored": [20], "baseline_assists": [8],
            "baseline_xg": [18.2], "baseline_xa": [7.4],
            "Production profile": ["Complete Forward"], "Usage": ["Regular Starter"],
        }
    )
    comparison = comparison_table(enriched)
    assert comparison.iloc[0]["Evidence"] == "Previous season"
    assert comparison.iloc[0]["Points"] == 180
