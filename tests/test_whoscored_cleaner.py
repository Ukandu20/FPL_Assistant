from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_assistant.providers.whoscored.clean.whoscored_cleaner import (
    _aggregate_event_flags,
    _authoritative_fpl_positions,
    _build_schedule_table,
    _build_team_match,
    _event_aggregates,
    _match_resolution,
    _normalize_events,
    _player_resolution,
    _pivot_stats,
    _resolve_player_positions,
    _season_table,
    _set_piece_roles,
    normalize_season,
    PLAYER_TABLES,
    ROLE_COLUMNS,
    TEAM_EXTRA_TABLES,
    TEAM_ONLY_TABLES,
)


IDENTITY_COLUMNS = [
    "entity_type",
    "provider",
    "provider_id",
    "provider_name",
    "canonical_id",
    "valid_from",
    "valid_to",
    "match_method",
    "match_confidence",
    "review_status",
]


def test_normalize_whoscored_split_season():
    assert normalize_season("2025") == ("2025-2026", "2025")
    assert normalize_season("2025-2026") == ("2025-2026", "2025")


def test_team_stat_pivot_preserves_text_metadata_without_suffix_columns():
    raw = pd.DataFrame(
        [
            {"game_id": 1, "team_id": 2, "team": "ARS", "stat_key": "ratings", "value": "7.1", "value_text": "7.1"},
            {"game_id": 1, "team_id": 2, "team": "ARS", "stat_key": "manager_name", "value": "Manager", "value_text": "Manager"},
            {"game_id": 1, "team_id": 2, "team": "ARS", "stat_key": "country_name", "value": "England", "value_text": "England"},
        ]
    )

    result, _ = _pivot_stats(raw, keys=["game_id", "team_id", "team"])

    assert result.loc[0, "manager_name"] == "Manager"
    assert result.loc[0, "country_name"] == "England"
    assert not any(column.endswith(("_x", "_y")) for column in result)


def test_event_metric_schema_covers_all_direct_metric_families():
    required = {
        "passing": {
            "big_chances_created", "shot_creating_actions", "goal_creating_actions",
            "crosses_completed", "cross_completion_pct", "box_entries_by_pass",
        },
        "passing_types": {
            "long_balls_completed", "head_passes_completed",
            "through_balls_completed", "layoffs_completed",
            "chipped_passes_completed",
        },
        "shooting": {
            "big_chance_shots", "big_chances_scored", "big_chances_missed",
            "assisted_shots", "first_touch_shots", "one_on_one_shots",
            "fast_break_shots", "shots_from_corner", "shots_from_set_piece",
            "direct_free_kick_shots",
        },
        "defense": {
            "errors_leading_to_attempt", "errors_leading_to_goal",
            "last_man_actions", "offsides_provoked",
            "possessions_won_attacking_third", "defensive_actions_penalty_area",
        },
        "possession": {
            "possessions_won_attacking_third",
            "possessions_lost_defensive_third",
            "shielding_actions", "overruns", "good_skills",
        },
        "keepers": {
            "diving_saves", "standing_saves", "saves_penalty_area",
            "saves_outside_box", "keeper_throws", "goal_kicks",
        },
    }

    for table, metrics in required.items():
        assert metrics <= set(PLAYER_TABLES[table])
    assert {
        "big_chances_created", "big_chance_shots", "shot_creating_actions",
        "goal_creating_actions", "big_chances_conceded",
    } <= set(TEAM_EXTRA_TABLES["goal_shot_creation"])
    assert "big_chances_conceded" in TEAM_ONLY_TABLES["defense"]


def test_player_resolution_uses_registry_alias_and_team_season_surname():
    players = pd.DataFrame(
        [
            {
                "provider_player_id": 1,
                "provider_player_name": "Known Player",
                "provider_team_id": 13,
            },
            {
                "provider_player_id": 2,
                "provider_player_name": "Tino Livramento",
                "provider_team_id": 23,
            },
            {
                "provider_player_id": 3,
                "provider_player_name": "New Academy Player",
                "provider_team_id": 13,
            },
            {
                "provider_player_id": 4,
                "provider_player_name": "Altay Bayindir",
                "provider_team_id": 1,
            },
            {
                "provider_player_id": 5,
                "provider_player_name": "Andy Robertson",
                "provider_team_id": 2,
            },
        ]
    )
    master = {
        "p1": {
            "name": "Known Player",
            "career": {"2025-2026": {"team_id": "ars"}},
        },
        "p2": {
            "name": "Valentino Livramento",
            "career": {"2025-2026": {"team_id": "new"}},
        },
        "p4": {
            "name": "Altay Bayındır",
            "career": {"2025-2026": {"team_id": "mun"}},
        },
        "p5": {
            "name": "Andrew Robertson",
            "career": {"2025-2026": {"team_id": "liv"}},
        },
    }

    audit, bridges = _player_resolution(
        players,
        player_lookup={
            "known player": "p1",
            "valentino livramento": "p2",
            "altay bayındır": "p4",
            "andrew robertson": "p5",
        },
        master_players=master,
        player_aliases={},
        team_map={"13": "ars", "23": "new", "1": "mun", "2": "liv"},
        season="2025-2026",
        existing=pd.DataFrame(columns=IDENTITY_COLUMNS),
    )

    resolved = dict(zip(audit["provider_id"], audit["canonical_id"]))
    assert resolved["1"] == "p1"
    assert resolved["2"] == "p2"
    assert pd.isna(resolved["3"])
    assert resolved["4"] == "p4"
    assert resolved["5"] == "p5"
    assert set(bridges["provider_id"]) == {"1", "2", "4", "5"}


def test_player_resolution_includes_new_official_fpl_players():
    players = pd.DataFrame(
        [{
            "provider_player_id": 520919,
            "provider_player_name": "Alysson Edward",
            "provider_team_id": 24,
        }]
    )
    official = pd.DataFrame(
        [{
            "player_id": "new-fpl-id",
            "name": "Alysson Edward Franco da Rocha dos Santos",
            "first_name": "Alysson Edward",
            "second_name": "Franco da Rocha dos Santos",
            "web_name": "Alysson",
            "team_id": "avl",
        }]
    )

    audit, _ = _player_resolution(
        players,
        player_lookup={"alysson": "different-player"},
        master_players={},
        official_fpl_players=official,
        player_aliases={},
        team_map={"24": "avl"},
        season="2025-2026",
        existing=pd.DataFrame(columns=IDENTITY_COLUMNS),
    )

    assert audit.loc[0, "canonical_id"] == "new-fpl-id"
    assert audit.loc[0, "match_method"] == "exact_normalized_name"


def test_player_resolution_uses_builtin_given_name_alias():
    players = pd.DataFrame(
        [{
            "provider_player_id": 473957,
            "provider_player_name": "Joe Johnson",
            "provider_team_id": 95,
        }]
    )
    master = {
        "b1e71b49": {
            "name": "Joseph Johnson",
            "career": {"2023-2024": {"team_id": "lut"}},
        }
    }

    audit, _ = _player_resolution(
        players,
        player_lookup={"joseph johnson": "b1e71b49"},
        master_players=master,
        player_aliases={},
        team_map={"95": "lut"},
        season="2023-2024",
        existing=pd.DataFrame(columns=IDENTITY_COLUMNS),
    )

    assert audit.loc[0, "canonical_id"] == "b1e71b49"
    assert audit.loc[0, "match_method"] == "configured_alias"


def test_player_resolution_distinguishes_emerson_palmieri_and_royal():
    players = pd.DataFrame(
        [
            {
                "provider_player_id": 101955,
                "provider_player_name": "Emerson",
                "provider_team_id": 29,
            },
            {
                "provider_player_id": 328512,
                "provider_player_name": "Emerson Royal",
                "provider_team_id": 30,
            },
        ]
    )

    audit, _ = _player_resolution(
        players,
        player_lookup={
            "emerson": "royal-id",
            "emerson palmieri": "palmieri-id",
            "emerson royal": "royal-id",
        },
        master_players={},
        player_aliases={},
        team_map={"29": "whu", "30": "tot"},
        season="2023-2024",
        existing=pd.DataFrame(columns=IDENTITY_COLUMNS),
    )

    resolved = dict(zip(audit["provider_id"], audit["canonical_id"]))
    assert resolved == {"101955": "palmieri-id", "328512": "royal-id"}


def test_player_resolution_prefers_current_fpl_id_over_stale_registry_id():
    players = pd.DataFrame(
        [{
            "provider_player_id": 520919,
            "provider_player_name": "Alysson Edward",
            "provider_team_id": 24,
        }]
    )
    official = pd.DataFrame(
        [{
            "player_id": "generated-fpl-id",
            "name": "Alysson Edward Franco da Rocha dos Santos",
            "first_name": "Alysson Edward Franco",
            "second_name": "da Rocha dos Santos",
            "web_name": "Alysson",
            "team_id": "avl",
        }]
    )
    master = {
        "registry-id": {
            "name": "Alysson",
            "career": {"2025-2026": {"team_id": "avl"}},
        }
    }

    audit, _ = _player_resolution(
        players,
        player_lookup={"alysson": "registry-id"},
        master_players=master,
        official_fpl_players=official,
        player_aliases={},
        team_map={"24": "avl"},
        season="2025-2026",
        existing=pd.DataFrame(columns=IDENTITY_COLUMNS),
    )

    assert audit.loc[0, "canonical_id"] == "generated-fpl-id"


def test_player_resolution_accepts_empty_preseason_player_set():
    audit, bridges = _player_resolution(
        pd.DataFrame(
            columns=[
                "provider_player_id",
                "provider_player_name",
                "provider_team_id",
            ]
        ),
        player_lookup={},
        master_players={},
        official_fpl_players=pd.DataFrame(),
        player_aliases={},
        team_map={},
        season="2026-2027",
        existing=pd.DataFrame(columns=IDENTITY_COLUMNS),
    )

    assert audit.empty
    assert list(bridges.columns) == IDENTITY_COLUMNS


def test_match_resolution_handles_rescheduled_unique_team_pair():
    fixture_path = Path(".tmp") / "test_whoscored_fixture_calendar.csv"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "fbref_id": "fb-match",
                "fpl_id": 42,
                "date_played": "2026-03-21",
                "home_id": "wol",
                "away_id": "ars",
                "gw_played": 31,
            }
        ]
    ).to_csv(fixture_path, index=False)
    schedule = pd.DataFrame(
        [
            {
                "league": "ENG-Premier League",
                "game_id": 1903469,
                "game": "2026-02-18 Wolves-Arsenal",
                "date": "2026-02-18",
                "start_time": "2026-02-18T20:00:00Z",
                "home_team_id": 161,
                "home_team": "Wolves",
                "away_team_id": 13,
                "away_team": "Arsenal",
            }
        ]
    )
    existing = pd.DataFrame(
        columns=[
            "provider",
            "provider_match_id",
            "match_id",
            "provider_game",
            "match_method",
            "match_confidence",
        ]
    )

    _, audit, bridges = _match_resolution(
        schedule,
        season="2025-2026",
        team_map={"161": "wol", "13": "ars"},
        fixture_path=fixture_path,
        existing=existing,
    )

    assert audit.loc[0, "match_id"] == "fb-match"
    assert audit.loc[0, "match_method"] == "registry_fixture_team_pair"
    assert audit.loc[0, "match_confidence"] == 0.98
    assert bridges.loc[0, "provider_match_id"] == "1903469"


def test_event_aggregation_keeps_action_families_separate():
    events = pd.DataFrame(
        [
            ["m1", "p1", "t1", "", "Tackle", True, "[]", 40, 50, 45, 50, True, ""],
            ["m1", "p1", "t1", "", "Pass", True, "[]", 40, 50, 75, 50, True, ""],
            ["m1", "p1", "t1", "", "Goal", True, "[]", 90, 50, None, None, True, ""],
            ["m1", "p2", "t2", "p1", "Foul", False, "[]", 50, 50, None, None, False, ""],
        ],
        columns=[
            "provider_match_id",
            "provider_player_id",
            "provider_team_id",
            "provider_related_player_id",
            "type",
            "is_successful",
            "qualifiers",
            "x",
            "y",
            "end_x",
            "end_y",
            "is_touch",
            "card_type",
        ],
    )

    aggregated, _ = _event_aggregates(events)
    player = aggregated[aggregated["provider_player_id"].eq("p1")].iloc[0]

    assert player["tackles"] == 1
    assert player["passes_attempted"] == 1
    assert player["progressive_passes"] == 1
    assert player["goals"] == 1
    assert player["fouls_drawn"] == 1


def test_event_flag_aggregation_batches_preserve_grouped_sums():
    work = pd.DataFrame(
        {
            "provider_match_id": ["m1", "m1", "m1", "m2"],
            "provider_player_id": ["p1", "p1", "p2", "p1"],
            "provider_team_id": ["t1", "t1", "t1", "t2"],
        }
    )
    flags = {
        f"metric_{index}": pd.Series(
            [True, index % 2 == 0, index % 3 == 0, False]
        )
        for index in range(7)
    }

    result = _aggregate_event_flags(
        work,
        keys=["provider_match_id", "provider_player_id", "provider_team_id"],
        flags=flags,
        batch_size=2,
    ).set_index(["provider_match_id", "provider_player_id", "provider_team_id"])

    for index in range(7):
        assert result.loc[("m1", "p1", "t1"), f"metric_{index}"] == (
            2 if index % 2 == 0 else 1
        )
        assert result.loc[("m1", "p2", "t1"), f"metric_{index}"] == int(index % 3 == 0)


def test_event_aggregation_accepts_empty_preseason_event_set():
    events = pd.DataFrame(
        columns=[
            "provider_match_id",
            "provider_player_id",
            "provider_team_id",
            "provider_related_player_id",
            "type",
            "is_successful",
            "qualifiers",
            "x",
            "y",
            "end_x",
            "end_y",
            "is_touch",
            "card_type",
        ]
    )

    aggregated, normalized = _event_aggregates(events)

    assert aggregated.empty
    assert normalized.empty
    assert "fouls_drawn" in aggregated


def test_event_aggregation_publishes_direct_chance_context_and_spatial_metrics():
    def qualifiers(*names: str) -> str:
        return str(
            [
                {"type": {"value": index, "displayName": name}}
                for index, name in enumerate(names, start=1)
            ]
        )

    def event(
        player_id: str,
        event_type: str,
        *,
        successful: bool = True,
        related_player_id: str = "",
        qualifier_names: tuple[str, ...] = (),
        x: float = 50,
        y: float = 50,
        end_x: float | None = None,
        end_y: float | None = None,
    ) -> dict:
        return {
            "provider_match_id": "m1",
            "provider_player_id": player_id,
            "provider_team_id": "t1",
            "provider_related_player_id": related_player_id,
            "type": event_type,
            "is_successful": successful,
            "qualifiers": qualifiers(*qualifier_names),
            "x": x,
            "y": y,
            "end_x": end_x,
            "end_y": end_y,
            "is_touch": True,
            "card_type": "",
        }

    events = pd.DataFrame(
        [
            event(
                "creator",
                "Pass",
                qualifier_names=(
                    "KeyPass", "BigChanceCreated", "Cross", "Longball",
                    "HeadPass", "Throughball", "LayOff", "Chipped",
                ),
                x=60,
                y=50,
                end_x=90,
                end_y=50,
            ),
            event(
                "creator",
                "BallTouch",
                qualifier_names=("KeyPass", "BigChanceCreated"),
                x=85,
            ),
            event(
                "shooter",
                "Goal",
                related_player_id="creator",
                qualifier_names=(
                    "Assisted", "BigChance", "FirstTouch", "OneOnOne", "FastBreak",
                ),
                x=90,
            ),
            event(
                "shooter",
                "MissedShots",
                related_player_id="creator",
                qualifier_names=(
                    "Assisted", "BigChance", "FromCorner", "SetPiece",
                    "DirectFreekick",
                ),
                x=88,
            ),
            event("creator", "Error", qualifier_names=("LeadingToAttempt",), x=20),
            event("creator", "Error", qualifier_names=("LeadingToGoal",), x=20),
            event("creator", "Tackle", qualifier_names=("LastMan",), x=70),
            event("creator", "Interception", qualifier_names=("LastMan",), x=10),
            event("creator", "Save", qualifier_names=("LastMan",), x=5),
            event("creator", "BallRecovery", x=75),
            event("creator", "OffsideProvoked", x=25),
            event("creator", "CornerAwarded", x=92),
            event("creator", "Clearance", qualifier_names=("BlockedCross",), x=10),
            event("creator", "Pass", successful=False, x=20, end_x=40, end_y=50),
        ]
    )

    aggregated, _ = _event_aggregates(events)
    creator = aggregated.set_index("provider_player_id").loc["creator"]
    shooter = aggregated.set_index("provider_player_id").loc["shooter"]

    assert creator["key_passes"] == 2
    assert creator["big_chances_created"] == 2
    assert creator["crosses_completed"] == 1
    assert creator["long_balls_completed"] == 1
    assert creator["head_passes_completed"] == 1
    assert creator["through_balls_completed"] == 1
    assert creator["layoffs_completed"] == 1
    assert creator["chipped_passes_completed"] == 1
    assert creator["box_entries_by_pass"] == 1
    assert creator["errors_leading_to_attempt"] == 1
    assert creator["errors_leading_to_goal"] == 1
    assert creator["last_man_actions"] == 3
    assert creator["last_man_tackles"] == 1
    assert creator["last_man_interceptions"] == 1
    assert creator["last_man_saves"] == 1
    assert creator["offsides_provoked"] == 1
    assert creator["corners_won"] == 1
    assert creator["crosses_blocked"] == 1
    assert creator["possessions_won_attacking_third"] == 2
    assert creator["possessions_lost_defensive_third"] == 3
    assert creator["defensive_actions_penalty_area"] == 3
    assert creator["shot_creating_actions"] == 2
    assert creator["goal_creating_actions"] == 1

    assert shooter["big_chance_shots"] == 2
    assert shooter["big_chances_scored"] == 1
    assert shooter["big_chances_missed"] == 1
    assert shooter["assisted_shots"] == 2
    assert shooter["first_touch_shots"] == 1
    assert shooter["one_on_one_shots"] == 1
    assert shooter["fast_break_shots"] == 1
    assert shooter["fast_break_goals"] == 1
    assert shooter["open_play_shots"] == 1
    assert shooter["shots_from_corner"] == 1
    assert shooter["shots_from_set_piece"] == 1
    assert shooter["direct_free_kick_shots"] == 1


def test_event_aggregation_infers_carries_and_box_entries_from_event_sequence():
    events = pd.DataFrame(
        [
            {
                "provider_match_id": "m1", "provider_player_id": "passer",
                "provider_team_id": "t1", "provider_related_player_id": "",
                "type": "Pass", "is_successful": True, "qualifiers": "[]",
                "x": 60, "y": 50, "end_x": 75, "end_y": 50,
                "expanded_minute": 10, "second": 0, "is_touch": True, "card_type": "",
            },
            {
                "provider_match_id": "m1", "provider_player_id": "receiver",
                "provider_team_id": "t1", "provider_related_player_id": "",
                "type": "Pass", "is_successful": True, "qualifiers": "[]",
                "x": 86, "y": 50, "end_x": 90, "end_y": 50,
                "expanded_minute": 10, "second": 5, "is_touch": True, "card_type": "",
            },
        ]
    )

    aggregated, _ = _event_aggregates(events)
    receiver = aggregated.set_index("provider_player_id").loc["receiver"]

    assert receiver["carries"] == 1
    assert receiver["progressive_carries"] == 1
    assert receiver["box_entries_by_carry"] == 1
    assert receiver["box_entries"] == 1


def test_event_aggregation_publishes_remaining_direct_provider_signals():
    def event(player: str, kind: str, *qualifiers: str, successful: bool = True) -> dict:
        return {
            "provider_match_id": "m1", "provider_player_id": player,
            "provider_team_id": "t1", "provider_related_player_id": "",
            "type": kind, "is_successful": successful,
            "qualifiers": str([
                {"type": {"value": i, "displayName": name}}
                for i, name in enumerate(qualifiers, 1)
            ]),
            "x": 50, "y": 50, "end_x": 70, "end_y": 50,
            "expanded_minute": 1, "second": 0, "is_touch": True, "card_type": "",
        }

    events = pd.DataFrame(
        [
            event("victim", "Foul", "Penalty"),
            event("offender", "Foul", "Penalty", "AerialFoul", successful=False),
            event("passer", "Pass", "ThrowIn", "ShotAssist", "IntentionalAssist"),
            event("corner", "Pass", "CornerTaken"),
            event("keeper", "Save", "DivingSave", "KeeperSaveInTheBox"),
            event("keeper", "Pass", "KeeperThrow"),
            event("shooter", "Goal", "LeftFoot", "Volley", "IndividualPlay"),
            event("skill", "ShieldBallOpp", "OverRun"),
            event("skill", "GoodSkill"),
            event("", "FormationChange"),
        ]
    )

    result, _ = _event_aggregates(events)
    by_player = result.set_index("provider_player_id")

    assert by_player.loc["victim", "penalties_won"] == 1
    assert by_player.loc["offender", "penalties_conceded"] == 1
    assert by_player.loc["offender", "aerial_fouls"] == 1
    assert by_player.loc["passer", "throw_ins_completed"] == 1
    assert by_player.loc["passer", "shot_assists"] == 1
    assert by_player.loc["passer", "intentional_assists"] == 1
    assert by_player.loc["corner", "corners_completed"] == 1
    assert by_player.loc["keeper", "diving_saves"] == 1
    assert by_player.loc["keeper", "saves_penalty_area"] == 1
    assert by_player.loc["keeper", "keeper_throws"] == 1
    assert by_player.loc["shooter", "left_foot_shots"] == 1
    assert by_player.loc["shooter", "left_foot_goals"] == 1
    assert by_player.loc["shooter", "volleys"] == 1
    assert by_player.loc["skill", "shielding_actions"] == 1
    assert by_player.loc["skill", "good_skills"] == 1
    assert by_player.loc["", "formation_changes"] == 1


def test_season_rates_are_recomputed_from_summed_event_counts():
    frame = pd.DataFrame(
        [
            {
                "league": "ENG-Premier League", "season": "2025-2026",
                "provider_season": "2025", "match_id": "m1", "team_id": "t1",
                "crosses": 2, "crosses_completed": 1,
                "throw_ins": 4, "throw_ins_completed": 3,
                "corners": 2, "corners_completed": 1,
                "big_chance_shots": 1, "big_chances_scored": 1,
                "clean_sheets": 1,
            },
            {
                "league": "ENG-Premier League", "season": "2025-2026",
                "provider_season": "2025", "match_id": "m2", "team_id": "t1",
                "crosses": 8, "crosses_completed": 2,
                "throw_ins": 6, "throw_ins_completed": 5,
                "corners": 3, "corners_completed": 1,
                "big_chance_shots": 3, "big_chances_scored": 0,
                "clean_sheets": 0,
            },
        ]
    )

    result = _season_table(
        frame,
        player=False,
        metrics=[
            "crosses", "crosses_completed", "cross_completion_pct",
            "throw_ins", "throw_ins_completed", "throw_in_completion_pct",
            "corners", "corners_completed", "corner_completion_pct",
            "big_chance_shots", "big_chances_scored", "big_chance_conversion_pct",
            "clean_sheets",
        ],
    )

    assert result.loc[0, "cross_completion_pct"] == 30.0
    assert result.loc[0, "throw_in_completion_pct"] == 80.0
    assert result.loc[0, "corner_completion_pct"] == 40.0
    assert result.loc[0, "big_chance_conversion_pct"] == 25.0
    assert result.loc[0, "clean_sheets"] == 1


def test_team_metrics_include_events_without_player_attribution():
    player_match = pd.DataFrame(
        [
            {
                "provider_match_id": "m1", "provider_team_id": "13",
                "corners_won": 1, "shots_on_target": 0,
            },
            {
                "provider_match_id": "m1", "provider_team_id": "14",
                "corners_won": 0, "shots_on_target": 0,
            },
        ]
    )
    event_metrics = pd.DataFrame(
        [
            {
                "provider_match_id": "m1", "provider_team_id": "13",
                "provider_player_id": "p1", "corners_won": 1,
                "shots_on_target": 0,
            },
            {
                "provider_match_id": "m1", "provider_team_id": "13",
                "provider_player_id": "", "corners_won": 1,
                "shots_on_target": 0,
            },
            {
                "provider_match_id": "m1", "provider_team_id": "14",
                "provider_player_id": "p2", "corners_won": 0,
                "shots_on_target": 0,
            },
        ]
    )
    context = pd.DataFrame(
        [
            {
                "provider_match_id": "m1", "game": "ARS-CHE",
                "game_date": "2025-08-01", "kickoff_utc": "2025-08-01T12:00:00Z",
                "gameweek": 1, "status": "complete", "home_team_id": "ars",
                "away_team_id": "che", "home": "ARS", "away": "CHE",
                "home_score": 0, "away_score": 0, "score": "0-0",
                "league": "ENG-Premier League",
            }
        ]
    )
    team_stats = pd.DataFrame(columns=["game_id", "team_id", "team"])

    result = _build_team_match(
        team_stats,
        player_match,
        event_metrics=event_metrics,
        context=context,
        match_map={"m1": "canonical-m1"},
        team_map={"13": "ars", "14": "che"},
        team_names={"ars": "ARS", "che": "CHE"},
        season="2025-2026",
        provider_season="2025",
    )

    arsenal = result[result["team_id"].eq("ars")].iloc[0]
    assert arsenal["corners_won"] == 2


def test_team_big_chances_conceded_comes_from_opponent_shots():
    player_match = pd.DataFrame(
        [
            {
                "provider_match_id": "m1", "provider_team_id": "13",
                "shots_total": 10, "shots_on_target": 4, "shots_off_target": 3,
                "shots_blocked": 3, "shots_on_post": 1, "shots_box": 7,
                "shots_outside_box": 3, "headed_shots": 2, "open_play_shots": 8,
                "shots_from_set_piece": 2, "shots_from_corner": 1,
                "direct_free_kick_shots": 1, "penalty_attempts": 0,
                "big_chance_shots": 2, "box_entries_by_pass": 9,
                "box_entries_by_carry": 4, "box_entries": 13,
            },
            {
                "provider_match_id": "m1", "provider_team_id": "14",
                "shots_total": 16, "shots_on_target": 7, "shots_off_target": 5,
                "shots_blocked": 4, "shots_on_post": 0, "shots_box": 12,
                "shots_outside_box": 4, "headed_shots": 5, "open_play_shots": 11,
                "shots_from_set_piece": 5, "shots_from_corner": 3,
                "direct_free_kick_shots": 1, "penalty_attempts": 1,
                "big_chance_shots": 5, "box_entries_by_pass": 14,
                "box_entries_by_carry": 6, "box_entries": 20,
            },
        ]
    )
    context = pd.DataFrame(
        [{
            "provider_match_id": "m1", "game": "ARS-CHE", "game_date": "2025-08-01",
            "kickoff_utc": "2025-08-01T12:00:00Z", "gameweek": 1, "status": "complete",
            "home_team_id": "ars", "away_team_id": "che", "home": "ARS", "away": "CHE",
            "home_score": 1, "away_score": 0, "score": "1-0", "league": "ENG-Premier League",
        }]
    )

    result = _build_team_match(
        pd.DataFrame(columns=["game_id", "team_id", "team"]), player_match,
        context=context, match_map={"m1": "canonical-m1"},
        team_map={"13": "ars", "14": "che"}, team_names={"ars": "ARS", "che": "CHE"},
        season="2025-2026", provider_season="2025",
    ).set_index("team_id")

    assert result.loc["ars", "big_chances_conceded"] == 5
    assert result.loc["che", "big_chances_conceded"] == 2
    assert result.loc["ars", "shots_against"] == 16
    assert result.loc["ars", "shots_conceded"] == 16
    assert result.loc["ars", "shots_on_target_against"] == 7
    assert result.loc["ars", "shots_off_target_against"] == 5
    assert result.loc["ars", "shots_blocked_against"] == 4
    assert result.loc["ars", "shots_blocked_defensively"] == 4
    assert result.loc["ars", "shots_box_against"] == 12
    assert result.loc["ars", "shots_outside_box_against"] == 4
    assert result.loc["ars", "headed_shots_against"] == 5
    assert result.loc["ars", "open_play_shots_against"] == 11
    assert result.loc["ars", "shots_from_set_piece_against"] == 5
    assert result.loc["ars", "penalty_attempts_against"] == 1
    assert result.loc["ars", "box_entries_by_pass_against"] == 14
    assert result.loc["ars", "box_entries_by_carry_against"] == 6
    assert result.loc["ars", "box_entries_against"] == 20
    assert result.loc["ars", "box_entries_allowed"] == 20
    assert result.loc["ars", "box_entries_conceded"] == 20
    assert result.loc["ars", "box_entries_by_pass_allowed"] == 14
    assert result.loc["ars", "box_entries_by_carry_allowed"] == 6
    assert result.loc["che", "shots_against"] == 10
    assert result.loc["che", "box_entries_allowed"] == 13
    assert result.loc["ars", "clean_sheets"] == 1
    assert result.loc["che", "clean_sheets"] == 0


def test_schedule_clean_sheets_require_completed_scores():
    context = pd.DataFrame(
        [
            {
                "provider_match_id": "m1", "league": "ENG-Premier League",
                "game": "ARS-CHE", "status": 6, "home_team_id": "ars",
                "away_team_id": "che", "provider_home_team_id": "13",
                "provider_away_team_id": "14", "home_team": "Arsenal",
                "away_team": "Chelsea", "home_score": 0, "away_score": 0,
            },
            {
                "provider_match_id": "m2", "league": "ENG-Premier League",
                "game": "ARS-CHE", "status": "complete", "home_team_id": "ars",
                "away_team_id": "che", "provider_home_team_id": "13",
                "provider_away_team_id": "14", "home_team": "Arsenal",
                "away_team": "Chelsea", "home_score": 2, "away_score": 0,
            },
            {
                "provider_match_id": "m3", "league": "ENG-Premier League",
                "game": "ARS-CHE", "status": 1, "home_team_id": "ars",
                "away_team_id": "che", "provider_home_team_id": "13",
                "provider_away_team_id": "14", "home_team": "Arsenal",
                "away_team": "Chelsea", "home_score": pd.NA, "away_score": pd.NA,
            },
        ]
    )

    result = _build_schedule_table(
        context,
        match_map={"m1": "cm1", "m2": "cm2", "m3": "cm3"},
        team_names={"ars": "Arsenal", "che": "Chelsea"},
        season="2025-2026",
        provider_season="2025",
    ).set_index(["provider_match_id", "team_id"])

    assert result.loc[("m1", "ars"), "clean_sheets"] == 1
    assert result.loc[("m1", "che"), "clean_sheets"] == 1
    assert result.loc[("m2", "ars"), "clean_sheets"] == 1
    assert result.loc[("m2", "che"), "clean_sheets"] == 0
    assert pd.isna(result.loc[("m3", "ars"), "clean_sheets"])
    assert pd.isna(result.loc[("m3", "che"), "clean_sheets"])


def test_set_piece_roles_rank_observed_takers_and_keep_corner_side_separate():
    def event(
        event_id: str,
        match_id: str,
        player_id: str,
        player: str,
        role_qualifier: str,
        *,
        event_type: str = "Pass",
        game_date: str = "2025-08-10",
        x: float = 80,
        y: float = 80,
        end_x: float = 90,
        end_y: float = 50,
    ) -> dict:
        return {
            "event_id": event_id,
            "match_id": match_id,
            "provider_match_id": match_id,
            "team_id": "team-1",
            "team": "ARS",
            "provider_team_id": "13",
            "player_id": player_id,
            "player": player,
            "provider_player_id": player_id,
            "type": event_type,
            "qualifiers": (
                "[{'type': {'value': 1, 'displayName': '"
                + role_qualifier
                + "'}}]"
            ),
            "x": x,
            "y": y,
            "end_x": end_x,
            "end_y": end_y,
            "game_date": game_date,
            "league": "ENG-Premier League",
            "season": "2025-2026",
            "provider_season": "2025",
            "coverage_status": "complete",
        }

    events = pd.DataFrame(
        [
            event("c1", "m1", "p1", "Primary", "CornerTaken", y=90),
            event("c2", "m1", "p1", "Primary", "CornerTaken", y=90),
            event("c3", "m2", "p2", "Secondary", "CornerTaken", y=90),
            event("c4", "m2", "p2", "Secondary", "CornerTaken", y=10),
            event("p1", "m1", "p1", "Primary", "Penalty", event_type="Goal"),
            event("p2", "m2", "p1", "Primary", "Penalty", event_type="SavedShot"),
            event("p3", "m2", "p2", "Secondary", "Penalty", event_type="Goal"),
            event("d1", "m2", "p1", "Primary", "DirectFreekick", event_type="MissedShots"),
            event("f1", "m2", "p1", "Primary", "FreekickTaken"),
            event("i1", "m2", "p2", "Secondary", "IndirectFreekickTaken"),
            event("l1", "m2", "p2", "Secondary", "ThrowIn", x=60, end_x=85, y=10, end_y=10),
        ]
    )
    appearances = pd.DataFrame(
        [
            {"match_id": "m1", "team_id": "team-1", "player_id": "p1", "minutes": 90},
            {"match_id": "m1", "team_id": "team-1", "player_id": "p2", "minutes": 0},
            {"match_id": "m2", "team_id": "team-1", "player_id": "p1", "minutes": 90},
            {"match_id": "m2", "team_id": "team-1", "player_id": "p2", "minutes": 90},
        ]
    )

    roles = _set_piece_roles(events, appearances)

    assert set(roles["role"]) == {
        "corner",
        "penalty",
        "direct_free_kick",
        "free_kick",
        "indirect_free_kick",
        "long_throw",
    }
    corners = roles[roles["role"].eq("corner")]
    assert set(corners["side"]) == {"left", "right"}
    assert corners["role"].eq("corner").all()
    left = corners[corners["side"].eq("left")].set_index("player_id")
    assert left.loc["p1", "role_rank"] == "primary"
    assert left.loc["p2", "role_rank"] == "secondary"
    penalties = roles[roles["role"].eq("penalty")].set_index("player_id")
    assert penalties.loc["p1", "attempts"] == 2
    assert penalties.loc["p1", "role_rank"] == "primary"
    assert roles["opportunity_policy"].eq(
        "team_events_in_matches_with_minutes"
    ).all()
    assert roles["as_of"].eq("2025-08-10").all()


def test_set_piece_roles_are_empty_when_season_has_no_events():
    roles = _set_piece_roles(pd.DataFrame())

    assert roles.empty
    assert list(roles.columns) == ROLE_COLUMNS


def test_normalized_events_get_stable_unique_event_ids():
    events = pd.DataFrame(
        [
            {
                "game_id": 10,
                "id": 99,
                "event_id": 1,
                "type": "Pass",
                "period": "FirstHalf",
                "minute": 1,
                "second": 2,
                "team_id": 13,
                "team": "Arsenal",
                "player_id": 7,
                "player": "Player One",
                "related_player_id": None,
                "outcome_type": "Successful",
                "qualifiers": "[]",
            },
            {
                "game_id": 10,
                "id": 99,
                "event_id": 2,
                "type": "OffsideGiven",
                "period": "FirstHalf",
                "minute": 2,
                "second": None,
                "team_id": 14,
                "team": "Chelsea",
                "player_id": 8,
                "player": "Player Two",
                "related_player_id": None,
                "outcome_type": "Successful",
                "qualifiers": "[]",
            },
        ]
    )
    context = pd.DataFrame(
        [
            {
                "provider_match_id": "10",
                "game_date": "2025-08-01",
                "kickoff_utc": "2025-08-01T12:00:00Z",
                "gameweek": 1,
                "home_team_id": "ars",
                "away_team_id": "che",
            }
        ]
    )

    normalized = _normalize_events(
        events,
        match_map={"10": "m1"},
        team_map={"13": "ars", "14": "che"},
        player_map={"7": "p1", "8": "p2"},
        player_names={"p1": "Player One", "p2": "Player Two"},
        team_names={"ars": "ARS", "che": "CHE"},
        context=context,
        season="2025-2026",
        provider_season="2025",
    )

    assert normalized["event_id"].is_unique
    assert normalized["provider_event_id"].is_unique
    assert set(normalized["match_id"]) == {"m1"}
    assert set(normalized["player_identity_status"]) == {"resolved"}


def test_player_season_has_one_row_per_registry_player():
    frame = pd.DataFrame(
        [
            {
                "league": "ENG-Premier League",
                "season": "2025-2026",
                "provider_season": "2025",
                "match_id": "m1",
                "player_id": "p1",
                "player": "Player One",
                "nation": "ENG",
                "born": 2000,
                "position": "CB",
                "fpl_pos": "DEF",
                "minutes": 90,
                "tackles": 2,
            },
            {
                "league": "ENG-Premier League",
                "season": "2025-2026",
                "provider_season": "2025",
                "match_id": "m2",
                "player_id": "p1",
                "player": "Player One",
                "nation": "ENG",
                "born": 2000,
                "position": "DM",
                "fpl_pos": "MID",
                "minutes": 30,
                "tackles": 1,
            },
        ]
    )

    season = _season_table(frame, player=True, metrics=["tackles"])

    assert len(season) == 1
    assert season.loc[0, "matches_played"] == 2
    assert season.loc[0, "minutes"] == 120
    assert season.loc[0, "tackles"] == 3


def test_positions_preserve_match_role_and_use_minutes_for_primary_position():
    roster = pd.DataFrame(
        [
            {
                "player_id": "p1", "position": "MC", "minutes": 20,
                "is_first_eleven": True, "kickoff_utc": "2025-08-01T12:00:00Z",
            },
            {
                "player_id": "p1", "position": "MC", "minutes": 20,
                "is_first_eleven": True, "kickoff_utc": "2025-08-08T12:00:00Z",
            },
            {
                "player_id": "p1", "position": "AMC", "minutes": 90,
                "is_first_eleven": True, "kickoff_utc": "2025-08-15T12:00:00Z",
            },
        ]
    )

    result = _resolve_player_positions(roster, master_players={}, season="2025-2026")

    assert result["provider_position_match"].tolist() == ["MC", "MC", "AMC"]
    assert result["is_starter"].tolist() == [1, 1, 1]
    assert result["position_detail_match"].tolist() == ["CM", "CM", "AM"]
    assert set(result["primary_position"]) == {"AM"}
    assert set(result["fpl_pos"]) == {"MID"}


def test_substitute_keeps_unobserved_match_role_but_receives_determined_positions():
    roster = pd.DataFrame(
        [
            {
                "player_id": "p1", "position": "DC", "minutes": 90,
                "is_first_eleven": True, "kickoff_utc": "2025-08-01T12:00:00Z",
            },
            {
                "player_id": "p1", "position": "Sub", "minutes": 25,
                "is_first_eleven": False, "kickoff_utc": "2025-08-08T12:00:00Z",
            },
            {
                "player_id": "p2", "position": "Sub", "minutes": 12,
                "is_first_eleven": False, "kickoff_utc": "2025-08-08T12:00:00Z",
            },
            {
                "player_id": "p3", "position": "Sub", "minutes": 0,
                "is_first_eleven": False, "kickoff_utc": "2025-08-08T12:00:00Z",
            },
        ]
    )
    master = {
        "p1": {"career": {"2025-26": {"fpl_position": "DEF", "position_detail": "CB"}}},
        "p2": {"career": {"2025-26": {"fpl_pos": "MID", "position_detail": "DM"}}},
        "p3": {"career": {"2024-2025": {"fpl_position": "GK", "position": "GK"}}},
    }

    result = _resolve_player_positions(roster, master_players=master, season="2025-2026")
    p1_sub = result[(result["player_id"] == "p1") & (result["provider_position_match"] == "Sub")].iloc[0]
    p2_sub = result[result["player_id"] == "p2"].iloc[0]

    assert p1_sub["position_detail_match"] == "UNK"
    assert p1_sub["is_starter"] == 0
    assert p1_sub["position"] == "UNK"
    assert p1_sub["primary_position"] == "CB"
    assert p1_sub["position_detail_match_imputed"] == "CB"
    assert p1_sub["position_imputation_source"] == "whoscored.season_primary"
    assert p1_sub["fpl_pos"] == "DEF"
    assert p1_sub["fpl_position_source"] == "registry.fpl_position"

    assert p2_sub["position_detail_match"] == "UNK"
    assert p2_sub["primary_position"] == "UNK"
    assert p2_sub["position_detail_match_imputed"] == "DM"
    assert p2_sub["position_imputation_source"] == "registry.season_position"
    assert p2_sub["fpl_pos"] == "MID"
    assert p2_sub["is_starter"] == 0

    p3_sub = result[result["player_id"] == "p3"].iloc[0]
    assert p3_sub["fpl_pos"] == "GKP"
    assert p3_sub["fpl_position_source"] == "registry.latest_prior.fpl_position"
    assert p3_sub["fpl_position_confidence"] == 0.7


def test_wingback_codes_are_kept_and_classified_as_defenders():
    roster = pd.DataFrame(
        [{"player_id": "p1", "position": "DMR", "minutes": 60, "is_first_eleven": True}]
    )

    result = _resolve_player_positions(roster, master_players={}, season="2025-2026")

    assert result.loc[0, "provider_position_match"] == "DMR"
    assert result.loc[0, "position_detail_match"] == "RWB"
    assert result.loc[0, "fpl_pos"] == "DEF"


def test_official_fpl_position_overrides_registry_and_tactical_role():
    roster = pd.DataFrame(
        [{"player_id": "p1", "position": "FW", "minutes": 90, "is_first_eleven": True}]
    )
    master_players = {
        "p1": {"career": {"2025-2026": {"fpl_position": "FWD", "position": "FW"}}}
    }
    authority = _authoritative_fpl_positions(
        pd.DataFrame([{"player_id": "p1", "fpl_pos": "MID"}]),
        {},
        "2025-2026",
    )

    result = _resolve_player_positions(
        roster,
        master_players=master_players,
        season="2025-2026",
        authoritative_fpl=authority,
    )

    assert result.loc[0, "primary_position"] == "FW"
    assert result.loc[0, "fpl_pos"] == "MID"
    assert result.loc[0, "fpl_position_source"] == "fpl.cleaned_players.fpl_pos"
    assert result.loc[0, "fpl_position_confidence"] == 1.0


def test_master_fpl_fills_absent_official_player_before_provider_registry():
    roster = pd.DataFrame(
        [{"player_id": "p1", "position": "AMC", "minutes": 30, "is_first_eleven": True}]
    )
    master_players = {
        "p1": {"career": {"2025-2026": {"fpl_position": "MID", "position": "MID"}}}
    }
    master_fpl = {
        "p1": {"career": {"2025-26": {"fpl_pos": "FWD"}}}
    }
    authority = _authoritative_fpl_positions(
        pd.DataFrame(),
        master_fpl,
        "2025-2026",
    )

    result = _resolve_player_positions(
        roster,
        master_players=master_players,
        season="2025-2026",
        authoritative_fpl=authority,
    )

    assert result.loc[0, "fpl_pos"] == "FWD"
    assert result.loc[0, "fpl_position_source"] == "fpl.master_fpl.fpl_pos"
    assert result.loc[0, "fpl_position_confidence"] == 0.99
