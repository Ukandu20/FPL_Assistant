from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_assistant.providers.whoscored.clean.whoscored_cleaner import (
    _authoritative_fpl_positions,
    _event_aggregates,
    _match_resolution,
    _normalize_events,
    _player_resolution,
    _resolve_player_positions,
    _season_table,
    normalize_season,
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
