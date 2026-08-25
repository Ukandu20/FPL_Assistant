from __future__ import annotations

import json

import pandas as pd
import pytest

from fpl_assistant.providers.fpl.pipelines.clean_and_enrich import (
    GOALKEEPER_SEASON_STAT_COLUMNS,
    PLAYER_SEASON_STAT_COLUMNS,
    backfill_published_season_stats,
    enrich_player_season_stats,
)
from fpl_assistant.testing.paths import get_test_run_dir


LEAGUE = "ENG-Premier League"
TEST_ROOT = get_test_run_dir("player_season_stats_enrichment")


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _case_dir(name):
    path = TEST_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_historical_fbref_mapping_preserves_rows_and_transfer_team_grain():
    tmp_path = _case_dir("historical_mapping")
    fbref_root = tmp_path / "fbref"
    season = "2024-2025"
    _write_csv(
        fbref_root / LEAGUE / season / "player_season" / "defense.csv",
        [
            {"player_id": "p1", "team": "ARS", "blocks": 2, "int": 3, "clr": 4, "tklw": 2},
            {"player_id": "p1", "team": "CHE", "blocks": 1, "int": 1, "clr": 1, "tklw": 1},
        ],
    )
    _write_csv(
        fbref_root / LEAGUE / season / "player_season" / "misc.csv",
        [
            {"player_id": "p1", "team": "ARS", "recov": 20},
            {"player_id": "p1", "team": "CHE", "recov": 3},
        ],
    )
    _write_csv(
        fbref_root / LEAGUE / season / "player_season" / "standard.csv",
        [
            {"player_id": "p1", "team": "ARS", "xg": 1.2, "xag": 2.3},
            {"player_id": "p1", "team": "CHE", "xg": 0.4, "xag": 0.5},
        ],
    )
    players = pd.DataFrame(
        [
            {"player_id": "p1", "name": "Player", "team": "ARS", "minutes": 900},
            {"player_id": "p1", "name": "Player", "team": "CHE", "minutes": 90},
            {"player_id": "bench", "name": "Bench", "team": "ARS", "minutes": 0},
        ]
    )

    result, audit = enrich_player_season_stats(
        players, season, LEAGUE, fbref_root=fbref_root
    )

    assert len(result) == len(players)
    assert audit["row_count_preserved"] is True
    assert result.loc[0, ["blocks", "interceptions", "clearances", "tackles_won", "recoveries", "defcon"]].tolist() == [2, 3, 4, 2, 20, 11]
    assert result.loc[1, ["blocks", "interceptions", "clearances", "tackles_won", "recoveries", "defcon"]].tolist() == [1, 1, 1, 1, 3, 4]
    assert result.loc[0, "xa"] == pytest.approx(2.3)
    assert result.loc[2, PLAYER_SEASON_STAT_COLUMNS].eq(0).all()
    assert result.loc[2, "defensive_stats_source"] == "derived.zero_minutes"


def test_modern_sources_leave_unmatched_active_players_null():
    tmp_path = _case_dir("modern_mapping")
    whoscored_root = tmp_path / "whoscored"
    understat_root = tmp_path / "understat"
    season = "2025-2026"
    _write_csv(
        whoscored_root / LEAGUE / season / "player_season" / "defense.csv",
        [{"player_id": "p1", "blocks": 5, "interceptions": 4, "clearances": 3, "tackles_won": 2, "recoveries": 6}],
    )
    _write_csv(
        understat_root / LEAGUE / season / "player_season.csv",
        [{"player_id": "p1", "xg": 6.5, "xa": 2.5}],
    )
    players = pd.DataFrame(
        [
            {"player_id": "p1", "name": "Matched", "team": "ARS", "fpl_pos": "MID", "minutes": 900},
            {"player_id": "active", "name": "Unmatched", "team": "CHE", "fpl_pos": "DEF", "minutes": 90},
            {"player_id": "bench", "name": "Bench", "team": "LIV", "fpl_pos": "FWD", "minutes": 0},
        ]
    )

    result, audit = enrich_player_season_stats(
        players,
        season,
        LEAGUE,
        whoscored_root=whoscored_root,
        understat_root=understat_root,
    )

    assert result.loc[0, "defcon"] == 20
    assert result.loc[0, ["xg", "xa"]].tolist() == [6.5, 2.5]
    assert result.loc[1, PLAYER_SEASON_STAT_COLUMNS].isna().all()
    assert result.loc[1, "stats_coverage_status"] == "unmatched_active"
    assert result.loc[2, PLAYER_SEASON_STAT_COLUMNS].eq(0).all()
    assert audit["incomplete_active_count"] == 1


def test_modern_sources_use_official_fpl_metric_fallbacks():
    tmp_path = _case_dir("modern_official_fallback")
    whoscored_root = tmp_path / "whoscored"
    understat_root = tmp_path / "understat"
    season = "2026-2027"
    _write_csv(
        whoscored_root / LEAGUE / season / "player_season" / "defense.csv",
        [
            {
                "player_id": "alex",
                "blocks": 1,
                "interceptions": 1,
                "clearances": 0,
                "tackles_won": 2,
                "recoveries": 6,
            }
        ],
    )
    _write_csv(
        understat_root / LEAGUE / season / "player_season.csv",
        [{"player_id": "alex", "xg": 0.0, "xa": None}],
    )
    players = pd.DataFrame(
        [
            {
                "player_id": "alex",
                "name": "Alex",
                "team": "BOU",
                "fpl_pos": "MID",
                "minutes": 90,
                "expected_goals": 0.01,
                "expected_assists": 0.01,
                "defensive_contribution": 11,
            },
            {
                "player_id": "pedro",
                "name": "Pedro",
                "team": "CHE",
                "fpl_pos": "FWD",
                "minutes": 90,
                "expected_goals": 0.63,
                "expected_assists": 0.06,
                "defensive_contribution": 3,
            },
        ]
    )

    result, _ = enrich_player_season_stats(
        players,
        season,
        LEAGUE,
        whoscored_root=whoscored_root,
        understat_root=understat_root,
    )

    assert result.loc[0, ["xg", "xa", "defcon"]].tolist() == [0.0, 0.01, 11]
    assert result.loc[1, ["xg", "xa", "defcon"]].tolist() == [0.63, 0.06, 3]
    assert result.loc[1, "expected_stats_source"] == "fpl.official.expected_metrics"
    assert result.loc[1, "defensive_stats_source"] == (
        "fpl.official.defensive_contribution"
    )


def test_future_season_without_provider_files_remains_unavailable():
    tmp_path = _case_dir("future_missing")
    players = pd.DataFrame(
        [{"player_id": "p1", "name": "Future", "team": "ARS", "minutes": 0}]
    )

    result, audit = enrich_player_season_stats(
        players,
        "2026-2027",
        LEAGUE,
        whoscored_root=tmp_path / "whoscored",
        understat_root=tmp_path / "understat",
    )

    assert result.loc[0, PLAYER_SEASON_STAT_COLUMNS].isna().all()
    assert result.loc[0, "stats_coverage_status"] == "provider_data_unavailable"
    assert audit["source_available"] == {
        "defensive": False,
        "expected": False,
        "goalkeeper": False,
    }


def test_historical_fbref_goalkeeper_fields_and_penalties_are_normalized():
    tmp_path = _case_dir("historical_goalkeeper")
    fbref_root = tmp_path / "fbref"
    season = "2024-2025"
    _write_csv(
        fbref_root / LEAGUE / season / "player_season" / "keeper.csv",
        [
            {
                "player_id": "gk1",
                "team": "ARS",
                "sota": 120,
                "saves": 86,
                "ga": 34,
                "save": 74.2,
                "pkatt": 3,
                "pka": 3,
                "pksv": 0,
                "pkm": 0,
                "save_save": 0.0,
            }
        ],
    )
    players = pd.DataFrame(
        [
            {
                "player_id": "gk1",
                "name": "Keeper",
                "team": "ARS",
                "fpl_pos": "GKP",
                "minutes": 3420,
            },
            {
                "player_id": "outfield",
                "name": "Outfield",
                "team": "ARS",
                "fpl_pos": "DEF",
                "minutes": 1000,
            },
        ]
    )

    result, audit = enrich_player_season_stats(
        players, season, LEAGUE, fbref_root=fbref_root
    )

    keeper = result.iloc[0]
    assert keeper["shots_on_target_against"] == 120
    assert keeper["saves"] == 86
    assert keeper["goals_against"] == 34
    assert keeper["save_pct"] == pytest.approx(74.2)
    assert keeper[
        [
            "penalties_faced",
            "penalties_allowed",
            "penalties_saved",
            "penalties_missed",
            "penalty_save_pct",
        ]
    ].tolist() == [3, 3, 0, 0, 0]
    assert keeper["goalkeeper_stats_coverage"] == "complete"
    assert result.loc[1, GOALKEEPER_SEASON_STAT_COLUMNS].isna().all()
    assert result.loc[1, "goalkeeper_stats_coverage"] == "not_applicable"
    assert audit["matched_rows"]["goalkeeper"] == 1


def test_modern_goalkeepers_are_aggregated_from_actual_appearances():
    tmp_path = _case_dir("modern_goalkeeper")
    whoscored_root = tmp_path / "whoscored"
    season = "2025-2026"
    keeper_path = (
        whoscored_root / LEAGUE / season / "player_match" / "keepers.csv"
    )
    _write_csv(
        keeper_path,
        [
            {
                "provider_match_id": "m1",
                "team_id": "t1",
                "player_id": "gk1",
                "fpl_pos": "GKP",
                "minutes": 90,
                "shots_on_target_against": 5,
                "goals_against": 2,
                "saves": 3,
                "penalties_faced": 1,
            },
            {
                "provider_match_id": "m2",
                "team_id": "t1",
                "player_id": "gk1",
                "fpl_pos": "GKP",
                "minutes": 90,
                "shots_on_target_against": 4,
                "goals_against": 1,
                "saves": 3,
                "penalties_faced": 0,
            },
            {
                "provider_match_id": "m1",
                "team_id": "t1",
                "player_id": "reserve",
                "fpl_pos": "GKP",
                "minutes": 0,
                "shots_on_target_against": 5,
                "goals_against": 2,
                "saves": 0,
                "penalties_faced": 0,
            },
            {
                "provider_match_id": "m1",
                "team_id": "t1",
                "player_id": "forward",
                "fpl_pos": "FWD",
                "minutes": 90,
                "shots_on_target_against": 5,
                "goals_against": 2,
                "saves": 0,
                "penalties_faced": 0,
            },
        ],
    )
    players = pd.DataFrame(
        [
            {"player_id": "gk1", "name": "Keeper", "team": "ARS", "fpl_pos": "GKP", "minutes": 180},
            {"player_id": "reserve", "name": "Reserve", "team": "ARS", "fpl_pos": "GKP", "minutes": 0},
            {"player_id": "forward", "name": "Forward", "team": "ARS", "fpl_pos": "FWD", "minutes": 180},
        ]
    )

    result, audit = enrich_player_season_stats(
        players, season, LEAGUE, whoscored_root=whoscored_root
    )

    keeper = result.iloc[0]
    assert keeper["shots_on_target_against"] == 9
    assert keeper["goals_against"] == 3
    assert keeper["saves"] == 6
    assert keeper["save_pct"] == pytest.approx(100 * 6 / 9)
    assert keeper["penalties_faced"] == 1
    assert keeper["goalkeeper_stats_coverage"] == "complete"
    assert result.loc[1, "shots_on_target_against"] == 0
    assert pd.isna(result.loc[1, "save_pct"])
    assert result.loc[1, "goalkeeper_stats_coverage"] == "zero_minutes"
    assert result.loc[2, GOALKEEPER_SEASON_STAT_COLUMNS].isna().all()
    assert audit["shared_goalkeeper_appearance_rows"] == 0


def test_duplicate_provider_keys_fail_instead_of_multiplying_rows():
    tmp_path = _case_dir("duplicate_provider")
    fbref_root = tmp_path / "fbref"
    season = "2024-2025"
    duplicate = {
        "player_id": "p1",
        "team": "ARS",
        "blocks": 2,
        "int": 3,
        "clr": 4,
        "tklw": 1,
    }
    _write_csv(
        fbref_root / LEAGUE / season / "player_season" / "defense.csv",
        [duplicate, duplicate],
    )
    _write_csv(
        fbref_root / LEAGUE / season / "player_season" / "standard.csv",
        [{"player_id": "p1", "team": "ARS", "xg": 1, "xag": 1}],
    )

    with pytest.raises(ValueError, match="duplicate join keys"):
        enrich_player_season_stats(
            pd.DataFrame(
                [{"player_id": "p1", "name": "Player", "team": "ARS", "minutes": 90}]
            ),
            season,
            LEAGUE,
            fbref_root=fbref_root,
        )


def test_unresolved_provider_keys_do_not_block_valid_player_matches():
    tmp_path = _case_dir("unresolved_provider_keys")
    understat_root = tmp_path / "understat"
    season = "2026-2027"
    _write_csv(
        understat_root / LEAGUE / season / "player_season.csv",
        [
            {"player_id": "p1", "xg": 1.2, "xa": 0.4},
            {"player_id": None, "xg": 0.0, "xa": 0.0},
            {"player_id": None, "xg": 0.0, "xa": 0.0},
        ],
    )

    result, audit = enrich_player_season_stats(
        pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "name": "Player",
                    "team": "ARS",
                    "fpl_pos": "MID",
                    "minutes": 90,
                }
            ]
        ),
        season,
        LEAGUE,
        understat_root=understat_root,
    )

    assert result.loc[0, ["xg", "xa"]].tolist() == [1.2, 0.4]
    assert audit["matched_rows"]["expected"] == 1


def test_stats_only_backfill_writes_audit_and_preserves_roster_count():
    tmp_path = _case_dir("backfill")
    season = "2026-2027"
    season_dir = tmp_path / "fpl" / LEAGUE / season
    roster_path = season_dir / "season" / "cleaned_players.csv"
    _write_csv(
        roster_path,
        [
            {"player_id": "p1", "name": "One", "team": "ARS", "minutes": 0},
            {"player_id": "p2", "name": "Two", "team": "CHE", "minutes": 0},
        ],
    )

    audit = backfill_published_season_stats(
        season_dir,
        LEAGUE,
        whoscored_root=tmp_path / "whoscored",
        understat_root=tmp_path / "understat",
    )

    result = pd.read_csv(roster_path)
    second_audit = backfill_published_season_stats(
        season_dir,
        LEAGUE,
        whoscored_root=tmp_path / "whoscored",
        understat_root=tmp_path / "understat",
    )
    repeated = pd.read_csv(roster_path)
    audit_path = season_dir / "_manual_review" / f"player_season_stat_enrichment_{season}.json"
    stored_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(result) == 2
    pd.testing.assert_frame_equal(result, repeated)
    assert set(PLAYER_SEASON_STAT_COLUMNS).issubset(result.columns)
    assert audit["row_count_preserved"] is True
    assert second_audit["row_count_preserved"] is True
    assert stored_audit["row_count_before"] == stored_audit["row_count_after"] == 2
