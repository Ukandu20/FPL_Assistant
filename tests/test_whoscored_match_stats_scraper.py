from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from fpl_assistant.providers.whoscored import whoscored_match_stats_scraper as ws
from fpl_assistant.providers.whoscored import whoscored_scraper as ws_legacy
from fpl_assistant.providers.whoscored import (
    CompetitionConfig,
    extract_match_centre_payload,
    parse_calendar_mask,
    parse_embedded_tournament_fixtures,
    parse_missing_players_html,
    parse_schedule_month_payload,
    parse_season_options,
)
from fpl_assistant.testing.paths import get_test_soccerdata_dir


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "whoscored"
os.environ.setdefault("SOCCERDATA_DIR", str(get_test_soccerdata_dir().resolve()))


def _sample_payload() -> dict:
    return json.loads((FIXTURE_DIR / "sample_match_raw.json").read_text(encoding="utf-8"))


def _schedule_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "league": "ENG-Premier League",
                "season": 2025,
                "game": "2025-08-15 Home FC-Away FC",
                "game_id": 123456,
                "date": "2025-08-15",
                "start_time": "2025-08-15T19:00:00Z",
                "status": "FT",
                "stage": "Regular Season",
                "home_team_id": 10,
                "home_team": "Home FC",
                "away_team_id": 20,
                "away_team": "Away FC",
                "home_score": 2,
                "away_score": 1,
                "incidents": [
                    {
                        "minute": 11,
                        "type": {"displayName": "Goal"},
                        "subType": {"displayName": "OpenPlay"},
                        "teamId": 10,
                        "playerId": 101,
                        "playerName": "Home Starter",
                        "text": "Schedule incident"
                    }
                ],
            }
        ]
    )


def test_to_ws_season_int_formats():
    assert ws._to_ws_season_int("2025-2026") == 2025
    assert ws._to_ws_season_int("2024-25") == 2024
    assert ws._to_ws_season_int("25-26") == 2025
    assert ws._to_ws_season_int("2025") == 2025


def test_resolve_competitions_supports_explicit_and_groups():
    leagues = ws.resolve_competitions(
        explicit_leagues=["ENG-Premier League"],
        competition_groups=["major-international"],
        all_known_leagues=False,
        all_configured_competitions=False,
    )
    assert leagues[0] == "ENG-Premier League"
    assert "INT-World Cup" in leagues
    assert "INT-Copa America" in leagues


def test_resolve_competitions_rejects_unknown_group():
    with pytest.raises(ValueError):
        ws.resolve_competitions(
            explicit_leagues=None,
            competition_groups=["not-a-group"],
            all_known_leagues=False,
            all_configured_competitions=False,
        )


def test_normalize_events_from_raw_payload_fixture():
    payload = _sample_payload()
    schedule_row = _schedule_df().iloc[0].to_dict()

    df = ws._normalize_events_from_payload(
        payload=payload,
        schedule_row=schedule_row,
        league="ENG-Premier League",
        season_int=2025,
        game_id=123456,
    )

    assert {"game", "game_id", "type", "team", "player", "qualifiers"}.issubset(df.columns)
    assert df.shape[0] == 3
    assert set(df["type"].dropna()) >= {"FormationSet", "Start", "Pass"}
    pass_row = df.loc[df["type"] == "Pass"].iloc[0]
    assert pass_row["team"] == "Home FC"
    assert pass_row["player"] == "Home Starter"


def test_build_derived_tables_from_raw_payload_fixture():
    payloads = {123456: _sample_payload()}
    derived = ws.build_derived_tables(
        payloads=payloads,
        schedule_df=_schedule_df(),
        league="ENG-Premier League",
        season_int=2025,
        derived_tables=ws.DERIVED_TABLE_CHOICES,
    )

    assert derived["match_info"].iloc[0]["venue"] == "Example Stadium"
    assert derived["match_info"].iloc[0]["referee"] == "Jane Ref"
    assert derived["incidents"].shape[0] == 2
    assert set(derived["incidents"]["source"]) == {"raw", "schedule"}
    assert derived["player_dictionary"]["player_id"].nunique() == 4
    assert set(derived["lineups"]["lineup_status"]) >= {"starter", "bench"}
    assert set(derived["formations"]["source"]) >= {"payload_team", "raw_event"}


def test_build_derived_tables_empty_payloads_are_schema_only():
    derived = ws.build_derived_tables(
        payloads={},
        schedule_df=pd.DataFrame(),
        league="ENG-Premier League",
        season_int=2025,
        derived_tables=ws.DERIVED_TABLE_CHOICES,
    )

    assert list(derived["match_info"].columns) == ws.COLS_MATCH_INFO
    assert list(derived["lineups"].columns) == ws.COLS_LINEUPS
    assert derived["formations"].empty


def test_build_match_stats_tables_from_raw_payload_fixture():
    stats = ws.build_match_stats_tables(
        payloads={123456: _sample_payload()},
        report_blobs={123456: {"matchFacts": {"xg": {"home": 1.7, "away": 0.9}}}},
        schedule_df=_schedule_df(),
        league="ENG-Premier League",
        season_int=2025,
        stats_mode="all-visible",
    )

    assert set(stats) == {"team_match_stats", "player_match_stats", "match_fact_stats"}
    assert {"team", "stat_group", "stat_key", "source_tab"}.issubset(
        stats["team_match_stats"].columns
    )
    assert (stats["team_match_stats"]["team"] == "Home FC").any()
    assert (stats["player_match_stats"]["player"] == "Home Starter").any()
    assert (stats["match_fact_stats"]["source_tab"] == "report:matchFacts").any()


def test_build_match_stats_tables_collapses_minute_keyed_stat_mappings():
    payload = _sample_payload()
    payload["home"]["players"][0]["stats"] = {
        "possession": {"3": 1, "5": 1, "7": 2, "90": 7},
        "ratings": {"0": 6.5, "45": 7.1, "90": 7.8},
        "shotsTotal": {"90": 4},
    }

    stats = ws.build_match_stats_tables(
        payloads={123456: payload},
        report_blobs={},
        schedule_df=_schedule_df(),
        league="ENG-Premier League",
        season_int=2025,
        stats_mode="all-visible",
    )

    player_stats = stats["player_match_stats"]
    target = player_stats[player_stats["player"] == "Home Starter"]

    assert (target["stat_key"] == "possession").sum() == 1
    assert (target["stat_key"] == "ratings").sum() == 1
    assert (target["stat_key"] == "shots_total").sum() == 1
    assert target.loc[target["stat_key"] == "possession", "value_text"].iloc[0] == "7"
    assert target.loc[target["stat_key"] == "ratings", "value_text"].iloc[0] == "7.8"


def test_parse_native_season_options():
    html = """
    <html><body>
    <select id="seasons">
      <option value="/Regions/252/Tournaments/2/Seasons/9685">2025/2026</option>
      <option value="/Regions/252/Tournaments/2/Seasons/9314">2024/2025</option>
    </select>
    </body></html>
    """
    competition = CompetitionConfig(
        key="ENG-Premier League",
        source_name="England - Premier League",
        region_id=252,
        tournament_id=2,
        competition_type="club",
        season_mode="split-year",
    )

    records = parse_season_options(html, competition=competition)

    assert [record.season for record in records] == ["2025", "2024"]


def test_parse_native_schedule_month_payload():
    payload = {
        "tournaments": [
            {
                "matches": [
                    {
                        "id": 123456,
                        "startTimeUtc": "2025-08-15T19:00:00Z",
                        "status": "FT",
                        "homeTeamId": 10,
                        "homeTeamName": "Home FC",
                        "awayTeamId": 20,
                        "awayTeamName": "Away FC",
                        "homeScore": 2,
                        "awayScore": 1,
                    }
                ]
            }
        ]
    }

    df = parse_schedule_month_payload(
        payload,
        league="ENG-Premier League",
        season="2025",
        stage_id=1,
        stage_name="Regular Season",
    )

    assert df.iloc[0]["game_id"] == 123456
    assert df.iloc[0]["game"] == "2025-08-15 Home FC-Away FC"


def test_parse_native_missing_players_html():
    html = """
    <div id="missing-players">
      <div></div>
      <div>
        <table><tbody>
          <tr>
            <td class="pn"><a href="/Players/101/Home-Starter">Home Starter</a></td>
            <td class="reason"><span title="Hamstring">Hamstring</span></td>
            <td class="confirmed">Confirmed</td>
          </tr>
        </tbody></table>
      </div>
      <div>
        <table><tbody>
          <tr>
            <td class="pn"><a href="/Players/201/Away-Starter">Away Starter</a></td>
            <td class="reason"><span title="Suspension">Suspension</span></td>
            <td class="confirmed">Likely</td>
          </tr>
        </tbody></table>
      </div>
    </div>
    """

    df = parse_missing_players_html(
        html,
        schedule_row=_schedule_df().iloc[0].to_dict(),
        league="ENG-Premier League",
        season="2025",
    )

    assert set(df["team"]) == {"Home FC", "Away FC"}
    assert set(df["player_id"]) == {101, 201}


def test_extract_match_centre_payload():
    html = """
    <script>
      require.config.params['args'] = {"matchCentreData": {"matchId": 123456, "events": []}};
    </script>
    """

    payload = extract_match_centre_payload(html)

    assert payload == {"matchId": 123456, "events": []}


def test_parse_calendar_mask_from_js_object():
    html = """
    <script>
      var wsCalendar = {
        min: (new Date(2025, 7, 15)).toString(),
        max: (new Date(2026, 4, 24)).toString(),
        mask:{2025:{7:{15:1,16:1},8:{13:1}},2026:{0:{1:1},4:{24:1}}}
      };
    </script>
    """

    mask = parse_calendar_mask(html)

    assert mask == {"2025": ["7", "8"], "2026": ["0", "4"]}


def test_parse_embedded_tournament_fixtures():
    html = """
    <script type="application/json" data-hypernova-key="tournamentfixtures"><!--
    {"tournaments":[{"matches":[{"id":123456,"startTimeUtc":"2025-08-15T19:00:00Z","status":"FT","homeTeamId":10,"homeTeamName":"Home FC","awayTeamId":20,"awayTeamName":"Away FC","homeScore":2,"awayScore":1}]}]}
    --></script>
    """

    df = parse_embedded_tournament_fixtures(
        html,
        league="ENG-Premier League",
        season="2025",
        stage_id=1,
        stage_name="Regular Season",
    )

    assert df.iloc[0]["game_id"] == 123456
    assert df.iloc[0]["home_team"] == "Home FC"


def test_legacy_whoscored_scraper_delegates(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)

    monkeypatch.setattr(ws_legacy, "whoscored_main", fake_main)
    ws_legacy.main(["--league", "ENG-Premier League", "--seasons", "2025-2026", "--delay", "0"])

    assert captured["argv"][:6] == [
        "--league",
        "ENG-Premier League",
        "--out-dir",
        "data/raw/whoscored",
        "--tables",
        "schedule",
    ]
    assert "--no-derived-tables" in captured["argv"]
