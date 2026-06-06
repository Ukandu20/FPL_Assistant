from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd

from scripts.clubelo_pipeline.clean.clubelo_understat_enricher import (
    TeamResolver,
    build_elo_lookup,
    clean_clubelo_history,
    enrich_understat_match_df,
    find_elo_pair,
    run_pipeline,
)


def test_clean_clubelo_history_standardizes_dates_league_and_team_codes():
    resolver = TeamResolver({"Arsenal": "ARS", "Manchester United": "MUN"}, {"Man United": "MUN"})
    raw = pd.DataFrame(
        [
            {
                "From": "2025-08-16",
                "To": "2025-08-17",
                "Club": "Man United",
                "Country": "ENG",
                "Level": "1",
                "Elo": "1600.5",
                "Rank": "12",
            }
        ]
    )

    out = clean_clubelo_history(raw, source_file="Man_United.csv", resolver=resolver)

    assert out.loc[0, "league"] == "ENG-Premier League"
    assert out.loc[0, "clubelo_league"] == "ENG_1"
    assert out.loc[0, "team_code"] == "MUN"
    assert out.loc[0, "from"] == pd.Timestamp("2025-08-16")
    assert out.loc[0, "to"] == pd.Timestamp("2025-08-17")
    assert out.loc[0, "elo"] == 1600.5


def test_find_elo_pair_uses_same_or_next_day_post_match_row():
    hist = pd.DataFrame(
        [
            {"from": pd.Timestamp("2025-08-10"), "to": pd.Timestamp("2025-08-17"), "elo": 1500.0},
            {"from": pd.Timestamp("2025-08-18"), "to": pd.Timestamp("2025-08-24"), "elo": 1512.0},
        ]
    )

    pair = find_elo_pair(hist, "2025-08-17", is_result=True)

    assert pair.start_elo == 1500.0
    assert pair.end_elo == 1512.0
    assert pair.reason == ""


def test_enrich_understat_match_df_adds_team_opp_elos_and_diffs_idempotently():
    history = pd.DataFrame(
        [
            {
                "league": "ENG-Premier League",
                "clubelo_league": "ENG_1",
                "team_code": "ARS",
                "team": "Arsenal",
                "from": pd.Timestamp("2025-08-10"),
                "to": pd.Timestamp("2025-08-17"),
                "elo": 1500.0,
            },
            {
                "league": "ENG-Premier League",
                "clubelo_league": "ENG_1",
                "team_code": "ARS",
                "team": "Arsenal",
                "from": pd.Timestamp("2025-08-18"),
                "to": pd.Timestamp("2025-08-24"),
                "elo": 1510.0,
            },
            {
                "league": "ENG-Premier League",
                "clubelo_league": "ENG_1",
                "team_code": "MUN",
                "team": "Man United",
                "from": pd.Timestamp("2025-08-10"),
                "to": pd.Timestamp("2025-08-17"),
                "elo": 1600.0,
            },
            {
                "league": "ENG-Premier League",
                "clubelo_league": "ENG_1",
                "team_code": "MUN",
                "team": "Man United",
                "from": pd.Timestamp("2025-08-17"),
                "to": pd.Timestamp("2025-08-24"),
                "elo": 1594.0,
            },
        ]
    )
    understat = pd.DataFrame(
        [
            {
                "league": "ENG-Premier League",
                "season": "2025-2026",
                "game_id": "g1",
                "game_date": "2025-08-17",
                "team": "ARS",
                "opp": "MUN",
                "result": "W",
                "team_start_elo": "old",
            }
        ]
    )

    out, audit = enrich_understat_match_df(understat, build_elo_lookup(history))

    assert audit.empty
    assert out.columns.tolist().count("team_start_elo") == 1
    assert out.loc[0, "team_start_elo"] == "1500"
    assert out.loc[0, "team_end_elo"] == "1510"
    assert out.loc[0, "opp_start_elo"] == "1600"
    assert out.loc[0, "opp_end_elo"] == "1594"
    assert out.loc[0, "elo_diff_start"] == "-100"
    assert out.loc[0, "elo_diff_end"] == "-84"


def test_future_fixture_gets_start_elo_only_and_missing_history_is_audited():
    history = pd.DataFrame(
        [
            {
                "league": "ENG-Premier League",
                "team_code": "ARS",
                "from": pd.Timestamp("2025-08-18"),
                "to": pd.Timestamp("2025-08-24"),
                "elo": 1510.0,
            }
        ]
    )
    understat = pd.DataFrame(
        [
            {
                "league": "ENG-Premier League",
                "season": "2025-2026",
                "game_id": "g2",
                "game_date": "2025-08-20",
                "team": "ARS",
                "opp": "XXX",
                "is_result": "False",
            }
        ]
    )

    out, audit = enrich_understat_match_df(understat, build_elo_lookup(history))

    assert out.loc[0, "team_start_elo"] == "1510"
    assert out.loc[0, "team_end_elo"] == ""
    assert out.loc[0, "opp_start_elo"] == ""
    assert set(audit["reason"]) == {"missing_history"}


def test_run_pipeline_writes_processed_clubelo_and_overwrites_understat():
    tmp_root = Path(".tmp") / f"clubelo_understat_{uuid.uuid4().hex}"
    raw_dir = tmp_root / "raw" / "clubelo" / "team_history"
    understat_dir = tmp_root / "processed" / "understat"
    out_root = tmp_root / "processed" / "clubelo"
    config_dir = tmp_root / "config"
    raw_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    season_dir = understat_dir / "ENG-Premier League" / "2025-2026"
    season_dir.mkdir(parents=True)

    (config_dir / "teams.json").write_text(
        json.dumps({"Arsenal": "ARS", "Manchester United": "MUN"}),
        encoding="utf-8",
    )
    (config_dir / "clubelo_team_aliases.json").write_text(
        json.dumps({"Man United": "MUN"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"from": "2025-08-10", "to": "2025-08-17", "team": "Arsenal", "country": "ENG", "level": "1", "elo": "1500"},
            {"from": "2025-08-18", "to": "2025-08-24", "team": "Arsenal", "country": "ENG", "level": "1", "elo": "1510"},
        ]
    ).to_csv(raw_dir / "Arsenal.csv", index=False)
    pd.DataFrame(
        [
            {"from": "2025-08-10", "to": "2025-08-17", "team": "Man United", "country": "ENG", "level": "1", "elo": "1600"},
            {"from": "2025-08-17", "to": "2025-08-24", "team": "Man United", "country": "ENG", "level": "1", "elo": "1594"},
        ]
    ).to_csv(raw_dir / "Man_United.csv", index=False)
    pd.DataFrame(
        [
            {
                "league": "ENG-Premier League",
                "season": "2025-2026",
                "game_id": "g1",
                "game_date": "2025-08-17",
                "team": "ARS",
                "opp": "MUN",
                "result": "W",
            }
        ]
    ).to_csv(season_dir / "team_match.csv", index=False)

    summary = run_pipeline(
        raw_clubelo_dir=raw_dir,
        understat_root=understat_dir,
        out_root=out_root,
        teams_config_path=config_dir / "teams.json",
        aliases_path=config_dir / "clubelo_team_aliases.json",
        overwrite=True,
    )

    assert summary["clubelo_processed_files_written"] == 1
    assert summary["understat_match_files_written"] == 1
    assert (out_root / "ENG-Premier League" / "2025-2026" / "team_history.csv").exists()

    enriched = pd.read_csv(season_dir / "team_match.csv", dtype=str, keep_default_na=False)
    assert enriched.loc[0, "team_start_elo"] == "1500"
    assert enriched.loc[0, "team_end_elo"] == "1510"
    assert enriched.loc[0, "opp_start_elo"] == "1600"
    assert enriched.loc[0, "opp_end_elo"] == "1594"
