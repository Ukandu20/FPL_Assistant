import json
import uuid
from pathlib import Path

import pandas as pd
import pytest

from scripts.fbref_pipeline.clean.csv_cleaner import (
    _normalise_team_value,
    canonicalize_team_season_ids,
    normalise_fpl_position,
    repair_team_match_identity,
    validate_raw_season_dir,
)
from scripts.fbref_pipeline.scrape.match_stats_scraper import (
    validate_schedule_season,
)


def _case_dir(name: str) -> Path:
    return Path(".tmp") / f"{name}_{uuid.uuid4().hex}"


def test_schedule_guard_accepts_dates_inside_requested_season():
    frame = pd.DataFrame({"date": ["2026-08-15", "2027-05-23"]})

    validate_schedule_season(frame, "2026-2027")


def test_schedule_guard_rejects_wrong_century():
    frame = pd.DataFrame({"date": ["1926-08-28", "1927-05-07"]})

    with pytest.raises(ValueError, match="outside 2026-2027"):
        validate_schedule_season(frame, "2026-2027")


def test_cleaner_rejects_wrong_season_schedule():
    season_dir = _case_dir("fbref_wrong_season") / "2026-2027"
    season_dir.mkdir(parents=True)
    pd.DataFrame({"date": ["1926-08-28"]}).to_csv(
        season_dir / "schedule.csv", index=False
    )

    with pytest.raises(ValueError, match="outside 2026-2027"):
        validate_raw_season_dir(season_dir, "2026-2027")


def test_cleaner_rejects_wholly_schema_only_manifest():
    season_dir = _case_dir("fbref_schema_only") / "2026-2027"
    meta_dir = season_dir / "_meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "coverage_manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "level": "player_season",
                        "stat_type": "standard",
                        "status": "schema_only",
                        "rows": 0,
                    },
                    {
                        "level": "team_season",
                        "stat_type": "standard",
                        "status": "schema_only",
                        "rows": 0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="every requested statistical table"):
        validate_raw_season_dir(season_dir, "2026-2027")


def test_team_normalisation_handles_null_values():
    assert pd.isna(_normalise_team_value(float("nan"), {}))
    assert _normalise_team_value(" Arsenal ", {"arsenal": "ARS"}) == "ARS"
    assert normalise_fpl_position("GK") == "GKP"
    assert normalise_fpl_position("MID") == "MID"


def test_team_match_identity_is_repaired_from_schedule():
    season_dir = _case_dir("fbref_team_repair") / "2025-2026"
    team_match_dir = season_dir / "team_match"
    team_match_dir.mkdir(parents=True)
    report = "/en/matches/a071faa8/example"
    pd.DataFrame(
        {
            "date": ["2025-08-15"],
            "home_team": ["Liverpool"],
            "away_team": ["Bournemouth"],
            "match_report": [report],
        }
    ).to_csv(season_dir / "schedule.csv", index=False)
    source = team_match_dir / "shooting.csv"
    frame = pd.DataFrame(
        {
            "team": [float("nan"), float("nan")],
            "game": ["2025-08-15 Liverpool-nan"] * 2,
            "venue": ["Home", "Away"],
            "match_report": [report, report],
        }
    )

    repaired = repair_team_match_identity(frame, source)

    assert repaired["team"].tolist() == ["Liverpool", "Bournemouth"]
    assert repaired["game"].tolist() == [
        "2025-08-15 Liverpool-Bournemouth",
        "2025-08-15 Liverpool-Bournemouth",
    ]


def test_team_season_uses_registry_id_and_retains_fbref_native_id():
    frame = pd.DataFrame(
        [{"team": "ARS", "team_id": "18bb7c10", "matches": 38}]
    )

    result = canonicalize_team_season_ids(
        frame,
        mt_lookup={"ars": "1dd1f33c"},
        team_map={"ars": "ARS"},
    )

    assert result.loc[0, "team_id"] == "1dd1f33c"
    assert result.loc[0, "fbref_team_id"] == "18bb7c10"
