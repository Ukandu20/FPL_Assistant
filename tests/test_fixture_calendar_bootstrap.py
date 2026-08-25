import json

import pandas as pd

from fpl_assistant.providers.fbref.integrate.fixtures_meta_builder import (
    _fixture_finished_mask,
    _observed_match_mask,
    build_bootstrap_fixture_calendar,
)
from fpl_assistant.canonical.identity import stable_canonical_id
from fpl_assistant.testing.paths import get_test_run_dir


def test_bootstrap_calendar_uses_only_fpl_and_stable_team_pair_identity():
    test_root = get_test_run_dir("fixture_calendar_bootstrap")
    fpl_csv = test_root / "fixtures.csv"
    teams_csv = test_root / "teams.csv"
    team_map = test_root / "team_ids.json"
    short_map = test_root / "teams.json"
    out_root = test_root / "registry" / "fixtures"

    pd.DataFrame(
        [
            {
                "id": 1,
                "event": 1,
                "kickoff_time": "2026-08-21T19:00:00Z",
                "team_h": 1,
                "team_a": 2,
                "finished": False,
                "team_h_score": None,
                "team_a_score": None,
            }
        ]
    ).to_csv(fpl_csv, index=False)
    pd.DataFrame(
        [
            {"id": 1, "name": "Arsenal", "short_name": "ARS"},
            {"id": 2, "name": "Coventry City", "short_name": "COV"},
        ]
    ).to_csv(teams_csv, index=False)
    team_map.write_text(json.dumps({"ars": "ars-id"}), encoding="utf-8")
    short_map.write_text(json.dumps({"Arsenal": "ARS", "Coventry City": "COV"}), encoding="utf-8")

    assert build_bootstrap_fixture_calendar(
        season="2026-2027",
        league="ENG-Premier League",
        fpl_csv=fpl_csv,
        teams_csv=teams_csv,
        team_map_fp=team_map,
        short_map_fp=short_map,
        out_dir=out_root,
        force=True,
    )
    output = pd.read_csv(out_root / "2026-2027" / "fixture_calendar.csv")
    assert len(output) == 2
    assert output["match_id"].nunique() == 1
    assert output["fbref_id"].equals(output["match_id"])
    assert set(output["team"]) == {"ARS", "COV"}
    cov_id = stable_canonical_id("team", "fpl", "COV", length=12)
    assert set(output["team_id"]) == {"ars-id", cov_id}
    assert output["date_played"].isna().all()
    assert output["sched_missing"].eq(0).all()

    first_match_id = output.loc[0, "match_id"]
    fixtures = pd.read_csv(fpl_csv)
    fixtures.loc[0, "kickoff_time"] = "2026-09-30T19:00:00Z"
    fixtures.to_csv(fpl_csv, index=False)
    assert build_bootstrap_fixture_calendar(
        season="2026-2027",
        league="ENG-Premier League",
        fpl_csv=fpl_csv,
        teams_csv=teams_csv,
        team_map_fp=team_map,
        short_map_fp=short_map,
        out_dir=out_root,
        force=True,
    )
    rerun = pd.read_csv(out_root / "2026-2027" / "fixture_calendar.csv")
    assert rerun.loc[0, "match_id"] == first_match_id


def test_observed_match_mask_ignores_provisional_schedule_rows():
    rows = pd.DataFrame(
        [
            {
                "is_result": False,
                "has_data": False,
                "team_goals": None,
                "opp_goals": None,
                "team_xg": None,
                "opp_xg": None,
            },
            {
                "is_result": True,
                "has_data": True,
                "team_goals": 2,
                "opp_goals": 1,
                "team_xg": 1.7,
                "opp_xg": 0.8,
            },
        ]
    )

    assert _observed_match_mask(rows).tolist() == [False, True]


def test_fixture_finished_mask_accepts_provisional_and_elapsed_matches():
    fixtures = pd.DataFrame(
        {
            "finished": [False, False, False],
            "finished_provisional": [True, False, False],
            "started": [True, True, False],
            "minutes": [90, 90, 0],
        }
    )

    assert _fixture_finished_mask(fixtures).tolist() == [True, True, False]
