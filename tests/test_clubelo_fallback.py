from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from fpl_assistant.providers.clubelo.fallback.elo_fallback import (
    FallbackConfig,
    FallbackError,
    build_fallback,
    build_schedule_output,
    elo_expected,
    prepare_anchor_and_fixtures,
    prepare_completed_matches,
    publish_fallback,
    replay_matches,
)


def _anchor_schedule() -> pd.DataFrame:
    common = {
        "elo_preseason_as_of": "2026-08-20",
        "status": "scheduled",
        "date_played": "",
        "gf": "",
        "ga": "",
        "result": "",
        "elo_provider": "clubelo",
    }
    rows = []
    fixtures = [
        ("m1", "2026-08-21", "A", "a-id", 1600.0, "B", "b-id", 1500.0),
        ("m2", "2026-08-28", "B", "b-id", 1500.0, "A", "a-id", 1600.0),
    ]
    for match_id, date, home, home_id, home_elo, away, away_id, away_elo in fixtures:
        rows.extend(
            [
                {
                    **common,
                    "match_id": match_id,
                    "date_sched": date,
                    "team": home,
                    "team_id": home_id,
                    "opponent_id": away_id,
                    "is_home": 1,
                    "home_id": home_id,
                    "away_id": away_id,
                    "elo_preseason": home_elo,
                },
                {
                    **common,
                    "match_id": match_id,
                    "date_sched": date,
                    "team": away,
                    "team_id": away_id,
                    "opponent_id": home_id,
                    "is_home": 0,
                    "home_id": home_id,
                    "away_id": away_id,
                    "elo_preseason": away_elo,
                },
            ]
        )
    return pd.DataFrame(rows)


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1", "game_date": "2026-08-21", "game_time": "19:00:00",
                "team": "A", "team_id": "a-id", "opp": "B", "opp_id": "b-id",
                "venue": "H", "team_goals": 2, "opp_goals": 0,
                "is_result": True, "has_data": True, "round": 1,
            },
            {
                "match_id": "m1", "game_date": "2026-08-21", "game_time": "19:00:00",
                "team": "B", "team_id": "b-id", "opp": "A", "opp_id": "a-id",
                "venue": "A", "team_goals": 0, "opp_goals": 2,
                "is_result": True, "has_data": True, "round": 1,
            },
            {
                "match_id": "m2", "game_date": "2026-08-28", "game_time": "19:00:00",
                "team": "B", "team_id": "b-id", "opp": "A", "opp_id": "a-id",
                "venue": "H", "team_goals": pd.NA, "opp_goals": pd.NA,
                "is_result": False, "has_data": False, "round": 2,
            },
            {
                "match_id": "m2", "game_date": "2026-08-28", "game_time": "19:00:00",
                "team": "A", "team_id": "a-id", "opp": "B", "opp_id": "b-id",
                "venue": "A", "team_goals": pd.NA, "opp_goals": pd.NA,
                "is_result": False, "has_data": False, "round": 2,
            },
        ]
    )


def _prepared(as_of: str = "2026-08-25"):
    source = _anchor_schedule()
    anchors, fixtures = prepare_anchor_and_fixtures(
        source, expected_team_count=2, expected_fixture_count=2,
        expected_anchor_date="2026-08-20",
    )
    matches, missing = prepare_completed_matches(
        _results(), fixtures, anchors, as_of=as_of
    )
    return source, anchors, fixtures, matches, missing


def test_expected_result_is_neutral_and_symmetric():
    assert elo_expected(1500, 1500, hfa=0) == pytest.approx(0.5)
    forward = elo_expected(1620, 1500, hfa=0)
    reverse = elo_expected(1500, 1620, hfa=0)
    assert forward + reverse == pytest.approx(1.0)


def test_replay_is_zero_sum_and_records_pre_match_before_update():
    _, anchors, _, matches, _ = _prepared()
    ledger, history, final = replay_matches(
        anchors, matches, config=FallbackConfig(hfa=0)
    )

    row = ledger.iloc[0]
    assert row["home_elo_pre_match"] == 1600.0
    assert row["away_elo_pre_match"] == 1500.0
    assert row["applied_delta"] > 0
    assert row["home_elo_post_match"] == pytest.approx(1600 + row["applied_delta"])
    assert math.fsum(final.values()) == pytest.approx(3100.0, abs=1e-10)
    assert len(history) == 4  # two anchors plus two post-match states


def test_calibrated_margin_requires_an_explicit_normalizer():
    with pytest.raises(FallbackError, match="margin_normalizer"):
        FallbackConfig(margin_mode="calibrated_margin").validate()

    _, anchors, _, matches, _ = _prepared()
    base, _, _ = replay_matches(anchors, matches, config=FallbackConfig(hfa=0))
    margin, _, _ = replay_matches(
        anchors,
        matches,
        config=FallbackConfig(
            hfa=0, margin_mode="calibrated_margin", margin_normalizer=1.0
        ),
    )
    assert margin.loc[0, "margin_factor"] == pytest.approx(math.sqrt(2))
    assert abs(margin.loc[0, "applied_delta"]) > abs(base.loc[0, "applied_delta"])


def test_partial_or_non_mirrored_result_is_rejected():
    source = _anchor_schedule()
    anchors, fixtures = prepare_anchor_and_fixtures(
        source, expected_team_count=2, expected_fixture_count=2
    )
    bad = _results()
    bad.loc[bad["venue"].eq("A") & bad["match_id"].eq("m1"), "opp_goals"] = 1

    with pytest.raises(FallbackError, match="scores do not mirror"):
        prepare_completed_matches(bad, fixtures, anchors, as_of="2026-08-25")


def test_past_fixture_without_result_fails_closed():
    source = _anchor_schedule()
    anchors, fixtures = prepare_anchor_and_fixtures(
        source, expected_team_count=2, expected_fixture_count=2
    )
    with pytest.raises(FallbackError, match="past fixtures have no accepted result"):
        prepare_completed_matches(
            _results(), fixtures, anchors, as_of="2026-08-30"
        )


def test_schedule_uses_match_pre_elo_and_latest_elo_for_future_fixture():
    source, anchors, _, matches, _ = _prepared()
    ledger, _, final = replay_matches(anchors, matches)
    schedule = build_schedule_output(source, ledger, final)

    completed_home = schedule.loc[
        schedule["match_id"].eq("m1") & schedule["team_id"].eq("a-id")
    ].iloc[0]
    future_home = schedule.loc[
        schedule["match_id"].eq("m2") & schedule["team_id"].eq("b-id")
    ].iloc[0]
    assert completed_home["elo_pre_match"] == 1600.0
    assert completed_home["status"] == "finished"
    assert completed_home["gf"] == 2
    assert completed_home["elo_provider"] == "clubelo_fallback"
    assert future_home["elo_pre_match"] == pytest.approx(final["b-id"])
    assert pd.isna(future_home["elo_post_match"])


def test_build_is_deterministic_and_publish_writes_complete_isolated_set(tmp_path: Path):
    anchor_path = tmp_path / "anchor.csv"
    results_path = tmp_path / "results.csv"
    _anchor_schedule().to_csv(anchor_path, index=False)
    _results().to_csv(results_path, index=False)

    kwargs = dict(
        anchor_schedule_path=anchor_path,
        results_schedule_path=results_path,
        as_of="2026-08-25",
        expected_team_count=2,
        expected_fixture_count=2,
        expected_anchor_date="2026-08-20",
    )
    first = build_fallback(**kwargs)
    second = build_fallback(**kwargs)
    assert first.audit["outputs"] == second.audit["outputs"]
    assert first.audit["counts"]["completed_matches"] == 1
    assert first.audit["invariants"]["zero_sum"] is True

    destination = tmp_path / "clubelo_fallback" / "league" / "season"
    publish_fallback(first, destination)
    assert {path.name for path in destination.iterdir()} == {
        "match_ledger.csv", "team_history.csv", "schedule.csv", "audit.json"
    }
    audit = json.loads((destination / "audit.json").read_text(encoding="utf-8"))
    assert audit["provider"] == "clubelo_fallback"
    assert not (tmp_path / "clubelo").exists()

    # Re-publication exercises the atomic destination swap and remains complete.
    publish_fallback(second, destination)
    assert {path.name for path in destination.iterdir()} == {
        "match_ledger.csv", "team_history.csv", "schedule.csv", "audit.json"
    }


def test_older_anchor_schedule_can_use_fbref_id_as_canonical_match_id(tmp_path: Path):
    anchor = _anchor_schedule().rename(columns={"match_id": "fbref_id"})
    anchor_path = tmp_path / "older_anchor.csv"
    results_path = tmp_path / "results.csv"
    anchor.to_csv(anchor_path, index=False)
    _results().to_csv(results_path, index=False)

    result = build_fallback(
        anchor_schedule_path=anchor_path,
        results_schedule_path=results_path,
        as_of="2026-08-25",
        expected_team_count=2,
        expected_fixture_count=2,
        expected_anchor_date="2026-08-20",
    )

    assert result.ledger["match_id"].tolist() == ["m1"]
    assert result.schedule["match_id"].tolist() == ["m1", "m1", "m2", "m2"]
