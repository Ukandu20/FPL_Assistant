from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fpl_assistant.apps.viewmodels.player_card import (
    forecast_summary,
    latest_forecast_path,
    prepare_player_forecast,
    profile_dimensions,
)
from fpl_assistant.domain.player_profiles import build_player_profiles
from fpl_assistant.providers.fpl.pipelines.player_profiles import (
    profile_output_path,
    publish_player_profiles,
)


def profile_inputs() -> tuple[pd.DataFrame, ...]:
    ids = ["m1", "m2", "m3", "g1", "g2", "g3", "missing"]
    fpl = pd.DataFrame(
        {
            "player_id": ids,
            "name": ["Mid One", "Mid Two", "Mid Three", "GK One", "GK Two", "GK Three", "Missing"],
            "team": ["AAA"] * len(ids),
            "fpl_pos": ["MID", "MID", "MID", "GKP", "GKP", "GKP", "MID"],
            "minutes": [450, 900, 1800, 900, 1800, 2700, 1800],
            "goals_scored": [1, 5, 8, 0, 0, 0, 2],
            "assists": [1, 4, 7, 0, 0, 0, 2],
            "xg": [1.5, 4.5, 7.0, 0, 0, 0, 2],
            "xa": [1.0, 3.5, 6.5, 0, 0, 0, 2],
        }
    )
    outfield_ids = ["m1", "m2", "m3"]
    shooting = pd.DataFrame(
        {
            "player_id": outfield_ids,
            "shots_total": [10, 30, 50],
            "shots_on_target": [4, 14, 24],
        }
    )
    passing = pd.DataFrame(
        {
            "player_id": ids[:-1],
            "key_passes": [8, 25, 50, 0, 0, 0],
            "big_chances_created": [1, 5, 12, 0, 0, 0],
            "passes_attempted": [100, 300, 700, 300, 800, 1400],
            "pass_completion_pct": [70, 75, 82, 65, 78, 90],
            "progressive_passes": [5, 20, 50, 10, 40, 100],
        }
    )
    defense = pd.DataFrame(
        {
            "player_id": outfield_ids,
            "tackles_won": [5, 20, 45],
            "interceptions": [4, 16, 35],
            "clearances": [2, 10, 20],
            "blocks": [1, 8, 15],
            "recoveries": [12, 45, 100],
        }
    )
    keepers = pd.DataFrame(
        {
            "player_id": ["g1", "g2", "g3"],
            "saves": [50, 70, 85],
            "save_pct": [82, 74, 68],
            "goals_against": [8, 20, 35],
            "keeper_sweeper_actions": [2, 20, 60],
            "smothers": [1, 8, 20],
        }
    )
    return fpl, shooting, passing, defense, keepers


def test_profiles_apply_reliability_and_bound_position_scores() -> None:
    fpl, shooting, passing, defense, keepers = profile_inputs()
    profiles = build_player_profiles(
        fpl,
        shooting=shooting,
        passing=passing,
        defense=defense,
        keepers=keepers,
        season="2025-2026",
    )

    mid_one = profiles.loc[profiles["player_id"].eq("m1")].iloc[0]
    mid_three = profiles.loc[profiles["player_id"].eq("m3")].iloc[0]
    assert mid_one["profile_status"] == "Provisional"
    assert mid_three["profile_status"] == "Established"
    assert mid_one["reliability"] == pytest.approx(1 / 3)
    assert mid_three["reliability"] == pytest.approx(2 / 3)

    eligible = profiles[profiles["profile_status"].isin(["Provisional", "Established"])]
    for column in [
        "goal_threat_score",
        "creativity_score",
        "defensive_threat_score",
        "shot_stopping_score",
        "sweeping_score",
        "distribution_score",
    ]:
        values = pd.to_numeric(eligible[column], errors="coerce").dropna()
        assert values.between(0, 100).all()


def test_goalkeepers_receive_goalkeeper_archetypes_and_dimensions() -> None:
    fpl, shooting, passing, defense, keepers = profile_inputs()
    profiles = build_player_profiles(
        fpl,
        shooting=shooting,
        passing=passing,
        defense=defense,
        keepers=keepers,
    )
    goalkeepers = profiles.loc[profiles["fpl_pos"].eq("GKP")]

    assert set(goalkeepers["production_archetype"]).issubset(
        {
            "Shot Stopper",
            "Sweeper Keeper",
            "Distributor",
            "Proactive Shot Stopper",
            "Ball-Playing Shot Stopper",
            "Ball-Playing Sweeper",
            "Complete Goalkeeper",
            "Balanced Profile",
            "Low-Production",
        }
    )
    dimensions = profile_dimensions(goalkeepers.iloc[0])
    assert [item["Dimension"] for item in dimensions] == [
        "Shot stopping",
        "Sweeping",
        "Distribution",
    ]


def test_missing_provider_coverage_is_not_inferred_as_zero() -> None:
    fpl, shooting, passing, defense, keepers = profile_inputs()
    profiles = build_player_profiles(
        fpl,
        shooting=shooting,
        passing=passing,
        defense=defense,
        keepers=keepers,
    )
    missing = profiles.loc[profiles["player_id"].eq("missing")].iloc[0]
    assert missing["profile_status"] == "Data unavailable"
    assert missing["production_archetype"] == "Data Unavailable"
    assert pd.isna(missing["goal_threat_score"])


def test_duplicate_fpl_player_ids_are_rejected() -> None:
    fpl, shooting, passing, defense, keepers = profile_inputs()
    duplicated = pd.concat([fpl, fpl.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate player IDs"):
        build_player_profiles(
            duplicated,
            shooting=shooting,
            passing=passing,
            defense=defense,
            keepers=keepers,
        )


def test_profile_publisher_writes_season_artifact(tmp_path: Path) -> None:
    fpl, shooting, passing, defense, keepers = profile_inputs()
    league = "Test League"
    season = "2025-2026"
    fpl_root = tmp_path / "data" / "processed" / "fpl" / league / season / "season"
    provider_root = (
        tmp_path
        / "data"
        / "processed"
        / "whoscored"
        / league
        / season
        / "player_season"
    )
    fpl_root.mkdir(parents=True)
    provider_root.mkdir(parents=True)
    fpl.to_csv(fpl_root / "cleaned_players.csv", index=False)
    shooting.to_csv(provider_root / "shooting.csv", index=False)
    passing.to_csv(provider_root / "passing.csv", index=False)
    defense.to_csv(provider_root / "defense.csv", index=False)
    keepers.to_csv(provider_root / "keepers.csv", index=False)

    profiles = publish_player_profiles(
        league, season, project_root=tmp_path
    )
    output = profile_output_path(tmp_path, league, season)

    assert output.is_file()
    assert len(pd.read_csv(output)) == len(profiles) == len(fpl)
    assert "production_archetype" in profiles


def test_forecast_selection_is_exact_season_and_prefers_parquet(tmp_path: Path) -> None:
    season_root = tmp_path / "2025-2026"
    season_root.mkdir()
    (season_root / "GW7_9.csv").touch()
    (season_root / "GW8_10.csv").touch()
    expected = season_root / "GW8_10.parquet"
    expected.touch()

    assert latest_forecast_path(tmp_path, "2025-2026") == expected
    assert latest_forecast_path(tmp_path, "2026-2027") is None


def test_player_forecast_filters_season_and_deduplicates_fixture() -> None:
    forecasts = pd.DataFrame(
        {
            "season": ["2025-2026", "2025-2026", "2025-2026", "2024-2025"],
            "player_id": ["p1", "p1", "p1", "p1"],
            "gw_orig": [8, 8, 9, 8],
            "game_id": ["g8", "g8", "g9", "old"],
            "opponent": ["AAA", "AAA", "BBB", "CCC"],
            "is_home": [1, 1, 0, 1],
            "pred_minutes": [80, 85, 70, 90],
            "xPts": [4.0, 4.5, 3.0, 8.0],
        }
    )
    result = prepare_player_forecast(forecasts, "p1", season="2025-2026")
    summary = forecast_summary(result)

    assert result["game_id"].tolist() == ["g8", "g9"]
    assert result.iloc[0]["pred_minutes"] == 85
    assert summary["expected_points"] == pytest.approx(7.5)
    assert summary["next_fixture"] == "AAA (H)"
