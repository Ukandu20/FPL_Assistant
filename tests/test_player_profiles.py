from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fpl_assistant.apps.viewmodels.player_card import (
    comparable_players,
    decision_factors,
    forecast_summary,
    latest_forecast_path,
    prepare_player_forecast,
    prepare_player_fixtures,
    player_placeholder_url,
    player_photo_urls,
    profile_dimensions,
    profile_trend,
    recent_form_summary,
    select_profile_snapshot,
    team_badge_url,
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


def test_factual_fixtures_survive_missing_forecast() -> None:
    schedule = pd.DataFrame(
        {
            "fpl_id": [1, 2, 3],
            "gw_orig": [1, 2, 1],
            "team": [10, 10, 20],
            "opponent": ["AAA", "BBB", "CCC"],
            "is_home": [True, False, True],
            "date_sched": ["2026-08-21", "2026-08-29", "2026-08-22"],
            "fdr": [2, 4, 3],
        }
    )

    fixtures = prepare_player_fixtures(schedule, 10, forecast=pd.DataFrame())

    assert fixtures["opponent"].tolist() == ["AAA", "BBB"]
    assert fixtures.iloc[0]["fdr"] == 2
    assert "xPts" not in fixtures


def test_fixtures_are_enriched_when_forecast_is_available() -> None:
    schedule = pd.DataFrame(
        {
            "fpl_id": [1],
            "gw_orig": [1],
            "team": [10],
            "opponent": ["AAA"],
            "is_home": [True],
            "date_sched": ["2026-08-21"],
            "fdr": [2],
        }
    )
    forecast = pd.DataFrame(
        {
            "gw_orig": [1],
            "opponent": ["AAA"],
            "is_home": [True],
            "pred_minutes": [82],
            "xPts": [5.4],
        }
    )

    fixtures = prepare_player_fixtures(schedule, 10, forecast=forecast)

    assert fixtures.iloc[0]["pred_minutes"] == 82
    assert fixtures.iloc[0]["xPts"] == pytest.approx(5.4)


def test_profile_snapshot_uses_labelled_prior_baseline_when_current_is_empty() -> None:
    current = pd.DataFrame(
        {
            "player_id": ["p1"],
            "profile_status": ["Data unavailable"],
            "production_archetype": ["Data Unavailable"],
        }
    )
    previous = pd.DataFrame(
        {
            "player_id": ["p1"],
            "profile_status": ["Established"],
            "production_archetype": ["Playmaker"],
        }
    )

    profile, season, is_carryover = select_profile_snapshot(
        current,
        previous,
        "p1",
        current_season="2026-2027",
        previous_season="2025-2026",
    )

    assert profile is not None and profile["production_archetype"] == "Playmaker"
    assert season == "2025-2026"
    assert is_carryover is True


def test_recent_form_and_comparable_players_are_decision_ready() -> None:
    history = pd.DataFrame(
        {
            "GW": [1, 2, 3],
            "Total FPL points": [2, 8, 6],
            "Minutes": [90, 70, 25],
            "Starts": [1, 1, 0],
            "Returns": [0, 1, 1],
        }
    )
    summary = recent_form_summary(history)
    assert summary == {
        "gameweeks": 3,
        "points": 16.0,
        "minutes": 185.0,
        "appearances": 3,
        "starts": 2,
        "returns": 2,
    }

    players = pd.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4"],
            "name": ["Selected", "Popular", "Scorer", "Too expensive"],
            "team": ["AAA", "BBB", "CCC", "DDD"],
            "fpl_pos": ["MID", "MID", "MID", "MID"],
            "now_cost": [75, 76, 72, 90],
            "total_points": [5, 4, 8, 20],
            "selected_by_percent": [10, 20, 5, 30],
        }
    )
    alternatives = comparable_players(players, "p1")
    assert alternatives["Player"].tolist() == ["Scorer", "Popular"]


def test_decision_factors_surface_fixture_profile_and_uncertainty() -> None:
    fixtures = pd.DataFrame(
        {
            "opponent": ["AAA"],
            "is_home": [True],
            "fdr": [2],
            "pred_minutes": [pd.NA],
        }
    )
    profile = pd.Series(
        {
            "profile_status": "Established",
            "goal_threat_percentile": 82,
            "creativity_percentile": 65,
            "defensive_threat_percentile": 40,
            "fpl_pos": "MID",
        }
    )
    positives, risks = decision_factors(
        fixtures,
        {},
        profile,
        None,
        profile_is_carryover=True,
    )

    assert any("Favourable next fixture" in reason for reason in positives)
    assert any("P82" in reason for reason in positives)
    assert any("not been published" in risk for risk in risks)
    assert any("previous-season baseline" in risk for risk in risks)


def test_profile_trend_requires_compatible_reliable_seasons() -> None:
    previous = pd.DataFrame(
        {
            "player_id": ["p1"],
            "fpl_pos": ["MID"],
            "profile_status": ["Established"],
            "goal_threat_percentile": [60],
            "creativity_percentile": [70],
            "defensive_threat_percentile": [50],
        }
    )
    current = previous.copy()
    current["goal_threat_percentile"] = 80
    current["creativity_percentile"] = 90
    current["defensive_threat_percentile"] = 70

    assert profile_trend(current, previous, "p1") == {
        "label": "Rising",
        "delta": 20.0,
    }

    current["profile_status"] = "Insufficient data"
    assert profile_trend(current, previous, "p1") == {}


def test_player_photo_urls_use_versioned_collection_without_legacy_prefix() -> None:
    urls = player_photo_urls("223340.jpg", asset_version="25")

    assert urls["current"] == (
        "https://resources.premierleague.com/premierleague25/"
        "photos/players/110x140/223340.png"
    )
    assert urls["legacy"] == (
        "https://resources.premierleague.com/premierleague/"
        "photos/players/110x140/p223340.png"
    )
    assert player_photo_urls("../../unsafe", asset_version="25") == {}
    assert player_photo_urls("223340.jpg", asset_version="latest") == {}


def test_player_placeholder_url_uses_versioned_collection() -> None:
    assert player_placeholder_url(asset_version="25") == (
        "https://resources.premierleague.com/premierleague25/"
        "photos/players/110x140/placeholder.png"
    )
    assert player_placeholder_url(asset_version="latest") is None


def test_team_badge_url_uses_validated_raw_fpl_code() -> None:
    assert team_badge_url(3, asset_version="25") == (
        "https://resources.premierleague.com/"
        "premierleague25/badges-alt/3.svg"
    )
    assert team_badge_url("91", asset_version="25").endswith("/91.svg")
    assert team_badge_url("../3", asset_version="25") is None
    assert team_badge_url(3.5, asset_version="25") is None
