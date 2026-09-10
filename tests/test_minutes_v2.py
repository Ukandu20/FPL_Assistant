from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_assistant.minutes_v2.calibration import (
    CalibrationFoldData, HierarchicalCalibrators, ProbabilityCalibrator,
    select_calibration,
)
from fpl_assistant.minutes_v2.config import MinutesV2Config, load_config
from fpl_assistant.minutes_v2.data import (
    DatasetAudit,
    _normalize,
    canonical_training_rows,
    current_season_history_rows,
)
from fpl_assistant.minutes_v2.evaluation import (
    acceptance_gate, paired_block_bootstrap, probability_metrics, state_log_loss,
)
from fpl_assistant.minutes_v2.features import (
    CAMEO_FEATURES, CAMEO_MIN_FEATURES, P60_FEATURES, START_FEATURES,
    START_MIN_FEATURES, build_features,
)
from fpl_assistant.minutes_v2.folds import assert_holdout_isolated, generate_folds
from fpl_assistant.minutes_v2.inference import CalibrationBundle, add_legacy_compatibility, predict
from fpl_assistant.minutes_v2.legacy import prepare_v1_shadow_artifact
from fpl_assistant.minutes_v2.models import ConstantModel, HeadModel, ModelBundle
from fpl_assistant.minutes_v2.overrides import apply_confirmed_absence_overrides
from fpl_assistant.minutes_v2.schema import REQUIRED_OUTPUT_COLUMNS, validate_predictions


def _config() -> MinutesV2Config:
    return MinutesV2Config(
        architecture_version="minutes/v2.0", feature_version="minutes-v2.0.0",
        registry_root=Path("."), artifact_root=Path("artifacts"), shadow_root=Path("shadow"),
        canonical_label_seasons=("2022-2023", "2023-2024", "2024-2025", "2025-2026"),
        diagnostic_only_seasons=("2020-2021", "2021-2022"), forward_season="2026-2027",
        trusted_starter_sources=("fpl", "reconstructed_fpl_dnp"),
        untrusted_starter_sources=("fallback", "imputed", "pending", ""),
        acceptance_margins={"p60_brier": .002, "state_log_loss": .005, "xpoints_mae": .01},
        extensions={"season_priors_v2_1": False}, bootstrap_iterations=100,
    )


def test_repo_config_locks_contract_and_extensions() -> None:
    cfg = load_config("config/minutes_v2.json")
    assert cfg.architecture_version == "minutes/v2.0"
    assert cfg.calibration_gws == cfg.evaluation_gws == cfg.step_gws == 6
    assert not any(cfg.extensions.values())


def test_feature_lists_are_exact_and_exclude_v21_context() -> None:
    assert START_FEATURES == [
        "min_lag1", "min_ewm_hl2", "start_lag1", "start_rate_hl3",
        "start_streak", "bench_streak", "days_feat", "history_matches", "pos",
    ]
    assert START_MIN_FEATURES == ["min_lag1", "min_ewm_hl2", "start_rate_hl3", "days_feat", "history_matches", "pos"]
    assert CAMEO_FEATURES == ["min_lag1", "min_ewm_hl2", "start_rate_hl3", "bench_streak", "days_feat", "history_matches", "pos"]
    assert CAMEO_MIN_FEATURES == ["min_lag1", "min_ewm_hl2", "bench_streak", "days_feat", "history_matches", "pos"]
    assert P60_FEATURES == START_FEATURES
    assert not ({"played_last", "long_gap14", "fdr", "team_rot3"} & set(START_FEATURES))


def test_features_are_strictly_lagged_and_future_outcomes_do_not_change_them() -> None:
    base = pd.DataFrame({
        "season": ["2025-2026"] * 4, "player_id": ["p"] * 4, "pos": ["MF"] * 4,
        "date_sched": pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22"], utc=True),
        "minutes": [90, 30, np.nan, np.nan], "is_starter": [1, 0, np.nan, np.nan],
    })
    first = build_features(base, "2026-01-15T00:00:00Z")
    changed = base.copy()
    changed.loc[2:, "minutes"] = [90, 90]
    changed.loc[2:, "is_starter"] = [1, 1]
    second = build_features(changed, "2026-01-15T00:00:00Z")
    columns = ["min_lag1", "min_ewm_hl2", "start_lag1", "start_rate_hl3", "history_matches"]
    pd.testing.assert_frame_equal(first.loc[2:, columns], second.loc[2:, columns])
    assert first.loc[2, "min_lag1"] == 30
    assert first.loc[2, "history_matches"] == 2
    assert first.loc[0, "cold_start"] == 1


def test_training_filter_excludes_untrusted_seasons_labels_and_post_cutoff() -> None:
    rows = pd.DataFrame({
        "season": ["2020-2021", "2022-2023", "2022-2023", "2022-2023"],
        "date_sched": pd.to_datetime(["2021-01-01", "2023-01-01", "2023-01-02", "2027-01-01"], utc=True),
        "minutes": [90, 90, 30, 90], "is_starter": [1, 1, 0, 1],
        "eligible_for_fixture": [True] * 4, "confirmed_unavailable": [False] * 4,
        "information_timestamp": pd.to_datetime(["2020-12-01"] * 4, utc=True),
        "starter_source": ["fallback", "fpl", "fallback", "fpl"],
    })
    audit = DatasetAudit({}, {}, {}, {}, {}, {}, {})
    selected = canonical_training_rows(rows, _config(), "2026-01-01", audit)
    assert len(selected) == 1
    assert selected.iloc[0]["starter_source"] == "fpl"


def test_current_season_history_uses_only_trusted_outcomes_known_by_cutoff() -> None:
    rows = pd.DataFrame({
        "season": ["2026-2027"] * 4,
        "date_sched": pd.to_datetime(
            ["2026-08-21", "2026-08-21", "2026-08-21", "2026-08-31"], utc=True
        ),
        "minutes": [90, 30, 0, np.nan],
        "is_starter": [1, 0, 0, np.nan],
        "information_timestamp": pd.to_datetime(
            ["2026-08-25", "2026-08-27", "2026-08-25", "2026-08-25"], utc=True
        ),
        "starter_source": ["fpl", "fpl", "fallback", "pending"],
    })

    selected = current_season_history_rows(
        rows, "2026-2027", "2026-08-26T23:59:59Z", _config()
    )

    assert selected.index.tolist() == [0]


def test_registry_normalization_accepts_mixed_date_and_kickoff_precision() -> None:
    raw = pd.DataFrame({
        "date_sched": ["2026-08-21T20:00:00+01:00", "2026-08-22"],
        "date_played": [None, None],
        "information_timestamp": ["2026-08-20T22:00:00Z"] * 2,
        "eligible_for_fixture": [True, True],
        "confirmed_unavailable": [False, False],
        "eligibility_timestamp_safe": [True, True],
    })
    normalized = _normalize(raw, "2026-2027")
    assert normalized["date_sched"].notna().all()
    assert normalized.loc[0, "date_sched"] == pd.Timestamp("2026-08-21T19:00:00Z")


def _fake_bundle() -> ModelBundle:
    features = {
        "start": START_FEATURES, "start_minutes": START_MIN_FEATURES,
        "cameo": CAMEO_FEATURES, "cameo_minutes": CAMEO_MIN_FEATURES, "p60": P60_FEATURES,
    }
    values = {"start": .6, "start_minutes": 75, "cameo": .4, "cameo_minutes": 20, "p60": .5}
    heads = {}
    for head, feature_list in features.items():
        heads[head] = {
            family: HeadModel(head, family, feature_list, ConstantModel(values[head]), 100)
            for family in ("GK", "OUTFIELD")
        }
    return ModelBundle(heads=heads, seed=1)


def _prediction_rows() -> pd.DataFrame:
    rows = pd.DataFrame({column: [0.0, 0.0] for column in set(START_FEATURES + START_MIN_FEATURES + CAMEO_FEATURES + CAMEO_MIN_FEATURES) if column != "pos"})
    rows["pos"] = pd.Categorical(["GK", "MID"], categories=["GK", "DEF", "MID", "FWD"])
    rows["date_sched"] = pd.to_datetime(["2026-08-22", "2026-08-22"], utc=True)
    rows["player_id"] = ["g", "m"]
    rows["history_matches"] = [4, 5]
    rows["season_history_matches"] = [0, 0]
    rows["cold_start"] = [1, 1]
    return rows


def test_pure_mixture_schema_states_entropy_and_legacy_adapter() -> None:
    raw = HierarchicalCalibrators("raw", None, {})
    result = predict(_prediction_rows(), _fake_bundle(), CalibrationBundle(raw, raw, raw), "2026-08-21", "run")
    assert set(REQUIRED_OUTPUT_COLUMNS) <= set(result.columns)
    assert np.allclose(result["pred_exp_minutes"], .6 * 75 + .4 * .4 * 20)
    assert np.allclose(result[["p_start_state", "p_cameo_state", "p_dnp_state"]].sum(axis=1), 1)
    assert (result["state_entropy"] >= 0).all()
    validate_predictions(result)
    legacy = add_legacy_compatibility(result)
    assert legacy["pred_minutes"].equals(legacy["pred_exp_minutes"])


def test_timestamped_override_preserves_raw_and_rejects_late_knowledge() -> None:
    raw = HierarchicalCalibrators("raw", None, {})
    predictions = predict(_prediction_rows(), _fake_bundle(), CalibrationBundle(raw, raw, raw), "2026-08-21", "run")
    overrides = pd.DataFrame({
        "player_id": ["g", "m"], "confirmed_unavailable": [True, True],
        "override_reason": ["suspension", "injury"], "override_source": ["FA", "club"],
        "information_timestamp": ["2026-08-20T10:00:00Z", "2026-08-22T10:00:00Z"],
    })
    result = apply_confirmed_absence_overrides(predictions, overrides, "2026-08-21")
    assert result.loc[0, "pred_exp_minutes_raw"] > 0
    assert result.loc[0, "pred_exp_minutes_final"] == 0
    assert result.loc[0, "override_applied"]
    assert not result.loc[1, "override_applied"]


def test_deterministic_calibration_selection_and_raw_fallback() -> None:
    y = np.tile([0, 1], 100)
    raw = np.where(y == 1, .55, .45)
    folds = [CalibrationFoldData("f1", raw, y, raw, y), CalibrationFoldData("f2", raw, y, raw, y)]
    first = select_calibration(folds, _config())
    second = select_calibration(folds, _config())
    assert first.method == second.method
    assert first.fold_metrics == second.fold_metrics


def test_walk_forward_final_holdout_is_never_in_selection_folds() -> None:
    rows = []
    for season in ("2022-2023", "2023-2024", "2024-2025"):
        for gw in range(1, 25):
            rows.append({"season": season, "gw_played": gw, "date_sched": pd.Timestamp(f"{int(season[:4])+1}-01-01", tz="UTC") + pd.Timedelta(days=gw * 7)})
    folds = generate_folds(pd.DataFrame(rows), _config())
    assert_holdout_isolated(folds)
    holdout = next(fold for fold in folds if fold.is_final_holdout)
    assert len(holdout.evaluation_keys) == 6
    assert all(not fold.is_final_holdout for fold in folds[:-1])


def test_metrics_and_paired_block_bootstrap_are_complete_and_deterministic() -> None:
    metrics = probability_metrics(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]))
    assert {"auc", "brier", "prevalence_brier", "brier_skill", "log_loss", "calibration_intercept", "calibration_slope", "reliability_bins", "n"} <= set(metrics)
    assert len(metrics["reliability_bins"]) == 10
    assert state_log_loss(np.array([0, 1, 2]), np.eye(3) * .98 + (1 - np.eye(3)) * .01) < .1
    frame = pd.DataFrame({"fixture": [1, 1, 2, 2], "v2": [.1, .2, .1, .2], "v1": [.2, .3, .2, .3]})
    a = paired_block_bootstrap(frame, "v2", "v1", "fixture", 100, .95, 7)
    b = paired_block_bootstrap(frame, "v2", "v1", "fixture", 100, .95, 7)
    assert a == b
    assert acceptance_gate("p60", a, .002)["status"] == "pass"
    assert acceptance_gate("live", None, .01)["status"] == "deferred"


def test_legacy_shadow_adapter_marks_missing_v1_duration_heads_invalid(tmp_path: Path) -> None:
    raw_v1 = tmp_path / "v1.raw.csv"
    v2 = tmp_path / "v2.csv"
    output = tmp_path / "v1.csv"
    pd.DataFrame({
        "season": ["2026-2027"], "gw_orig": [1], "player_id": ["p"],
        "pred_minutes": [0.0],
    }).to_csv(raw_v1, index=False)
    pd.DataFrame({
        "season": ["2026-2027"], "gw_orig": [1], "player_id": ["p"],
        "match_id": ["m"], "team_id": ["t"], "opponent_id": ["o"],
        "player": ["Player"], "pos": ["MID"],
    }).to_csv(v2, index=False)
    v2.with_suffix(".csv.meta.json").write_text(json.dumps({
        "input_data_identifiers": {"2026-2027": {"sha256": "abc"}}
    }), encoding="utf-8")
    prepare_v1_shadow_artifact(raw_v1, v2, output, "2026-08-21T17:30:00Z")
    enriched = pd.read_csv(output)
    metadata = json.loads(output.with_suffix(".csv.meta.json").read_text(encoding="utf-8"))
    assert not enriched.loc[0, "v1_operational_valid"]
    assert not metadata["operational_valid"]
    assert "regression heads" in metadata["operational_warning"]
