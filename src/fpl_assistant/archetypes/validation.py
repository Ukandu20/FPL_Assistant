from __future__ import annotations

from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .team_ratings import TeamRatingParameters, build_pre_match_ratings


LOSS_METRICS = {"mae", "rmse", "poisson_deviance", "brier", "log_loss"}


def metric_value(actual: pd.Series, predicted: pd.Series, metric: str) -> float:
    pair = pd.DataFrame({"actual": actual, "predicted": predicted}).apply(pd.to_numeric, errors="coerce").dropna()
    if pair.empty:
        return float("nan")
    y = pair["actual"].to_numpy(float)
    prediction = pair["predicted"].to_numpy(float)
    if metric == "mae":
        return float(np.mean(np.abs(y - prediction)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y - prediction) ** 2)))
    if metric == "poisson_deviance":
        prediction = np.clip(prediction, 1e-9, None)
        terms = prediction.copy()
        positive = y > 0
        terms[positive] = (
            y[positive] * np.log(y[positive] / prediction[positive])
            - (y[positive] - prediction[positive])
        )
        return float(2 * np.mean(terms))
    if metric == "brier":
        return float(np.mean((y - np.clip(prediction, 0, 1)) ** 2))
    if metric == "log_loss":
        prediction = np.clip(prediction, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction)))
    raise ValueError(f"Unsupported validation metric: {metric}")


def walk_forward_validation_report(
    predictions: pd.DataFrame,
    *,
    target: str,
    candidate: str,
    baselines: Sequence[str],
    fold_column: str,
    metric: str,
    breakdowns: Sequence[str] = ("fpl_position", "season", "confidence_band", "evidence_level"),
    minimum_improvement: float = 0.02,
    minimum_fold_win_rate: float = 0.70,
    maximum_calibration_degradation: float = 0.01,
    calibration_column: str | None = None,
) -> dict[str, object]:
    required = {target, candidate, fold_column, *baselines}
    missing = required - set(predictions)
    if missing:
        raise KeyError(f"Validation predictions missing: {sorted(missing)}")
    if metric not in LOSS_METRICS:
        raise ValueError(f"Metric must be a loss metric, received {metric}")
    folds: list[dict[str, object]] = []
    for fold, group in predictions.groupby(fold_column, sort=True):
        candidate_loss = metric_value(group[target], group[candidate], metric)
        baseline_losses = {name: metric_value(group[target], group[name], metric) for name in baselines}
        finite = {name: loss for name, loss in baseline_losses.items() if math.isfinite(loss)}
        if not math.isfinite(candidate_loss) or not finite:
            continue
        strongest_name = min(finite, key=finite.get)
        strongest_loss = finite[strongest_name]
        improvement = (strongest_loss - candidate_loss) / strongest_loss if strongest_loss > 0 else 0.0
        folds.append(
            {
                "fold": fold, "candidate_loss": candidate_loss,
                "strongest_baseline": strongest_name, "strongest_baseline_loss": strongest_loss,
                "improvement": improvement, "won": candidate_loss < strongest_loss,
            }
        )
    candidate_loss = metric_value(predictions[target], predictions[candidate], metric)
    baseline_losses = {name: metric_value(predictions[target], predictions[name], metric) for name in baselines}
    finite = {name: loss for name, loss in baseline_losses.items() if math.isfinite(loss)}
    strongest_name = min(finite, key=finite.get)
    strongest_loss = finite[strongest_name]
    improvement = (strongest_loss - candidate_loss) / strongest_loss if strongest_loss > 0 else 0.0
    fold_win_rate = float(np.mean([bool(item["won"]) for item in folds])) if folds else 0.0

    calibration_degradation = 0.0
    if calibration_column:
        candidate_calibration = metric_value(predictions[target], predictions[candidate], "brier")
        baseline_calibration = metric_value(predictions[target], predictions[calibration_column], "brier")
        calibration_degradation = candidate_calibration - baseline_calibration
    breakdown_report: dict[str, list[dict[str, object]]] = {}
    for column in breakdowns:
        if column not in predictions:
            continue
        items: list[dict[str, object]] = []
        for value, group in predictions.groupby(column, dropna=False, sort=True):
            items.append(
                {
                    "group": None if pd.isna(value) else value,
                    "rows": int(len(group)),
                    "candidate_loss": metric_value(group[target], group[candidate], metric),
                    "baseline_losses": {name: metric_value(group[target], group[name], metric) for name in baselines},
                }
            )
        breakdown_report[column] = items
    passed = bool(
        improvement >= minimum_improvement
        and fold_win_rate >= minimum_fold_win_rate
        and calibration_degradation <= maximum_calibration_degradation
    )
    return {
        "metric": metric, "rows": int(len(predictions)), "folds": folds,
        "candidate_loss": candidate_loss, "baseline_losses": baseline_losses,
        "strongest_baseline": strongest_name, "improvement": improvement,
        "fold_win_rate": fold_win_rate,
        "calibration_degradation": calibration_degradation,
        "release_criteria": {
            "minimum_improvement": minimum_improvement,
            "minimum_fold_win_rate": minimum_fold_win_rate,
            "maximum_calibration_degradation": maximum_calibration_degradation,
        },
        "passed": passed, "breakdowns": breakdown_report,
    }


def team_parameter_grid(config: Mapping[str, object]) -> Iterable[TeamRatingParameters]:
    names = (
        "season_retention", "prior_equivalent_matches", "xg_weight",
        "elo_update_k", "elo_home_advantage_points", "home_log_effect",
    )
    values = [config[f"{name}_grid"] for name in names]
    for combination in itertools.product(*values):
        selected = dict(zip(names, map(float, combination)))
        yield TeamRatingParameters(
            **selected,
            information_floor=float(config.get("information_floor", 1e-6)),
            loss_floor=float(config.get("loss_floor", 1e-8)),
        )


def _team_targets(matches: pd.DataFrame) -> pd.DataFrame:
    optional = [column for column in ("season", "gameweek", "kickoff_utc") if column in matches]
    home = matches[["match_id", "home_team_id", "away_team_id", "home_xg", "away_xg", "home_goals", "away_goals", *optional]].rename(
        columns={"home_team_id": "team_id", "away_team_id": "opponent_id", "home_xg": "actual_xg", "away_xg": "opponent_xg", "home_goals": "actual_goals", "away_goals": "opponent_goals"}
    ).assign(venue="Home")
    away = matches[["match_id", "away_team_id", "home_team_id", "away_xg", "home_xg", "away_goals", "home_goals", *optional]].rename(
        columns={"away_team_id": "team_id", "home_team_id": "opponent_id", "away_xg": "actual_xg", "home_xg": "opponent_xg", "away_goals": "actual_goals", "home_goals": "opponent_goals"}
    ).assign(venue="Away")
    return pd.concat([home, away], ignore_index=True)


def build_team_validation_predictions(
    matches: pd.DataFrame,
    *,
    parameters: TeamRatingParameters,
    rolling_matches: int = 5,
) -> pd.DataFrame:
    snapshots = build_pre_match_ratings(matches, parameters=parameters)
    targets = _team_targets(matches)
    targets["kickoff_utc"] = pd.to_datetime(targets["kickoff_utc"], utc=True, errors="raise")
    targets = targets.sort_values(["kickoff_utc", "match_id", "venue"], kind="stable")
    for metric in ("actual_xg", "opponent_xg", "actual_goals", "opponent_goals"):
        targets[f"rolling_{metric}"] = targets.groupby("team_id", sort=False)[metric].transform(
            lambda series: series.shift(1).rolling(rolling_matches, min_periods=1).mean()
        )
    prediction = snapshots.merge(
        targets,
        on=["match_id", "team_id", "opponent_id", "venue", "kickoff_utc"],
        validate="one_to_one",
    )
    opponent_elo = prediction[["match_id", "team_id", "overall_elo_pre_match"]].rename(
        columns={"team_id": "opponent_id", "overall_elo_pre_match": "opponent_elo_pre_match"}
    )
    prediction = prediction.merge(opponent_elo, on=["match_id", "opponent_id"], validate="one_to_one")
    prediction["league_average_xg"] = prediction["league_xg_baseline_pre_match"]
    prediction["league_average_goals"] = prediction["league_goals_baseline_pre_match"]
    prediction["recent_xg_xga"] = (
        prediction["rolling_actual_xg"] + prediction["rolling_opponent_xg"]
    ).div(2).fillna(prediction["league_average_xg"])
    prediction["recent_goals_conceded"] = (
        prediction["rolling_actual_goals"] + prediction["rolling_opponent_goals"]
    ).div(2).fillna(prediction["league_average_goals"])
    home_points = np.where(prediction["venue"].eq("Home"), parameters.elo_home_advantage_points, 0.0)
    elo_delta = prediction["overall_elo_pre_match"] - prediction["opponent_elo_pre_match"] + home_points
    prediction["elo_only_xg"] = prediction["league_average_xg"] * np.exp(np.log(10) * elo_delta / 800.0)
    season = prediction["season"].astype(str) if "season" in prediction else pd.Series("", index=prediction.index)
    gameweek = prediction["gameweek"].astype(str) if "gameweek" in prediction else pd.Series("", index=prediction.index)
    prediction["fold"] = season + "|" + gameweek
    return prediction


def select_team_rating_parameters(
    matches: pd.DataFrame,
    candidates: Iterable[TeamRatingParameters],
) -> tuple[TeamRatingParameters, pd.DataFrame]:
    targets = _team_targets(matches)
    rows: list[dict[str, object]] = []
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("At least one team-rating parameter candidate is required")
    for index, parameters in enumerate(candidate_list):
        snapshots = build_pre_match_ratings(matches, parameters=parameters)
        evaluated = snapshots.merge(targets, on=["match_id", "venue"], validate="one_to_one")
        xg_mae = metric_value(evaluated["actual_xg"], evaluated["expected_xg_pre_match"], "mae")
        goal_deviance = metric_value(evaluated["actual_goals"], evaluated["expected_goals_pre_match"], "poisson_deviance")
        baseline_xg = metric_value(evaluated["actual_xg"], evaluated["league_xg_baseline_pre_match"], "mae")
        baseline_goals = metric_value(evaluated["actual_goals"], evaluated["league_goals_baseline_pre_match"], "poisson_deviance")
        normalized = 0.70 * xg_mae / max(baseline_xg, 1e-9) + 0.30 * goal_deviance / max(baseline_goals, 1e-9)
        rows.append(
            {"candidate_index": index, "normalized_loss": normalized, "xg_mae": xg_mae,
             "goal_poisson_deviance": goal_deviance, **asdict(parameters)}
        )
    results = pd.DataFrame(rows).sort_values(["normalized_loss", "candidate_index"], kind="stable").reset_index(drop=True)
    return candidate_list[int(results.iloc[0]["candidate_index"])], results


def write_validation_artifacts(
    report: Mapping[str, object],
    selected_parameters: Mapping[str, object],
    *,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / "walk_forward_report.json"
    parameters_path = target / "selected_parameters.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    parameters_path.write_text(json.dumps(selected_parameters, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report_path, parameters_path


__all__ = [
    "build_team_validation_predictions", "metric_value", "select_team_rating_parameters", "team_parameter_grid",
    "walk_forward_validation_report", "write_validation_artifacts",
]
