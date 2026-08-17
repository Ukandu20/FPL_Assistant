from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .adapters import understat_team_rows_to_matches
from .config import load_config
from .validation import (
    build_team_validation_predictions,
    select_team_rating_parameters,
    walk_forward_validation_report,
    write_validation_artifacts,
)
from .team_ratings import TeamRatingParameters


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _default_team_match_paths() -> list[Path]:
    root = PROJECT_ROOT / "data" / "processed" / "understat" / "ENG-Premier League"
    return sorted(root.glob("*/team_match.csv"))


def _coordinate_candidates(config: dict[str, object]) -> list[TeamRatingParameters]:
    default = TeamRatingParameters(
        season_retention=float(config["season_retention"]),
        prior_equivalent_matches=float(config["prior_equivalent_matches"]),
        xg_weight=float(config["xg_weight"]),
        elo_update_k=float(config["elo_update_k"]),
        elo_home_advantage_points=float(config["elo_home_advantage_points"]),
        home_log_effect=float(config["home_log_effect"]),
        information_floor=float(config["information_floor"]),
        loss_floor=float(config["loss_floor"]),
    )
    candidates = [default]
    for field in (
        "season_retention", "prior_equivalent_matches", "xg_weight", "elo_update_k",
        "elo_home_advantage_points", "home_log_effect",
    ):
        for value in config[f"{field}_grid"]:
            candidates.append(replace(default, **{field: float(value)}))
    unique: dict[tuple[tuple[str, object], ...], TeamRatingParameters] = {}
    for candidate in candidates:
        unique[tuple(sorted(asdict(candidate).items()))] = candidate
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chronological V1 team-rating validation.")
    parser.add_argument("--team-match", action="append", default=[])
    parser.add_argument(
        "--output-directory",
        default="artifacts/archetypes/1.0.0/validation/team_ratings",
    )
    args = parser.parse_args()
    paths = [Path(path) for path in args.team_match] or _default_team_match_paths()
    if not paths:
        raise FileNotFoundError("No processed Understat team_match.csv files were found")
    team_rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    matches = understat_team_rows_to_matches(team_rows)
    seasons = sorted(matches["season"].unique())
    if len(seasons) < 2:
        raise ValueError("At least two seasons are required for chronological validation")
    held_out_season = seasons[-1]
    training = matches[matches["season"].ne(held_out_season)].copy()
    held_out = matches[matches["season"].eq(held_out_season)].copy()

    config = load_config()
    selected, candidates = select_team_rating_parameters(
        training, _coordinate_candidates(dict(config.section("team_ratings")))
    )
    predictions = build_team_validation_predictions(held_out, parameters=selected)
    xg_report = walk_forward_validation_report(
        predictions,
        target="actual_xg", candidate="expected_xg_pre_match",
        baselines=["league_average_xg", "recent_xg_xga", "elo_only_xg"],
        fold_column="fold", metric="mae", breakdowns=("season", "venue"),
    )
    goal_report = walk_forward_validation_report(
        predictions,
        target="actual_goals", candidate="expected_goals_pre_match",
        baselines=["league_average_goals", "recent_goals_conceded"],
        fold_column="fold", metric="poisson_deviance", breakdowns=("season", "venue"),
    )
    minimum_training_seasons = int(config.section("validation")["minimum_training_seasons"])
    training_seasons = len(set(training["season"]))
    release_passed = bool(
        xg_report["passed"] and goal_report["passed"]
        and training_seasons >= minimum_training_seasons
    )
    report = {
        "model_version": config.model_version,
        "provider": "understat",
        "seasons": seasons,
        "training_seasons": sorted(training["season"].unique()),
        "held_out_season": held_out_season,
        "minimum_training_seasons": minimum_training_seasons,
        "selected_from_coordinate_grid": True,
        "candidate_count": int(len(candidates)),
        "xg": xg_report,
        "goals": goal_report,
        "release_passed": release_passed,
        "limitations": (
            [] if training_seasons >= minimum_training_seasons else
            [f"Only {training_seasons} complete training season(s) were available; {minimum_training_seasons} are required."]
        ),
    }
    output = Path(args.output_directory)
    report_path, parameters_path = write_validation_artifacts(
        report, asdict(selected), output_directory=output
    )
    candidates.to_csv(output / "parameter_candidates.csv", index=False)
    predictions.to_csv(output / "held_out_predictions.csv", index=False)
    print(f"Validation report: {report_path}")
    print(f"Selected parameters: {parameters_path}")
    print(f"Release passed: {release_passed}")


if __name__ == "__main__":
    main()
