from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TeamRatingParameters:
    season_retention: float = 0.70
    prior_equivalent_matches: float = 8.0
    xg_weight: float = 0.70
    elo_update_k: float = 20.0
    elo_home_advantage_points: float = 36.0
    home_log_effect: float = 0.15
    information_floor: float = 1e-6
    loss_floor: float = 1e-8


@dataclass
class TeamRating:
    elo: float = 1500.0
    attack: float = 0.0
    defence: float = 0.0


def regress_for_new_season(
    ratings: Mapping[str, TeamRating],
    *,
    retained: float = 0.70,
    promoted_teams: Iterable[str] = (),
) -> dict[str, TeamRating]:
    promoted = set(promoted_teams)
    result: dict[str, TeamRating] = {}
    for team_id, rating in ratings.items():
        if team_id in promoted:
            result[team_id] = TeamRating()
        else:
            result[team_id] = TeamRating(
                # The catalogue's 70/30 season regression is specified for
                # specialist attack/defence ratings, not overall Elo.
                elo=rating.elo,
                attack=retained * rating.attack,
                defence=retained * rating.defence,
            )
    return result


def elo_expected(home_elo: float, away_elo: float, home_advantage: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(home_elo - away_elo + home_advantage) / 400.0))


def _result_score(home_goals: float, away_goals: float) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def _prediction(
    attack: float,
    opposition_defence: float,
    baseline: float,
    home_effect: float,
) -> float:
    return math.exp(baseline + home_effect + attack - opposition_defence)


def _loss_gradient_hessian(
    *,
    attack: float,
    defence: float,
    observed_xg: float,
    observed_goals: float,
    xg_baseline: float,
    goal_baseline: float,
    home_effect: float,
    xg_loss_baseline: float,
    goal_loss_baseline: float,
    xg_weight: float,
    loss_floor: float,
) -> tuple[float, float]:
    lambda_xg = _prediction(attack, defence, xg_baseline, home_effect)
    lambda_goal = _prediction(attack, defence, goal_baseline, home_effect)
    predicted_log = math.log1p(lambda_xg)
    observed_log = math.log1p(max(0.0, observed_xg))
    residual = predicted_log - observed_log
    derivative = lambda_xg / (1.0 + lambda_xg)
    xg_scale = max(float(xg_loss_baseline), loss_floor)
    goal_scale = max(float(goal_loss_baseline), loss_floor)
    goal_weight = 1.0 - xg_weight
    gradient = (
        xg_weight * residual * derivative / xg_scale
        + goal_weight * (lambda_goal - observed_goals) / goal_scale
    )
    hessian = (
        xg_weight * derivative * derivative / xg_scale
        + goal_weight * lambda_goal / goal_scale
    )
    return gradient, max(hessian, loss_floor)


def _center_identifiable(ratings: Mapping[str, TeamRating]) -> None:
    if not ratings:
        return
    attack_mean = float(np.mean([rating.attack for rating in ratings.values()]))
    defence_mean = float(np.mean([rating.defence for rating in ratings.values()]))
    for rating in ratings.values():
        rating.attack -= attack_mean
        rating.defence -= defence_mean


def update_after_match(
    ratings: dict[str, TeamRating],
    match: Mapping[str, object],
    *,
    xg_log_baseline: float,
    goal_log_baseline: float,
    xg_loss_baseline: float = 1.0,
    goal_loss_baseline: float = 1.0,
    information: Mapping[str, float] | None = None,
    parameters: TeamRatingParameters = TeamRatingParameters(),
    iterations: int = 8,
) -> None:
    home_id = str(match["home_team_id"])
    away_id = str(match["away_team_id"])
    home = ratings.setdefault(home_id, TeamRating())
    away = ratings.setdefault(away_id, TeamRating())
    home_goals = float(match["home_goals"])
    away_goals = float(match["away_goals"])
    home_xg = float(match["home_xg"])
    away_xg = float(match["away_xg"])

    expected = elo_expected(home.elo, away.elo, parameters.elo_home_advantage_points)
    result = _result_score(home_goals, away_goals)
    change = parameters.elo_update_k * (result - expected)
    home.elo += change
    away.elo -= change

    prior = {
        "home_attack": home.attack,
        "away_attack": away.attack,
        "home_defence": home.defence,
        "away_defence": away.defence,
    }
    values = dict(prior)
    info = information or {}
    for _ in range(iterations):
        home_gradient, home_hessian = _loss_gradient_hessian(
            attack=values["home_attack"],
            defence=values["away_defence"],
            observed_xg=home_xg,
            observed_goals=home_goals,
            xg_baseline=xg_log_baseline,
            goal_baseline=goal_log_baseline,
            home_effect=parameters.home_log_effect,
            xg_loss_baseline=xg_loss_baseline,
            goal_loss_baseline=goal_loss_baseline,
            xg_weight=parameters.xg_weight,
            loss_floor=parameters.loss_floor,
        )
        away_gradient, away_hessian = _loss_gradient_hessian(
            attack=values["away_attack"],
            defence=values["home_defence"],
            observed_xg=away_xg,
            observed_goals=away_goals,
            xg_baseline=xg_log_baseline,
            goal_baseline=goal_log_baseline,
            home_effect=0.0,
            xg_loss_baseline=xg_loss_baseline,
            goal_loss_baseline=goal_loss_baseline,
            xg_weight=parameters.xg_weight,
            loss_floor=parameters.loss_floor,
        )
        for attack_key, defence_key, gradient, hessian in (
            ("home_attack", "away_defence", home_gradient, home_hessian),
            ("away_attack", "home_defence", away_gradient, away_hessian),
        ):
            attack_info = max(float(info.get(attack_key, 1.0)), parameters.information_floor)
            defence_info = max(float(info.get(defence_key, 1.0)), parameters.information_floor)
            attack_penalty = parameters.prior_equivalent_matches * attack_info
            defence_penalty = parameters.prior_equivalent_matches * defence_info
            attack_total_gradient = gradient + attack_penalty * (values[attack_key] - prior[attack_key])
            defence_total_gradient = -gradient + defence_penalty * (values[defence_key] - prior[defence_key])
            values[attack_key] -= attack_total_gradient / (hessian + attack_penalty)
            values[defence_key] -= defence_total_gradient / (hessian + defence_penalty)

    home.attack = values["home_attack"]
    away.attack = values["away_attack"]
    home.defence = values["home_defence"]
    away.defence = values["away_defence"]
    _center_identifiable(ratings)


def build_pre_match_ratings(
    matches: pd.DataFrame,
    *,
    initial_ratings: Mapping[str, TeamRating] | None = None,
    parameters: TeamRatingParameters = TeamRatingParameters(),
    xg_loss_baseline: float = 1.0,
    goal_loss_baseline: float = 1.0,
    initial_league_xg: float = 1.35,
    initial_league_goals: float = 1.35,
    baseline_prior_team_matches: float = 20.0,
) -> pd.DataFrame:
    required = {
        "match_id", "kickoff_utc", "home_team_id", "away_team_id",
        "home_goals", "away_goals", "home_xg", "away_xg",
    }
    missing = required - set(matches)
    if missing:
        raise KeyError(f"Team matches missing: {sorted(missing)}")
    work = matches.copy()
    work["kickoff_utc"] = pd.to_datetime(work["kickoff_utc"], utc=True, errors="raise")
    work = work.sort_values(["kickoff_utc", "match_id"], kind="stable")
    ratings = {
        str(team): TeamRating(value.elo, value.attack, value.defence)
        for team, value in (initial_ratings or {}).items()
    }
    team_ids = set(work["home_team_id"].astype(str)) | set(work["away_team_id"].astype(str))
    for team_id in team_ids:
        ratings.setdefault(team_id, TeamRating())

    xg_total = initial_league_xg * baseline_prior_team_matches
    goal_total = initial_league_goals * baseline_prior_team_matches
    team_match_count = baseline_prior_team_matches
    rolling_xg_loss = float(xg_loss_baseline)
    rolling_goal_loss = float(goal_loss_baseline)
    loss_match_count = 1.0
    rows: list[dict[str, object]] = []
    active_season: str | None = None
    previous_season_teams: set[str] = set()
    for match in work.to_dict("records"):
        season = str(match.get("season", ""))
        if season and season != active_season:
            season_series = work["season"].astype(str)
            current_teams = set(
                work.loc[season_series.eq(season), "home_team_id"].astype(str)
            ) | set(work.loc[season_series.eq(season), "away_team_id"].astype(str))
            if active_season is not None:
                promoted = current_teams - previous_season_teams
                regressed = regress_for_new_season(
                    ratings,
                    retained=parameters.season_retention,
                    promoted_teams=promoted,
                )
                ratings = {
                    team_id: regressed.get(team_id, TeamRating())
                    for team_id in current_teams
                }
                established = [ratings[team_id] for team_id in current_teams - promoted]
                if established:
                    attack_shift = sum(item.attack for item in established) / len(established)
                    defence_shift = sum(item.defence for item in established) / len(established)
                    for item in established:
                        item.attack -= attack_shift
                        item.defence -= defence_shift
            else:
                for team_id in current_teams:
                    ratings.setdefault(team_id, TeamRating())
            previous_season_teams = current_teams
            active_season = season
        league_xg = max(xg_total / team_match_count, parameters.loss_floor)
        league_goals = max(goal_total / team_match_count, parameters.loss_floor)
        xg_log_baseline = math.log(league_xg)
        goal_log_baseline = math.log(league_goals)
        home_id = str(match["home_team_id"])
        away_id = str(match["away_team_id"])
        home = ratings[home_id]
        away = ratings[away_id]
        expected_home_xg = _prediction(
            home.attack, away.defence, xg_log_baseline, parameters.home_log_effect
        )
        expected_away_xg = _prediction(away.attack, home.defence, xg_log_baseline, 0.0)
        expected_home_goals = _prediction(
            home.attack, away.defence, goal_log_baseline, parameters.home_log_effect
        )
        expected_away_goals = _prediction(away.attack, home.defence, goal_log_baseline, 0.0)
        rows.extend(
            [
                {
                    "match_id": match["match_id"], "kickoff_utc": match["kickoff_utc"],
                    "team_id": home_id, "opponent_id": away_id, "venue": "Home",
                    "overall_elo_pre_match": home.elo, "attack_rating_pre_match": home.attack,
                    "defence_rating_pre_match": home.defence,
                    "attack_index_pre_match": math.exp(home.attack),
                    "defence_index_pre_match": math.exp(home.defence),
                    "expected_xg_pre_match": expected_home_xg,
                    "expected_goals_pre_match": expected_home_goals,
                    "league_xg_baseline_pre_match": league_xg,
                    "league_goals_baseline_pre_match": league_goals,
                },
                {
                    "match_id": match["match_id"], "kickoff_utc": match["kickoff_utc"],
                    "team_id": away_id, "opponent_id": home_id, "venue": "Away",
                    "overall_elo_pre_match": away.elo, "attack_rating_pre_match": away.attack,
                    "defence_rating_pre_match": away.defence,
                    "attack_index_pre_match": math.exp(away.attack),
                    "defence_index_pre_match": math.exp(away.defence),
                    "expected_xg_pre_match": expected_away_xg,
                    "expected_goals_pre_match": expected_away_goals,
                    "league_xg_baseline_pre_match": league_xg,
                    "league_goals_baseline_pre_match": league_goals,
                },
            ]
        )
        update_after_match(
            ratings,
            match,
            xg_log_baseline=xg_log_baseline,
            goal_log_baseline=goal_log_baseline,
            xg_loss_baseline=rolling_xg_loss,
            goal_loss_baseline=rolling_goal_loss,
            parameters=parameters,
        )
        observed_xg = (float(match["home_xg"]), float(match["away_xg"]))
        observed_goals = (float(match["home_goals"]), float(match["away_goals"]))
        baseline_xg_loss = 0.5 * sum(
            (math.log1p(value) - math.log1p(league_xg)) ** 2 for value in observed_xg
        )
        baseline_goal_loss = 0.5 * sum(
            league_goals - value * math.log(league_goals) for value in observed_goals
        )
        rolling_xg_loss = (
            rolling_xg_loss * loss_match_count + baseline_xg_loss
        ) / (loss_match_count + 1.0)
        rolling_goal_loss = (
            rolling_goal_loss * loss_match_count + baseline_goal_loss
        ) / (loss_match_count + 1.0)
        loss_match_count += 1.0
        xg_total += sum(observed_xg)
        goal_total += sum(observed_goals)
        team_match_count += 2.0
    return pd.DataFrame(rows)


__all__ = [
    "TeamRating", "TeamRatingParameters", "build_pre_match_ratings",
    "elo_expected", "regress_for_new_season", "update_after_match",
]
