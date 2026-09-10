from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

import pandas as pd


DEFAULT_LEAGUE = "ENG-Premier League"
DEFAULT_SEASON = "2026-2027"
PROVIDER = "clubelo_fallback"


class FallbackError(ValueError):
    """Raised when fallback inputs or invariants are unsafe."""


@dataclass(frozen=True)
class FallbackConfig:
    formula_version: str = "clubelo_fallback_v1"
    k: float = 20.0
    hfa: float = 36.0
    elo_divisor: float = 400.0
    margin_mode: str = "base"
    margin_normalizer: float | None = None

    def validate(self) -> None:
        for name, value in (("k", self.k), ("elo_divisor", self.elo_divisor)):
            if not math.isfinite(value) or value <= 0:
                raise FallbackError(f"{name} must be finite and > 0")
        if not math.isfinite(self.hfa):
            raise FallbackError("hfa must be finite")
        if self.margin_mode not in {"base", "calibrated_margin"}:
            raise FallbackError("margin_mode must be 'base' or 'calibrated_margin'")
        if self.margin_mode == "calibrated_margin":
            if self.margin_normalizer is None:
                raise FallbackError(
                    "calibrated_margin requires an explicit, versioned margin_normalizer"
                )
            if not math.isfinite(self.margin_normalizer) or self.margin_normalizer <= 0:
                raise FallbackError("margin_normalizer must be finite and > 0")
        elif self.margin_normalizer is not None:
            raise FallbackError("margin_normalizer is only valid in calibrated_margin mode")


@dataclass(frozen=True)
class FallbackResult:
    ledger: pd.DataFrame
    team_history: pd.DataFrame
    schedule: pd.DataFrame
    audit: dict[str, Any]


def elo_expected(
    home_elo: float,
    away_elo: float,
    *,
    hfa: float = 36.0,
    divisor: float = 400.0,
) -> float:
    if not all(math.isfinite(v) for v in (home_elo, away_elo, hfa, divisor)):
        raise FallbackError("Elo inputs must be finite")
    if divisor <= 0:
        raise FallbackError("Elo divisor must be > 0")
    return 1.0 / (1.0 + 10.0 ** (-(home_elo - away_elo + hfa) / divisor))


def _actual_home(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def _margin_factor(home_goals: int, away_goals: int, config: FallbackConfig) -> float:
    margin = abs(home_goals - away_goals)
    if margin == 0 or config.margin_mode == "base":
        return 1.0
    assert config.margin_normalizer is not None
    return math.sqrt(float(margin)) / config.margin_normalizer


def _clean_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise FallbackError(f"{label} missing required columns: {missing}")


def _bool_series(series: pd.Series, label: str) -> pd.Series:
    values = _clean_string(series).str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    invalid = sorted(set(values) - set(mapping))
    if invalid:
        raise FallbackError(f"{label} contains invalid boolean values: {invalid}")
    return values.map(mapping).astype(bool)


def _parse_as_of(value: str | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise FallbackError("as_of is not a valid timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_match_id(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Accept older enriched schedules whose canonical key was named fbref_id."""
    output = frame.copy()
    if "match_id" not in output:
        if "fbref_id" not in output:
            raise FallbackError(f"{label} missing required match_id")
        output["match_id"] = output["fbref_id"]
    elif "fbref_id" in output:
        match_ids = _clean_string(output["match_id"])
        fbref_ids = _clean_string(output["fbref_id"])
        conflicts = match_ids.ne("") & fbref_ids.ne("") & match_ids.ne(fbref_ids)
        if conflicts.any():
            raise FallbackError(f"{label} contains conflicting match_id and fbref_id values")
        output["match_id"] = match_ids.mask(match_ids.eq(""), fbref_ids)
    return output


def prepare_anchor_and_fixtures(
    schedule: pd.DataFrame,
    *,
    expected_team_count: int | None = 20,
    expected_fixture_count: int | None = 380,
    expected_anchor_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "match_id",
        "team",
        "team_id",
        "opponent_id",
        "is_home",
        "home_id",
        "away_id",
        "date_sched",
        "elo_preseason",
        "elo_preseason_as_of",
    }
    _require_columns(schedule, required, "anchor schedule")
    work = schedule.copy()
    for column in ("match_id", "team", "team_id", "opponent_id", "home_id", "away_id"):
        work[column] = _clean_string(work[column])
        if work[column].eq("").any():
            raise FallbackError(f"anchor schedule contains blank {column}")
    work["is_home"] = pd.to_numeric(work["is_home"], errors="coerce")
    if work["is_home"].isna().any() or not work["is_home"].isin([0, 1]).all():
        raise FallbackError("anchor schedule is_home must contain only 0/1")
    work["elo_preseason"] = pd.to_numeric(work["elo_preseason"], errors="coerce")
    if work["elo_preseason"].isna().any() or not work["elo_preseason"].map(math.isfinite).all():
        raise FallbackError("anchor schedule contains missing or non-finite preseason Elo")
    work["anchor_date"] = pd.to_datetime(
        work["elo_preseason_as_of"], errors="coerce", utc=True
    ).dt.normalize()
    if work["anchor_date"].isna().any():
        raise FallbackError("anchor schedule contains invalid elo_preseason_as_of")

    anchor_columns = ["team", "team_id", "elo_preseason", "anchor_date"]
    distinct_anchors = work[anchor_columns].drop_duplicates()
    conflicts = distinct_anchors.groupby("team_id", sort=False).size()
    if conflicts.gt(1).any():
        raise FallbackError(
            "conflicting preseason anchors for team_ids: "
            + ", ".join(conflicts[conflicts.gt(1)].index.astype(str))
        )
    anchors = distinct_anchors.drop_duplicates("team_id").sort_values("team_id").reset_index(drop=True)
    if anchors["team"].duplicated().any():
        raise FallbackError("one team code maps to multiple team_ids in anchor schedule")
    if expected_team_count is not None and len(anchors) != expected_team_count:
        raise FallbackError(
            f"expected {expected_team_count} anchored teams, found {len(anchors)}"
        )
    anchor_dates = anchors["anchor_date"].drop_duplicates()
    if len(anchor_dates) != 1:
        raise FallbackError("all teams must share one preseason anchor date")
    if expected_anchor_date is not None:
        expected = _parse_as_of(expected_anchor_date).normalize()
        if anchor_dates.iloc[0] != expected:
            raise FallbackError(
                f"expected anchor date {expected.date()}, found {anchor_dates.iloc[0].date()}"
            )

    if work.duplicated(["match_id", "team_id"]).any():
        raise FallbackError("anchor schedule contains duplicate match_id + team_id rows")
    fixtures: list[dict[str, Any]] = []
    for match_id, group in work.groupby("match_id", sort=False):
        if len(group) != 2 or set(group["is_home"].astype(int)) != {0, 1}:
            raise FallbackError(f"fixture {match_id} must contain exactly one home and one away row")
        home_row = group.loc[group["is_home"].eq(1)].iloc[0]
        away_row = group.loc[group["is_home"].eq(0)].iloc[0]
        home_id = str(home_row["team_id"])
        away_id = str(away_row["team_id"])
        if home_id == away_id:
            raise FallbackError(f"fixture {match_id} has the same home and away team")
        if (
            str(home_row["opponent_id"]) != away_id
            or str(away_row["opponent_id"]) != home_id
            or not group["home_id"].eq(home_id).all()
            or not group["away_id"].eq(away_id).all()
        ):
            raise FallbackError(f"fixture {match_id} has inconsistent mirrored team identities")
        dates = pd.to_datetime(group["date_sched"], errors="coerce", utc=True).dt.normalize()
        if dates.isna().any() or dates.nunique() != 1:
            raise FallbackError(f"fixture {match_id} has an invalid or conflicting scheduled date")
        fixtures.append(
            {
                "match_id": str(match_id),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "date_sched": dates.iloc[0],
            }
        )
    fixture_frame = pd.DataFrame(fixtures).sort_values(["date_sched", "match_id"]).reset_index(drop=True)
    if expected_fixture_count is not None and len(fixture_frame) != expected_fixture_count:
        raise FallbackError(
            f"expected {expected_fixture_count} fixtures, found {len(fixture_frame)}"
        )
    anchor_ids = set(anchors["team_id"])
    fixture_ids = set(fixture_frame["home_team_id"]) | set(fixture_frame["away_team_id"])
    if fixture_ids != anchor_ids:
        raise FallbackError("fixture team set does not exactly match preseason anchor team set")
    return anchors, fixture_frame


def prepare_completed_matches(
    results: pd.DataFrame,
    fixtures: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    reject_missing_past_results: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    required = {
        "match_id",
        "game_date",
        "team",
        "team_id",
        "opp",
        "opp_id",
        "venue",
        "team_goals",
        "opp_goals",
        "is_result",
        "has_data",
    }
    _require_columns(results, required, "results schedule")
    work = results.copy()
    for column in ("match_id", "team", "team_id", "opp", "opp_id", "venue"):
        work[column] = _clean_string(work[column])
    result_flag = _bool_series(work["is_result"], "is_result")
    data_flag = _bool_series(work["has_data"], "has_data")
    has_score = (
        pd.to_numeric(work["team_goals"], errors="coerce").notna()
        | pd.to_numeric(work["opp_goals"], errors="coerce").notna()
    )
    candidate_ids = set(work.loc[result_flag | data_flag | has_score, "match_id"])
    candidate_ids.discard("")
    candidates = work.loc[work["match_id"].isin(candidate_ids)].copy()
    candidates["_is_result"] = result_flag.loc[candidates.index]
    candidates["_has_data"] = data_flag.loc[candidates.index]

    fixture_lookup = fixtures.set_index("match_id").to_dict("index")
    anchor_ids = set(anchors["team_id"])
    cutoff = _parse_as_of(as_of)
    anchor_time = anchors["anchor_date"].iloc[0]
    if cutoff < anchor_time:
        raise FallbackError("as_of cannot be earlier than the preseason anchor")

    matches: list[dict[str, Any]] = []
    for match_id, group in candidates.groupby("match_id", sort=False):
        match_id = str(match_id)
        if match_id not in fixture_lookup:
            raise FallbackError(f"completed result {match_id} is not in the anchor fixture schedule")
        if len(group) != 2 or group.duplicated("team_id").any():
            raise FallbackError(f"result {match_id} must contain exactly two unique team-side rows")
        if not group["_is_result"].all() or not group["_has_data"].all():
            raise FallbackError(f"result {match_id} has incomplete result/data flags")
        venues = group["venue"].str.upper()
        if set(venues) != {"H", "A"}:
            raise FallbackError(f"result {match_id} must contain exactly one H and one A row")
        home = group.loc[venues.eq("H")].iloc[0]
        away = group.loc[venues.eq("A")].iloc[0]
        home_id, away_id = str(home["team_id"]), str(away["team_id"])
        if home_id not in anchor_ids or away_id not in anchor_ids:
            raise FallbackError(f"result {match_id} contains a team outside the anchor set")
        fixture = fixture_lookup[match_id]
        if home_id != fixture["home_team_id"] or away_id != fixture["away_team_id"]:
            raise FallbackError(f"result {match_id} teams disagree with the anchor fixture")
        if str(home["opp_id"]) != away_id or str(away["opp_id"]) != home_id:
            raise FallbackError(f"result {match_id} opponent IDs do not mirror")

        scores = pd.to_numeric(
            pd.Series([home["team_goals"], home["opp_goals"], away["team_goals"], away["opp_goals"]]),
            errors="coerce",
        )
        if scores.isna().any() or (~scores.map(math.isfinite)).any() or scores.lt(0).any():
            raise FallbackError(f"result {match_id} contains invalid goals")
        if not scores.map(lambda value: float(value).is_integer()).all():
            raise FallbackError(f"result {match_id} contains fractional goals")
        home_goals, away_goals = int(scores.iloc[0]), int(scores.iloc[1])
        if int(scores.iloc[2]) != away_goals or int(scores.iloc[3]) != home_goals:
            raise FallbackError(f"result {match_id} scores do not mirror")

        dates = _clean_string(group["game_date"])
        times = (
            _clean_string(group["game_time"])
            if "game_time" in group
            else pd.Series("00:00:00", index=group.index, dtype="string")
        )
        datetimes = pd.to_datetime(
            dates + " " + times.mask(times.eq(""), "00:00:00"), errors="coerce", utc=True
        )
        if datetimes.isna().any() or datetimes.nunique() != 1:
            raise FallbackError(f"result {match_id} has an invalid or conflicting kickoff")
        kickoff = datetimes.iloc[0]
        if kickoff <= anchor_time:
            raise FallbackError(f"result {match_id} is not later than the preseason anchor")
        if kickoff > cutoff:
            raise FallbackError(f"result {match_id} is later than as_of")

        round_value: Any = pd.NA
        round_conflict = False
        if "round" in group:
            rounds = pd.to_numeric(group["round"], errors="coerce").dropna().unique()
            if len(rounds) > 1:
                # Round is descriptive and does not affect rating order. Preserve
                # the match but refuse to publish a guessed gameweek value.
                round_conflict = True
            elif len(rounds) == 1:
                round_value = int(rounds[0])
        home_xg = pd.to_numeric(pd.Series([home.get("team_xg", pd.NA)]), errors="coerce").iloc[0]
        away_xg = pd.to_numeric(pd.Series([away.get("team_xg", pd.NA)]), errors="coerce").iloc[0]
        matches.append(
            {
                "match_id": match_id,
                "kickoff_utc": kickoff,
                "game_date": kickoff.date().isoformat(),
                "round": round_value,
                "round_conflict": round_conflict,
                "home_team": str(home["team"]),
                "away_team": str(away["team"]),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_xg": home_xg,
                "away_xg": away_xg,
            }
        )

    completed = pd.DataFrame(matches)
    if completed.empty:
        completed = pd.DataFrame(
            columns=[
                "match_id", "kickoff_utc", "game_date", "round", "round_conflict", "home_team", "away_team",
                "home_team_id", "away_team_id", "home_goals", "away_goals", "home_xg", "away_xg",
            ]
        )
    else:
        completed = completed.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)

    past_fixture_ids = set(
        fixtures.loc[fixtures["date_sched"].lt(cutoff.normalize()), "match_id"].astype(str)
    )
    completed_ids = set(completed["match_id"].astype(str))
    missing_past = sorted(past_fixture_ids - completed_ids)
    if reject_missing_past_results and missing_past:
        sample = ", ".join(missing_past[:5])
        raise FallbackError(
            f"{len(missing_past)} past fixtures have no accepted result (sample: {sample})"
        )
    return completed, missing_past


def replay_matches(
    anchors: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    config: FallbackConfig = FallbackConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    config.validate()
    ratings = dict(zip(anchors["team_id"].astype(str), anchors["elo_preseason"].astype(float)))
    initial_total = math.fsum(ratings.values())
    team_code = dict(zip(anchors["team_id"].astype(str), anchors["team"].astype(str)))
    anchor_at = anchors["anchor_date"].iloc[0]
    history_rows: list[dict[str, Any]] = [
        {
            "league": DEFAULT_LEAGUE,
            "season": DEFAULT_SEASON,
            "team": team_code[team_id],
            "team_id": team_id,
            "effective_at_utc": anchor_at.isoformat(),
            "event_type": "preseason_anchor",
            "match_id": "",
            "opponent_id": "",
            "elo": elo,
            "elo_provider": PROVIDER,
            "formula_version": config.formula_version,
        }
        for team_id, elo in sorted(ratings.items())
    ]
    ledger_rows: list[dict[str, Any]] = []
    for match in matches.to_dict("records"):
        home_id, away_id = str(match["home_team_id"]), str(match["away_team_id"])
        if home_id not in ratings or away_id not in ratings:
            raise FallbackError(f"match {match['match_id']} contains an unanchored team")
        pre_home, pre_away = ratings[home_id], ratings[away_id]
        expected = elo_expected(
            pre_home, pre_away, hfa=config.hfa, divisor=config.elo_divisor
        )
        actual = _actual_home(int(match["home_goals"]), int(match["away_goals"]))
        base_delta = config.k * (actual - expected)
        margin_factor = _margin_factor(
            int(match["home_goals"]), int(match["away_goals"]), config
        )
        delta = base_delta * margin_factor
        post_home, post_away = pre_home + delta, pre_away - delta
        if not math.isclose(
            pre_home + pre_away, post_home + post_away, rel_tol=0.0, abs_tol=1e-12
        ):
            raise FallbackError(f"zero-sum invariant failed for match {match['match_id']}")
        ratings[home_id], ratings[away_id] = post_home, post_away
        ledger_rows.append(
            {
                **match,
                "home_elo_pre_match": pre_home,
                "away_elo_pre_match": pre_away,
                "expected_home": expected,
                "actual_home": actual,
                "goal_margin": abs(int(match["home_goals"]) - int(match["away_goals"])),
                "base_delta": base_delta,
                "margin_factor": margin_factor,
                "applied_delta": delta,
                "home_elo_post_match": post_home,
                "away_elo_post_match": post_away,
                "elo_provider": PROVIDER,
                "formula_version": config.formula_version,
            }
        )
        for current_id, opponent_id, value in (
            (home_id, away_id, post_home),
            (away_id, home_id, post_away),
        ):
            history_rows.append(
                {
                    "league": DEFAULT_LEAGUE,
                    "season": DEFAULT_SEASON,
                    "team": team_code[current_id],
                    "team_id": current_id,
                    "effective_at_utc": pd.Timestamp(match["kickoff_utc"]).isoformat(),
                    "event_type": "post_match",
                    "match_id": str(match["match_id"]),
                    "opponent_id": opponent_id,
                    "elo": value,
                    "elo_provider": PROVIDER,
                    "formula_version": config.formula_version,
                }
            )
    ledger = pd.DataFrame(ledger_rows)
    history = pd.DataFrame(history_rows).sort_values(
        ["effective_at_utc", "match_id", "team_id"], kind="stable"
    ).reset_index(drop=True)
    final_total = math.fsum(ratings.values())
    if not math.isclose(initial_total, final_total, rel_tol=0.0, abs_tol=1e-8):
        raise FallbackError("league-wide zero-sum invariant failed")
    return ledger, history, ratings


def build_schedule_output(
    anchor_schedule: pd.DataFrame,
    ledger: pd.DataFrame,
    final_ratings: Mapping[str, float],
) -> pd.DataFrame:
    output = anchor_schedule.copy()
    for column in ("status", "date_played", "result", "venue"):
        if column in output:
            output[column] = output[column].astype("object")
    for column in ("gf", "ga", "xg", "xga", "gw_played"):
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce").astype(float)
    for column in ("match_id", "team_id", "opponent_id"):
        output[column] = _clean_string(output[column])
    completed = ledger.set_index("match_id").to_dict("index") if not ledger.empty else {}
    pre_values: list[float] = []
    opp_pre_values: list[float] = []
    post_values: list[float | None] = []
    for row in output.to_dict("records"):
        match = completed.get(str(row["match_id"]))
        team_id, opponent_id = str(row["team_id"]), str(row["opponent_id"])
        if match is None:
            pre_values.append(float(final_ratings[team_id]))
            opp_pre_values.append(float(final_ratings[opponent_id]))
            post_values.append(None)
            continue
        is_home = team_id == str(match["home_team_id"])
        if not is_home and team_id != str(match["away_team_id"]):
            raise FallbackError(f"schedule identity mismatch for match {row['match_id']}")
        pre_values.append(float(match["home_elo_pre_match"] if is_home else match["away_elo_pre_match"]))
        opp_pre_values.append(float(match["away_elo_pre_match"] if is_home else match["home_elo_pre_match"]))
        post_values.append(float(match["home_elo_post_match"] if is_home else match["away_elo_post_match"]))
    output["elo_pre_match"] = pre_values
    output["opponent_elo_pre_match"] = opp_pre_values
    output["elo_diff_pre_match"] = output["elo_pre_match"] - output["opponent_elo_pre_match"]
    output["elo_post_match"] = post_values
    output["elo_provider"] = PROVIDER

    for index, row in output.iterrows():
        match = completed.get(str(row["match_id"]))
        if match is None:
            continue
        is_home = str(row["team_id"]) == str(match["home_team_id"])
        gf = int(match["home_goals"] if is_home else match["away_goals"])
        ga = int(match["away_goals"] if is_home else match["home_goals"])
        updates: dict[str, Any] = {
            "status": "finished",
            "date_played": match["game_date"],
            "gf": gf,
            "ga": ga,
            "result": "W" if gf > ga else "L" if gf < ga else "D",
        }
        if "gw_played" in output and not pd.isna(match.get("round")):
            updates["gw_played"] = int(match["round"])
        if "venue" in output:
            updates["venue"] = "Home" if is_home else "Away"
        xg = match.get("home_xg" if is_home else "away_xg")
        xga = match.get("away_xg" if is_home else "home_xg")
        if "xg" in output and not pd.isna(xg):
            updates["xg"] = xg
        if "xga" in output and not pd.isna(xga):
            updates["xga"] = xga
        for column, value in updates.items():
            output.at[index, column] = value
    return output


def reference_metrics(schedule: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    _require_columns(reference, {"match_id", "team_id", "elo_pre_match"}, "reference schedule")
    left = schedule[["match_id", "team_id", "elo_pre_match"]].copy()
    right_columns = ["match_id", "team_id", "elo_pre_match"]
    if "gw_orig" in reference:
        right_columns.append("gw_orig")
    right = reference[right_columns].copy()
    for frame in (left, right):
        frame["match_id"] = _clean_string(frame["match_id"])
        frame["team_id"] = _clean_string(frame["team_id"])
    left["generated"] = pd.to_numeric(left.pop("elo_pre_match"), errors="coerce")
    right["reference"] = pd.to_numeric(right.pop("elo_pre_match"), errors="coerce")
    joined = left.merge(right, on=["match_id", "team_id"], how="inner", validate="one_to_one").dropna()
    if joined.empty:
        raise FallbackError("reference schedule has no comparable Elo rows")
    errors = (joined["generated"] - joined["reference"]).abs()
    metrics: dict[str, Any] = {
        "rows_compared": int(len(joined)),
        "mae": float(errors.mean()),
        "median_absolute_error": float(errors.median()),
        "p90_absolute_error": float(errors.quantile(0.90)),
        "p95_absolute_error": float(errors.quantile(0.95)),
        "max_absolute_error": float(errors.max()),
        "rank_correlation": float(joined["generated"].rank().corr(joined["reference"].rank())),
    }
    if "gw_orig" in joined:
        joined["absolute_error"] = errors
        by_week = joined.groupby("gw_orig", dropna=False)["absolute_error"].agg(
            rows="size", mae="mean", max_absolute_error="max"
        )
        metrics["error_by_matchweek"] = {
            str(week): {
                "rows": int(row["rows"]),
                "mae": float(row["mae"]),
                "max_absolute_error": float(row["max_absolute_error"]),
            }
            for week, row in by_week.iterrows()
        }
    return metrics


def _logical_csv_hash(frame: pd.DataFrame) -> str:
    return sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def build_fallback(
    *,
    anchor_schedule_path: Path,
    results_schedule_path: Path,
    as_of: str | pd.Timestamp,
    config: FallbackConfig = FallbackConfig(),
    expected_team_count: int | None = 20,
    expected_fixture_count: int | None = 380,
    expected_anchor_date: str | None = None,
    reject_missing_past_results: bool = True,
    reference_schedule_path: Path | None = None,
    league: str = DEFAULT_LEAGUE,
    season: str = DEFAULT_SEASON,
) -> FallbackResult:
    config.validate()
    anchor_schedule_path = Path(anchor_schedule_path)
    results_schedule_path = Path(results_schedule_path)
    for path in (anchor_schedule_path, results_schedule_path):
        if not path.is_file():
            raise FallbackError(f"input file not found: {path}")
    anchor_schedule = _ensure_match_id(
        pd.read_csv(anchor_schedule_path, low_memory=False), "anchor schedule"
    )
    results_schedule = _ensure_match_id(
        pd.read_csv(results_schedule_path, low_memory=False), "results schedule"
    )
    anchors, fixtures = prepare_anchor_and_fixtures(
        anchor_schedule,
        expected_team_count=expected_team_count,
        expected_fixture_count=expected_fixture_count,
        expected_anchor_date=expected_anchor_date,
    )
    matches, missing_past = prepare_completed_matches(
        results_schedule,
        fixtures,
        anchors,
        as_of=as_of,
        reject_missing_past_results=reject_missing_past_results,
    )
    ledger, history, ratings = replay_matches(anchors, matches, config=config)
    history["league"] = league
    history["season"] = season
    schedule = build_schedule_output(anchor_schedule, ledger, ratings)
    cutoff = _parse_as_of(as_of)
    config_payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    anchor_total = float(math.fsum(anchors["elo_preseason"].astype(float)))
    final_total = float(math.fsum(ratings.values()))
    audit: dict[str, Any] = {
        "status": "validated",
        "provider": PROVIDER,
        "league": league,
        "season": season,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_utc": cutoff.isoformat(),
        "anchor_date": anchors["anchor_date"].iloc[0].date().isoformat(),
        "formula": asdict(config),
        "formula_sha256": sha256(config_payload.encode("utf-8")).hexdigest(),
        "inputs": {
            "anchor_schedule": str(anchor_schedule_path),
            "anchor_schedule_sha256": _file_hash(anchor_schedule_path),
            "results_schedule": str(results_schedule_path),
            "results_schedule_sha256": _file_hash(results_schedule_path),
        },
        "counts": {
            "anchored_teams": int(len(anchors)),
            "fixtures": int(len(fixtures)),
            "completed_matches": int(len(matches)),
            "completed_team_rows": int(len(matches) * 2),
            "missing_past_results": int(len(missing_past)),
            "conflicting_result_rounds": int(matches.get("round_conflict", pd.Series(dtype=bool)).sum()),
        },
        "missing_past_match_ids": missing_past,
        "conflicting_round_match_ids": (
            matches.loc[matches["round_conflict"].eq(True), "match_id"].astype(str).tolist()
            if "round_conflict" in matches
            else []
        ),
        "invariants": {
            "all_results_mirrored": True,
            "all_result_teams_anchored": True,
            "all_results_in_fixture_schedule": True,
            "zero_sum": math.isclose(anchor_total, final_total, rel_tol=0.0, abs_tol=1e-8),
            "anchor_elo_total": anchor_total,
            "final_elo_total": final_total,
            "absolute_total_drift": abs(anchor_total - final_total),
        },
        "outputs": {
            "ledger_rows": int(len(ledger)),
            "team_history_rows": int(len(history)),
            "schedule_rows": int(len(schedule)),
            "ledger_sha256": _logical_csv_hash(ledger),
            "team_history_sha256": _logical_csv_hash(history),
            "schedule_sha256": _logical_csv_hash(schedule),
        },
        "limitations": [
            "Premier League results only; cup and European matches are not applied.",
            "Generated ratings are not authoritative ClubElo values.",
        ],
    }
    if reference_schedule_path is not None:
        reference_path = Path(reference_schedule_path)
        if not reference_path.is_file():
            raise FallbackError(f"reference file not found: {reference_path}")
        reference = _ensure_match_id(
            pd.read_csv(reference_path, low_memory=False), "reference schedule"
        )
        audit["historical_reference"] = {
            "path": str(reference_path),
            "sha256": _file_hash(reference_path),
            **reference_metrics(schedule, reference),
        }
    if not audit["invariants"]["zero_sum"]:
        raise FallbackError("final audit failed the zero-sum invariant")
    return FallbackResult(ledger=ledger, team_history=history, schedule=schedule, audit=audit)


def publish_fallback(result: FallbackResult, destination: Path) -> None:
    destination = Path(destination).resolve()
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir()
    backup: Path | None = None
    try:
        result.ledger.to_csv(stage / "match_ledger.csv", index=False, lineterminator="\n")
        result.team_history.to_csv(stage / "team_history.csv", index=False, lineterminator="\n")
        result.schedule.to_csv(stage / "schedule.csv", index=False, lineterminator="\n")
        (stage / "audit.json").write_text(
            json.dumps(result.audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        required = {"match_ledger.csv", "team_history.csv", "schedule.csv", "audit.json"}
        if {path.name for path in stage.iterdir()} != required:
            raise FallbackError("staged fallback artifact set is incomplete")
        if destination.exists():
            backup = parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
            os.replace(destination, backup)
        os.replace(stage, destination)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if destination.exists() and backup is not None and backup.exists():
            shutil.rmtree(destination)
        if backup is not None and backup.exists():
            os.replace(backup, destination)
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or publish a temporary, isolated ClubElo fallback."
    )
    parser.add_argument("--as-of", required=True, help="Explicit UTC cutoff timestamp/date.")
    parser.add_argument("--league", default=DEFAULT_LEAGUE)
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--anchor-schedule", default=None)
    parser.add_argument("--results-schedule", default=None)
    parser.add_argument("--reference-schedule", default=None)
    parser.add_argument("--anchor-date", default="2026-08-20")
    parser.add_argument("--expected-team-count", type=int, default=20)
    parser.add_argument("--expected-fixture-count", type=int, default=380)
    parser.add_argument("--k", type=float, default=20.0)
    parser.add_argument("--hfa", type=float, default=36.0)
    parser.add_argument("--elo-divisor", type=float, default=400.0)
    parser.add_argument("--margin-mode", choices=["base", "calibrated_margin"], default="base")
    parser.add_argument("--margin-normalizer", type=float, default=None)
    parser.add_argument(
        "--allow-missing-past-results",
        action="store_true",
        help="Unsafe override: audit but do not reject past fixtures without results.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Write the validated artifact set. Without this flag, runs are dry-run only.",
    )
    parser.add_argument("--out-dir", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    base = Path("data/processed")
    anchor = Path(args.anchor_schedule) if args.anchor_schedule else (
        base / "clubelo" / args.league / args.season / "schedule.csv"
    )
    results = Path(args.results_schedule) if args.results_schedule else (
        base / "understat" / args.league / args.season / "schedule.csv"
    )
    destination = Path(args.out_dir) if args.out_dir else (
        base / "clubelo_fallback" / args.league / args.season
    )
    config = FallbackConfig(
        k=args.k,
        hfa=args.hfa,
        elo_divisor=args.elo_divisor,
        margin_mode=args.margin_mode,
        margin_normalizer=args.margin_normalizer,
    )
    try:
        result = build_fallback(
            anchor_schedule_path=anchor,
            results_schedule_path=results,
            as_of=args.as_of,
            config=config,
            expected_team_count=args.expected_team_count,
            expected_fixture_count=args.expected_fixture_count,
            expected_anchor_date=args.anchor_date,
            reject_missing_past_results=not args.allow_missing_past_results,
            reference_schedule_path=(
                Path(args.reference_schedule) if args.reference_schedule else None
            ),
            league=args.league,
            season=args.season,
        )
        if args.publish:
            result.audit["published_to"] = str(destination)
            result.audit["dry_run"] = False
            result.audit["status"] = "published"
            publish_fallback(result, destination)
        else:
            result.audit["dry_run"] = True
        print(json.dumps(result.audit, indent=2, sort_keys=True))
    except FallbackError as exc:
        raise SystemExit(f"ClubElo fallback validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
