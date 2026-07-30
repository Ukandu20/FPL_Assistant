from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


PLAYER_FIXTURE_COLUMNS = [
    "season",
    "gameweek",
    "match_id",
    "kickoff_utc",
    "player_id",
    "team_id",
    "opponent_id",
    "venue",
    "started",
    "named_on_bench",
    "minutes",
    "did_not_play",
    "observation_status",
    "availability_status",
    "availability_reason",
    "as_of_timestamp",
    "row_source",
]


def build_complete_player_fixture_panel(
    roster: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    observations: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
    as_of_timestamp: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Create one row for every registered player and club fixture.

    Completed fixtures with no player observation become explicit DNP rows.
    Future fixtures retain unknown minutes rather than being mislabeled as DNP.
    Roster validity dates prevent transferred players from being assigned to
    fixtures outside their registration window.
    """

    roster_required = {"season", "player_id", "team_id"}
    fixture_required = {
        "season",
        "match_id",
        "kickoff_utc",
        "home_team_id",
        "away_team_id",
    }
    missing_roster = roster_required - set(roster.columns)
    missing_fixtures = fixture_required - set(fixtures.columns)
    if missing_roster:
        raise KeyError(f"Roster missing columns: {sorted(missing_roster)}")
    if missing_fixtures:
        raise KeyError(f"Fixtures missing columns: {sorted(missing_fixtures)}")

    as_of = pd.Timestamp(
        as_of_timestamp or datetime.now(timezone.utc)
    )
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")

    roster_work = roster.copy()
    for column in ("valid_from", "valid_to"):
        if column not in roster_work:
            roster_work[column] = pd.NaT
        roster_work[column] = pd.to_datetime(
            roster_work[column], utc=True, errors="coerce"
        )

    fixture_work = fixtures.copy()
    fixture_work["kickoff_utc"] = pd.to_datetime(
        fixture_work["kickoff_utc"], utc=True, errors="coerce"
    )
    for column in ("gameweek", "status"):
        if column not in fixture_work:
            fixture_work[column] = pd.NA

    team_fixtures = pd.concat(
        [
            fixture_work.assign(
                team_id=fixture_work["home_team_id"],
                opponent_id=fixture_work["away_team_id"],
                venue="Home",
            ),
            fixture_work.assign(
                team_id=fixture_work["away_team_id"],
                opponent_id=fixture_work["home_team_id"],
                venue="Away",
            ),
        ],
        ignore_index=True,
    )
    panel = roster_work.merge(
        team_fixtures,
        on=["season", "team_id"],
        how="inner",
        validate="many_to_many",
    )
    active = (
        (panel["valid_from"].isna() | (panel["kickoff_utc"] >= panel["valid_from"]))
        & (panel["valid_to"].isna() | (panel["kickoff_utc"] <= panel["valid_to"]))
    )
    panel = panel.loc[active].copy()

    observation_columns = [
        "match_id",
        "player_id",
        "started",
        "named_on_bench",
        "minutes",
    ]
    if observations is not None and not observations.empty:
        missing = {"match_id", "player_id"} - set(observations.columns)
        if missing:
            raise KeyError(f"Observations missing columns: {sorted(missing)}")
        observed = observations.copy()
        for column in observation_columns:
            if column not in observed:
                observed[column] = pd.NA
        observed = observed[observation_columns].drop_duplicates(
            ["match_id", "player_id"], keep="last"
        )
        panel = panel.merge(
            observed,
            on=["match_id", "player_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        for column in observation_columns[2:]:
            panel[column] = pd.NA

    status = panel["status"].astype(str).str.lower()
    complete_status = status.isin(
        {"complete", "completed", "finished", "ft", "aet", "pen"}
    )
    abandoned_status = status.isin(
        {"postponed", "cancelled", "canceled", "abandoned", "suspended"}
    )
    completed = complete_status | (
        panel["kickoff_utc"].notna()
        & (panel["kickoff_utc"] < as_of)
        & ~abandoned_status
    )
    missing_observation = panel["minutes"].isna()
    panel.loc[completed & missing_observation, "minutes"] = 0
    panel.loc[completed & missing_observation, "started"] = False
    panel.loc[completed & missing_observation, "named_on_bench"] = False

    panel["did_not_play"] = pd.Series(pd.NA, index=panel.index, dtype="boolean")
    panel.loc[completed, "did_not_play"] = (
        pd.to_numeric(panel.loc[completed, "minutes"], errors="coerce")
        .fillna(0)
        .eq(0)
    )
    panel["observation_status"] = np.select(
        [
            ~completed,
            completed & pd.to_numeric(panel["minutes"], errors="coerce").fillna(0).gt(0),
            completed & panel["named_on_bench"].fillna(False).astype(bool),
        ],
        ["fixture_pending", "played", "unused_substitute"],
        default="not_in_matchday_squad",
    )

    panel["availability_status"] = pd.NA
    panel["availability_reason"] = pd.NA
    if availability is not None and not availability.empty:
        availability_required = {"match_id", "player_id", "observed_at"}
        missing = availability_required - set(availability.columns)
        if missing:
            raise KeyError(f"Availability missing columns: {sorted(missing)}")
        available = availability.copy()
        available["observed_at"] = pd.to_datetime(
            available["observed_at"], utc=True, errors="coerce"
        )
        available = available[available["observed_at"] <= as_of]
        available = available.sort_values("observed_at").drop_duplicates(
            ["match_id", "player_id"], keep="last"
        )
        for column in ("availability_status", "availability_reason"):
            source = column.replace("availability_", "")
            if column not in available and source in available:
                available[column] = available[source]
            if column not in available:
                available[column] = pd.NA
        panel = panel.drop(
            columns=["availability_status", "availability_reason"]
        ).merge(
            available[
                [
                    "match_id",
                    "player_id",
                    "availability_status",
                    "availability_reason",
                ]
            ],
            on=["match_id", "player_id"],
            how="left",
            validate="one_to_one",
        )

    panel["as_of_timestamp"] = as_of
    panel["row_source"] = "derived_roster_x_fixture"
    return (
        panel.reindex(columns=PLAYER_FIXTURE_COLUMNS)
        .sort_values(["season", "kickoff_utc", "team_id", "player_id"])
        .reset_index(drop=True)
    )
