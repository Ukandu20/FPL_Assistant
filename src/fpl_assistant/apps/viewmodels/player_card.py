"""Testable presentation helpers for the Streamlit Player Card."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


FORECAST_WINDOW = re.compile(r"^GW(?P<start>\d+)_(?P<end>\d+)\.(?P<ext>parquet|csv)$")
PLAYER_PHOTO_CODE = re.compile(r"^(?P<code>\d+)(?:\.[A-Za-z0-9]+)?$")


def player_photo_urls(
    photo: object, *, asset_version: str = "25"
) -> dict[str, str]:
    """Return current and legacy official portrait URLs for valid FPL metadata."""
    match = PLAYER_PHOTO_CODE.fullmatch(str(photo).strip())
    version = str(asset_version).strip()
    if match is None or not version.isdigit():
        return {}
    code = match.group("code")
    root = "https://resources.premierleague.com"
    return {
        "current": (
            f"{root}/premierleague{version}/photos/players/110x140/{code}.png"
        ),
        "legacy": (
            f"{root}/premierleague/photos/players/110x140/p{code}.png"
        ),
    }


def player_placeholder_url(*, asset_version: str = "25") -> str | None:
    """Return the official placeholder portrait for a valid asset collection."""
    version = str(asset_version).strip()
    if not version.isdigit():
        return None
    return (
        "https://resources.premierleague.com/"
        f"premierleague{version}/photos/players/110x140/placeholder.png"
    )


def team_badge_url(code: object, *, asset_version: str = "25") -> str | None:
    """Return the official season-versioned badge URL for a numeric team code."""
    numeric_code = pd.to_numeric(code, errors="coerce")
    version = str(asset_version).strip()
    if pd.isna(numeric_code) or float(numeric_code) % 1 or not version.isdigit():
        return None
    return (
        "https://resources.premierleague.com/"
        f"premierleague{version}/badges-alt/{int(numeric_code)}.svg"
    )


def latest_forecast_path(root: Path, season: str) -> Path | None:
    """Return the newest GW-windowed forecast for exactly ``season``.

    Parquet is preferred when both formats exist. No cross-season fallback is
    permitted because stale projections are worse than an explicit empty state.
    """
    season_root = root / season
    if not season_root.is_dir():
        return None

    candidates: list[tuple[int, int, int, Path]] = []
    for path in season_root.iterdir():
        match = FORECAST_WINDOW.match(path.name)
        if not match:
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        format_preference = 1 if match.group("ext") == "parquet" else 0
        candidates.append((end, start, format_preference, path))
    return max(candidates, default=(0, 0, 0, None))[-1]


def prepare_player_forecast(
    forecasts: pd.DataFrame,
    player_id: str,
    *,
    season: str,
) -> pd.DataFrame:
    """Return one deterministic upcoming-fixture row per player/game/fixture."""
    if forecasts.empty or "player_id" not in forecasts:
        return pd.DataFrame()
    data = forecasts.copy()
    if "season" in data:
        data = data.loc[data["season"].astype("string").eq(str(season))]
    data["player_id"] = data["player_id"].astype("string")
    data = data.loc[data["player_id"].eq(str(player_id))].copy()
    if data.empty:
        return data

    for column in [
        "gw_orig",
        "pred_minutes",
        "p_goal",
        "p_assist",
        "xg_mean",
        "xa_mean",
        "fdr",
        "xPts",
    ]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    identity = [column for column in ["player_id", "gw_orig", "game_id"] if column in data]
    if not identity:
        identity = ["player_id"]
    sort_columns = [column for column in ["gw_orig", "date_sched", "game_id"] if column in data]
    if sort_columns:
        data = data.sort_values(sort_columns, kind="mergesort")
    return data.drop_duplicates(identity, keep="last").reset_index(drop=True)


def profile_dimensions(profile: pd.Series) -> list[dict[str, float | str]]:
    """Return the three display dimensions for an outfielder or goalkeeper."""
    position = str(profile.get("fpl_pos", "")).upper()
    if position in {"GK", "GKP"}:
        dimensions = [
            ("Shot stopping", "shot_stopping_percentile"),
            ("Sweeping", "sweeping_percentile"),
            ("Distribution", "distribution_percentile"),
        ]
    else:
        dimensions = [
            ("Goal threat", "goal_threat_percentile"),
            ("Creativity", "creativity_percentile"),
            ("Defensive activity", "defensive_threat_percentile"),
        ]
    output = []
    for label, column in dimensions:
        value = pd.to_numeric(profile.get(column), errors="coerce")
        if pd.notna(value):
            output.append({"Dimension": label, "Percentile": float(value)})
    return output


def forecast_summary(forecast: pd.DataFrame) -> dict[str, float | int | str]:
    """Aggregate the selected upcoming window for Overview cards."""
    if forecast.empty:
        return {}
    xp = pd.to_numeric(
        forecast.get("xPts", pd.Series(index=forecast.index, dtype="float64")),
        errors="coerce",
    )
    minutes = pd.to_numeric(
        forecast.get(
            "pred_minutes", pd.Series(index=forecast.index, dtype="float64")
        ),
        errors="coerce",
    )
    first = forecast.iloc[0]
    opponent = str(first.get("opponent", "—"))
    home_value = first.get("is_home", False)
    is_home = str(home_value).strip().lower() in {"1", "1.0", "true", "yes"}
    venue = "H" if is_home else "A"
    return {
        "fixtures": int(len(forecast)),
        "expected_points": float(xp.sum(min_count=1)),
        "predicted_minutes": float(minutes.sum(min_count=1)),
        "next_fixture": f"{opponent} ({venue})",
    }


def prepare_player_fixtures(
    schedule: pd.DataFrame,
    team_numeric_id: object,
    *,
    forecast: pd.DataFrame | None = None,
    limit: int = 5,
) -> pd.DataFrame:
    """Return a team's next fixtures, enriched by—but independent of—forecasts."""
    if schedule.empty or "team" not in schedule:
        return pd.DataFrame()
    team_id = pd.to_numeric(team_numeric_id, errors="coerce")
    if pd.isna(team_id):
        return pd.DataFrame()

    fixtures = schedule.copy()
    fixtures["team"] = pd.to_numeric(fixtures["team"], errors="coerce")
    fixtures = fixtures.loc[fixtures["team"].eq(float(team_id))].copy()
    if fixtures.empty:
        return fixtures

    for column in ["gw_orig", "fdr"]:
        if column in fixtures:
            fixtures[column] = pd.to_numeric(fixtures[column], errors="coerce")
    sort_columns = [
        column for column in ["gw_orig", "kickoff_time", "date_sched", "fpl_id"]
        if column in fixtures
    ]
    if sort_columns:
        fixtures = fixtures.sort_values(sort_columns, kind="mergesort")
    fixtures = fixtures.drop_duplicates(
        [column for column in ["fpl_id", "team"] if column in fixtures],
        keep="last",
    ).head(limit)

    if forecast is not None and not forecast.empty:
        projection_columns = [
            column
            for column in [
                "gw_orig",
                "opponent",
                "is_home",
                "pred_minutes",
                "xPts",
                "p_goal",
                "p_assist",
            ]
            if column in forecast
        ]
        if {"gw_orig", "opponent", "is_home"}.issubset(projection_columns):
            projections = forecast[projection_columns].copy()
            projections["gw_orig"] = pd.to_numeric(
                projections["gw_orig"], errors="coerce"
            )
            for column in ["opponent", "is_home"]:
                projections[column] = projections[column].astype("string")
                fixtures[column] = fixtures[column].astype("string")
            projections = projections.drop_duplicates(
                ["gw_orig", "opponent", "is_home"], keep="last"
            )
            fixtures = fixtures.merge(
                projections,
                on=["gw_orig", "opponent", "is_home"],
                how="left",
                validate="one_to_one",
            )
    return fixtures.reset_index(drop=True)


def select_profile_snapshot(
    current_profiles: pd.DataFrame,
    previous_profiles: pd.DataFrame,
    player_id: str,
    *,
    current_season: str,
    previous_season: str | None,
) -> tuple[pd.Series | None, str | None, bool]:
    """Select current evidence or a clearly labelled prior-season baseline."""
    accepted = {"Established", "Provisional"}

    def find(frame: pd.DataFrame) -> pd.Series | None:
        if frame.empty or "player_id" not in frame:
            return None
        matches = frame.loc[frame["player_id"].astype("string").eq(str(player_id))]
        if matches.empty:
            return None
        return matches.iloc[0]

    current = find(current_profiles)
    if current is not None and str(current.get("profile_status")) in accepted:
        return current, current_season, False

    previous = find(previous_profiles)
    if previous is not None and str(previous.get("profile_status")) in accepted:
        return previous, previous_season, True
    return current, current_season if current is not None else None, False


def profile_trend(
    current_profiles: pd.DataFrame,
    previous_profiles: pd.DataFrame,
    player_id: str,
) -> dict[str, float | str]:
    """Compare compatible profile dimensions across adjacent seasons."""
    accepted = {"Established", "Provisional"}

    def find(frame: pd.DataFrame) -> pd.Series | None:
        if frame.empty or "player_id" not in frame:
            return None
        matches = frame.loc[frame["player_id"].astype("string").eq(str(player_id))]
        return None if matches.empty else matches.iloc[0]

    current = find(current_profiles)
    previous = find(previous_profiles)
    if current is None or previous is None:
        return {}
    if str(current.get("profile_status")) not in accepted:
        return {}
    if str(previous.get("profile_status")) not in accepted:
        return {}
    if str(current.get("fpl_pos", "")).upper() != str(
        previous.get("fpl_pos", "")
    ).upper():
        return {}

    current_dimensions = {
        str(item["Dimension"]): float(item["Percentile"])
        for item in profile_dimensions(current)
    }
    previous_dimensions = {
        str(item["Dimension"]): float(item["Percentile"])
        for item in profile_dimensions(previous)
    }
    shared = sorted(current_dimensions.keys() & previous_dimensions.keys())
    if not shared:
        return {}
    delta = sum(
        current_dimensions[label] - previous_dimensions[label] for label in shared
    ) / len(shared)
    label = "Rising" if delta >= 7 else "Declining" if delta <= -7 else "Stable"
    return {"label": label, "delta": round(delta, 1)}


def recent_form_summary(
    gameweek_history: pd.DataFrame, *, window: int = 5
) -> dict[str, float | int]:
    """Summarise the latest played gameweeks without fabricating preseason form."""
    if gameweek_history.empty:
        return {}
    recent = gameweek_history.tail(window)
    points = pd.to_numeric(recent.get("Total FPL points"), errors="coerce")
    minutes = pd.to_numeric(recent.get("Minutes"), errors="coerce")
    starts = (
        int(pd.to_numeric(recent["Starts"], errors="coerce").fillna(0).sum())
        if "Starts" in recent
        else int(minutes.ge(60).sum())
    )
    result: dict[str, float | int] = {
        "gameweeks": int(len(recent)),
        "points": float(points.sum(min_count=1)),
        "minutes": float(minutes.sum(min_count=1)),
        "appearances": int(minutes.gt(0).sum()),
        "starts": starts,
    }
    if "Returns" in recent:
        result["returns"] = int(
            pd.to_numeric(recent["Returns"], errors="coerce").fillna(0).sum()
        )
    return result


def comparable_players(
    season_players: pd.DataFrame,
    player_id: str,
    *,
    price_tolerance: int = 5,
    limit: int = 3,
) -> pd.DataFrame:
    """Return same-position alternatives within a raw-price tolerance."""
    required = {"player_id", "name", "fpl_pos", "now_cost"}
    if season_players.empty or not required.issubset(season_players.columns):
        return pd.DataFrame()
    ids = season_players["player_id"].astype("string")
    selected = season_players.loc[ids.eq(str(player_id))]
    if selected.empty:
        return pd.DataFrame()
    player = selected.iloc[0]
    price = pd.to_numeric(player.get("now_cost"), errors="coerce")
    if pd.isna(price):
        return pd.DataFrame()

    candidates = season_players.loc[
        season_players["fpl_pos"].astype("string").eq(str(player.get("fpl_pos")))
        & ~ids.eq(str(player_id))
    ].copy()
    candidates["_price"] = pd.to_numeric(candidates["now_cost"], errors="coerce")
    candidates = candidates.loc[candidates["_price"].sub(price).abs().le(price_tolerance)]
    candidates["_points"] = pd.to_numeric(
        candidates.get(
            "total_points", pd.Series(0, index=candidates.index)
        ), errors="coerce"
    ).fillna(0)
    candidates["_ownership"] = pd.to_numeric(
        candidates.get(
            "selected_by_percent", pd.Series(0, index=candidates.index)
        ), errors="coerce"
    ).fillna(0)
    candidates["_price_distance"] = candidates["_price"].sub(price).abs()
    candidates = candidates.sort_values(
        ["_points", "_ownership", "_price_distance", "name"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).head(limit)
    output = candidates[["player_id", "name", "team", "now_cost", "total_points", "selected_by_percent"]].copy()
    output["now_cost"] = output["now_cost"].map(
        lambda value: f"£{float(value) / 10:.1f}m" if pd.notna(value) else "—"
    )
    return output.rename(
        columns={
            "name": "Player",
            "team": "Team",
            "now_cost": "Price",
            "total_points": "Points",
            "selected_by_percent": "Ownership %",
        }
    ).reset_index(drop=True)


def decision_factors(
    fixtures: pd.DataFrame,
    recent: dict[str, float | int],
    profile: pd.Series | None,
    details: pd.Series | None,
    *,
    profile_is_carryover: bool,
) -> tuple[list[str], list[str]]:
    """Build concise, auditable reasons for and risks against selection."""
    positives: list[str] = []
    risks: list[str] = []

    if not fixtures.empty:
        next_fixture = fixtures.iloc[0]
        opponent = str(next_fixture.get("opponent", "opponent"))
        is_home = str(next_fixture.get("is_home", "")).lower() in {
            "1", "1.0", "true"
        }
        fdr = pd.to_numeric(next_fixture.get("fdr"), errors="coerce")
        if pd.notna(fdr) and fdr <= 2:
            positives.append(
                f"Favourable next fixture: {opponent} ({'H' if is_home else 'A'}), FDR {fdr:.0f}."
            )
        elif pd.notna(fdr) and fdr >= 4:
            risks.append(
                f"Difficult next fixture: {opponent} ({'H' if is_home else 'A'}), FDR {fdr:.0f}."
            )
        predicted = pd.to_numeric(next_fixture.get("pred_minutes"), errors="coerce")
        if pd.notna(predicted) and predicted >= 75:
            positives.append(f"Projected for {predicted:.0f} minutes in the next fixture.")
        elif pd.notna(predicted) and predicted < 60:
            risks.append(f"Only {predicted:.0f} projected minutes in the next fixture.")
        elif pd.isna(predicted):
            risks.append("The next-fixture minutes forecast has not been published.")
    else:
        risks.append("No upcoming fixture is currently published.")

    if profile is not None and str(profile.get("profile_status")) in {
        "Established", "Provisional"
    }:
        dimensions = profile_dimensions(profile)
        if dimensions:
            best = max(dimensions, key=lambda item: float(item["Percentile"]))
            if float(best["Percentile"]) >= 75:
                positives.append(
                    f"{best['Dimension']} ranks at P{float(best['Percentile']):.0f} among positional peers."
                )
        if profile_is_carryover:
            risks.append("The production profile is a previous-season baseline, not current-season evidence.")

    if recent:
        gameweeks = int(recent.get("gameweeks", 0))
        points = float(recent.get("points", 0))
        starts = int(recent.get("starts", 0))
        if gameweeks and points / gameweeks >= 5:
            positives.append(f"Averaging {points / gameweeks:.1f} points across the last {gameweeks} GWs.")
        if gameweeks >= 3 and starts <= gameweeks / 2:
            risks.append(f"Played 60+ minutes in only {starts} of the last {gameweeks} GWs.")

    if details is not None:
        chance = pd.to_numeric(
            details.get("chance_of_playing_next_round"), errors="coerce"
        )
        if pd.notna(chance) and chance < 75:
            risks.append(f"FPL lists only a {chance:.0f}% chance of playing next round.")
        transfers_in = pd.to_numeric(details.get("transfers_in_event"), errors="coerce")
        transfers_out = pd.to_numeric(details.get("transfers_out_event"), errors="coerce")
        if pd.notna(transfers_in) and pd.notna(transfers_out):
            net = float(transfers_in - transfers_out)
            if net >= 25000:
                positives.append(f"Net {net:,.0f} transfers in this gameweek.")
            elif net <= -25000:
                risks.append(f"Net {abs(net):,.0f} transfers out this gameweek.")

    return positives[:3], risks[:3]


__all__ = [
    "comparable_players",
    "decision_factors",
    "forecast_summary",
    "latest_forecast_path",
    "prepare_player_forecast",
    "prepare_player_fixtures",
    "player_photo_urls",
    "player_placeholder_url",
    "profile_trend",
    "profile_dimensions",
    "recent_form_summary",
    "select_profile_snapshot",
    "team_badge_url",
]
