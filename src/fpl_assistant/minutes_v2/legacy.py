from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .features import normalize_position


def prepare_v1_history_root(
    registry_root: Path, output_root: Path, seasons: list[str], team_map: dict[str, int]
) -> None:
    """Expose registry calendars under V1's legacy filename without editing V1."""
    for season in seasons:
        source = registry_root / season / "player_fixture_calendar.csv"
        if not source.exists():
            raise FileNotFoundError(source)
        destination = output_root / season / "player_minutes_calendar.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        history = pd.read_csv(source, low_memory=False)
        history["team_id"] = history["team_id"].astype(str).map(team_map)
        if destination.exists():
            destination.unlink()
        history.to_csv(destination, index=False)


def prepare_v1_inputs(
    player_calendar: Path,
    fixture_calendar: Path,
    prediction_cutoff: str | pd.Timestamp,
    gws: set[int],
    fixtures_output: Path,
    squads_output: Path,
    registry_root: Path | None = None,
    history_root: Path | None = None,
    history_seasons: list[str] | None = None,
) -> None:
    """Explicitly adapt timestamp-safe V2 registry inputs to V1's old schema."""
    cutoff = pd.to_datetime(prediction_cutoff, utc=True)
    players = pd.read_csv(player_calendar, low_memory=False)
    players["information_timestamp"] = pd.to_datetime(
        players["information_timestamp"], format="mixed", utc=True, errors="coerce"
    )
    players["date_sched"] = pd.to_datetime(
        players["date_sched"], format="mixed", utc=True, errors="coerce"
    )
    eligible = players["eligible_for_fixture"].astype("string").str.lower().isin({"1", "true", "yes"})
    safe = players["eligibility_timestamp_safe"].astype("string").str.lower().isin({"1", "true", "yes"})
    target = players[
        pd.to_numeric(players["gw_orig"], errors="coerce").isin(gws)
        & players["date_sched"].ge(cutoff)
        & eligible & safe & players["information_timestamp"].le(cutoff)
    ].copy()
    if target.empty:
        raise ValueError("No timestamp-safe V1 shadow roster rows at cutoff")
    fixtures = pd.read_csv(fixture_calendar, low_memory=False)
    fixtures = fixtures[pd.to_numeric(fixtures["gw_orig"], errors="coerce").isin(gws)].copy()
    fixture_dates = pd.to_datetime(
        fixtures["date_sched"], format="mixed", utc=True, errors="coerce"
    )
    fixtures = fixtures[fixture_dates.ge(cutoff)].copy()
    # V1 operates on timezone-naive datetimes. The cutoff comparison above is
    # timezone-aware; only the explicit legacy adapter removes timezone info.
    fixtures["date_played"] = pd.to_datetime(
        fixtures["date_sched"], format="mixed", utc=True, errors="raise"
    ).dt.tz_localize(None)
    team_ids = sorted(fixtures["team_id"].dropna().astype(str).unique())
    team_map = {team_id: index + 1 for index, team_id in enumerate(team_ids)}
    target["pos"] = normalize_position(target["pos"]).astype("string")
    target["team_id"] = target["team_id"].astype(str).map(team_map)
    squads = target[["player_id", "player", "pos", "team_id"]].drop_duplicates()
    if squads["team_id"].isna().any() or squads.duplicated(["player_id", "team_id"]).any():
        raise ValueError("Ambiguous timestamp-safe V1 shadow squad")
    for column in ("team_id", "opponent_id", "home_id", "away_id", "home_team_id", "away_team_id"):
        if column in fixtures:
            fixtures[column] = fixtures[column].astype(str).map(team_map)
    fixtures_output.parent.mkdir(parents=True, exist_ok=True)
    squads_output.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(fixtures_output, index=False)
    squads.to_csv(squads_output, index=False)
    if registry_root is not None and history_root is not None:
        prepare_v1_history_root(registry_root, history_root, history_seasons or [], team_map)


def prepare_v1_shadow_artifact(
    raw_v1: Path,
    v2_artifact: Path,
    output: Path,
    prediction_cutoff: str | pd.Timestamp,
) -> None:
    """Enrich unchanged V1 predictions with the V2 roster/audit contract."""
    cutoff = pd.to_datetime(prediction_cutoff, utc=True)
    v1 = pd.read_csv(raw_v1, low_memory=False)
    v2 = pd.read_csv(v2_artifact, low_memory=False)
    keys = ["season", "gw_orig", "player_id"]
    if any(key not in v1 or key not in v2 for key in keys):
        raise ValueError("V1 shadow enrichment requires season/GW/player keys")
    if v1.duplicated(keys).any() or v2.duplicated(keys).any():
        raise ValueError("V1 shadow enrichment requires unique player/GW rows")
    identity = [column for column in ("match_id", "team_id", "opponent_id", "player", "pos") if column in v2]
    value_columns = [column for column in v1.columns if column not in keys and column not in identity]
    enriched = v2[keys + identity].merge(v1[keys + value_columns], on=keys, how="left", validate="one_to_one")
    if enriched["pred_minutes"].isna().any():
        missing = enriched.loc[enriched["pred_minutes"].isna(), keys].head().to_dict(orient="records")
        raise ValueError(f"V1 did not predict the shared timestamp-safe roster: {missing}")
    enriched["prediction_cutoff"] = cutoff.isoformat()
    enriched["model_version"] = "minutes/v1"
    operational_valid = not (
        enriched["pred_minutes"].isna().any()
        or enriched["pred_minutes"].abs().max() == 0
    )
    enriched["v1_operational_valid"] = operational_valid
    enriched["v1_operational_warning"] = (
        "" if operational_valid else "legacy artifact has all-zero duration predictions; regression heads are absent"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)
    v2_meta_path = v2_artifact.with_suffix(v2_artifact.suffix + ".meta.json")
    v2_meta = json.loads(v2_meta_path.read_text(encoding="utf-8"))
    raw_hash = hashlib.sha256(raw_v1.read_bytes()).hexdigest()
    metadata = {
        "model_version": "minutes/v1",
        "prediction_cutoff": cutoff.isoformat(),
        "input_data_identifiers": v2_meta["input_data_identifiers"],
        "v1_raw_artifact": str(raw_v1.resolve()),
        "v1_raw_sha256": raw_hash,
        "compatibility_adapter": "minutes_v2.prepare_v1_shadow_artifact",
        "operational_valid": operational_valid,
        "operational_warning": "" if operational_valid else "all-zero duration predictions; legacy regression heads are absent",
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
