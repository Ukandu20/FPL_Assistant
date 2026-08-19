from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Mapping

import pandas as pd


def _safe_snapshot_name(value: str | pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").strftime("%Y-%m-%dT%H-%M-%SZ")


def _stable_json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _stable_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, float) and math.isfinite(value):
        rounded = round(value, 12)
        return 0.0 if rounded == 0 else rounded
    return value


def _stable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove insignificant platform-level float drift before hashing output."""
    stable = frame.copy()
    numeric = stable.select_dtypes(include=["floating"]).columns
    if len(numeric):
        stable[numeric] = stable[numeric].round(12)
        stable[numeric] = stable[numeric].mask(stable[numeric].eq(0), 0.0)
    for column in stable.select_dtypes(include=["object", "string"]).columns:
        def normalize(value: object) -> object:
            if not isinstance(value, str) or not value.startswith(("{", "[")):
                return value
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return value
            return json.dumps(
                _stable_json_value(parsed), sort_keys=True, separators=(",", ":")
            )

        stable[column] = stable[column].map(normalize)
    return stable


def persist_snapshot(
    archetypes: pd.DataFrame,
    team_ratings: pd.DataFrame,
    *,
    output_root: str | Path,
    snapshot_date: str | pd.Timestamp,
    model_version: str,
    evidence_tables: Mapping[str, pd.DataFrame] | None = None,
) -> Path:
    """Write an immutable, version-partitioned snapshot and content manifest."""
    target = (
        Path(output_root)
        / f"model_version={model_version}"
        / f"snapshot={_safe_snapshot_name(snapshot_date)}"
    )
    # Build every representation before touching the immutable snapshot directory.
    # In particular, this prevents a missing Parquet engine or a serialization
    # error from leaving a partially published snapshot behind.
    artifact_frames = {
        "archetypes": archetypes,
        "team_ratings": team_ratings,
    }
    for name, frame in (evidence_tables or {}).items():
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Invalid evidence artifact name: {name!r}")
        artifact_frames[name] = frame

    name_parts = [
        frame[["player_id", "player_name"]]
        for frame in artifact_frames.values()
        if {"player_id", "player_name"}.issubset(frame)
    ]
    if name_parts:
        player_names = (
            pd.concat(name_parts, ignore_index=True)
            .dropna(subset=["player_id", "player_name"])
            .assign(player_id=lambda frame: frame["player_id"].astype("string"))
            .drop_duplicates("player_id", keep="first")
            .set_index("player_id")["player_name"]
        )
    else:
        player_names = pd.Series(dtype="string")

    for name, frame in artifact_frames.items():
        if "player_id" not in frame:
            continue
        enriched = frame.copy()
        mapped_names = enriched["player_id"].astype("string").map(player_names)
        if "player_name" in enriched:
            enriched["player_name"] = enriched["player_name"].fillna(mapped_names)
        else:
            enriched.insert(1, "player_name", mapped_names.astype("string"))
        artifact_frames[name] = enriched

    sort_preferences = (
        "kickoff_utc", "match_id", "team_id", "player_id", "archetype_id",
        "evidence_window",
    )
    normalized_frames: dict[str, pd.DataFrame] = {}
    for name, frame in artifact_frames.items():
        stable = _stable_frame(frame)
        sort_columns = [column for column in sort_preferences if column in stable]
        normalized_frames[name] = (
            stable.sort_values(sort_columns, kind="stable")
            if sort_columns
            else stable
        )

    artifacts: dict[str, bytes] = {}
    artifact_metadata: dict[str, dict[str, object]] = {}
    for logical_name, frame in normalized_frames.items():
        jsonl_name = f"{logical_name}.jsonl"
        csv_name = f"{logical_name}.csv"
        parquet_name = f"{logical_name}.parquet"
        artifacts[jsonl_name] = frame.to_json(
            orient="records", lines=True, date_format="iso", double_precision=12
        ).encode("utf-8")
        artifacts[csv_name] = frame.to_csv(
            index=False,
            lineterminator="\n",
            date_format="%Y-%m-%dT%H:%M:%S.%f%z",
            float_format="%.12g",
        ).encode("utf-8")
        parquet_buffer = BytesIO()
        frame.to_parquet(
            parquet_buffer,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        artifacts[parquet_name] = parquet_buffer.getvalue()

        common_metadata: dict[str, object] = {
            "logical_table": logical_name,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        }
        for artifact_name, artifact_format in (
            (jsonl_name, "jsonl"),
            (csv_name, "csv"),
            (parquet_name, "parquet"),
        ):
            artifact_metadata[artifact_name] = {
                **common_metadata,
                "format": artifact_format,
            }

    hashes = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in artifacts.items()
    }
    manifest = {
        "model_version": model_version,
        "snapshot_date": pd.Timestamp(snapshot_date).isoformat(),
        "artifacts": hashes,
        "artifact_metadata": artifact_metadata,
    }
    manifest_path = target / "manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    # Compare the manifest first so a legacy JSONL-only snapshot cannot be
    # silently augmented, which would violate the snapshot's immutability.
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise FileExistsError(f"Immutable archetype manifest would be overwritten: {manifest_path}")

    for name, content in artifacts.items():
        path = target / name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Immutable archetype snapshot would be overwritten: {path}")

    target.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        (target / name).write_bytes(content)
    manifest_path.write_bytes(manifest_bytes)
    return target


__all__ = ["persist_snapshot"]
