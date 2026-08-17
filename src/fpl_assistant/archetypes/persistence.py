from __future__ import annotations

import hashlib
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
    target.mkdir(parents=True, exist_ok=True)
    artifact_frames = {
        "archetypes.jsonl": archetypes,
        "team_ratings.jsonl": team_ratings,
    }
    for name, frame in (evidence_tables or {}).items():
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Invalid evidence artifact name: {name!r}")
        artifact_frames[f"{name}.jsonl"] = frame

    sort_preferences = (
        "kickoff_utc", "match_id", "team_id", "player_id", "archetype_id",
        "evidence_window",
    )
    normalized_frames = {
        name: _stable_frame(frame) for name, frame in artifact_frames.items()
    }
    artifacts = {
        name: frame.sort_values(
            [column for column in sort_preferences if column in frame],
            kind="stable",
        ).to_json(
            orient="records", lines=True, date_format="iso", double_precision=12
        )
        for name, frame in normalized_frames.items()
    }
    hashes: dict[str, str] = {}
    for name, content in artifacts.items():
        path = target / name
        encoded = content.encode("utf-8")
        if path.exists() and path.read_bytes() != encoded:
            raise FileExistsError(f"Immutable archetype snapshot would be overwritten: {path}")
        path.write_bytes(encoded)
        hashes[name] = hashlib.sha256(encoded).hexdigest()
    manifest = {
        "model_version": model_version,
        "snapshot_date": pd.Timestamp(snapshot_date).isoformat(),
        "artifacts": hashes,
        "artifact_metadata": {
            name: {
                "rows": int(len(frame)),
                "columns": list(frame.columns),
            }
            for name, frame in normalized_frames.items()
        },
    }
    manifest_path = target / "manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise FileExistsError(f"Immutable archetype manifest would be overwritten: {manifest_path}")
    manifest_path.write_bytes(manifest_bytes)
    return target


__all__ = ["persist_snapshot"]
