from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


LEVEL_TEAM_SEASON = "team_season"
LEVEL_TEAM_MATCH = "team_match"
LEVEL_PLAYER_SEASON = "player_season"
LEVEL_PLAYER_MATCH = "player_match"
LEVEL_SUPPLEMENTARY = "supplementary"


# This is the reduced public surface supported by the current FBref reader.
# Downstream code must query this contract instead of assuming the historical
# advanced-table surface is present.
FBREF_CAPABILITIES: dict[str, tuple[str, ...]] = {
    LEVEL_TEAM_SEASON: (
        "standard",
        "keeper",
        "shooting",
        "playing_time",
        "misc",
    ),
    LEVEL_TEAM_MATCH: (
        "schedule",
        "keeper",
        "shooting",
        "misc",
    ),
    LEVEL_PLAYER_SEASON: (
        "standard",
        "keeper",
        "shooting",
        "playing_time",
        "misc",
    ),
    LEVEL_PLAYER_MATCH: (
        "summary",
        "keepers",
    ),
    LEVEL_SUPPLEMENTARY: (
        "lineups",
        "events",
    ),
}


# Aliases are accepted only at the provider boundary. Persisted raw/staged
# names always use the canonical value on the right.
FBREF_STAT_ALIASES: dict[str, dict[str, str]] = {
    LEVEL_TEAM_SEASON: {
        "keepers": "keeper",
        "goalkeeping": "keeper",
        "playingtime": "playing_time",
    },
    LEVEL_TEAM_MATCH: {
        "keepers": "keeper",
        "goalkeeping": "keeper",
    },
    LEVEL_PLAYER_SEASON: {
        "keepers": "keeper",
        "goalkeeping": "keeper",
        "playingtime": "playing_time",
    },
    LEVEL_PLAYER_MATCH: {
        "keeper": "keepers",
        "goalkeeping": "keepers",
    },
    LEVEL_SUPPLEMENTARY: {
        "lineup": "lineups",
    },
}


HISTORICAL_UNAVAILABLE_STATS: tuple[str, ...] = (
    "defense",
    "goal_shot_creation",
    "gca",
    "keeper_adv",
    "keepersadv",
    "passing",
    "passing_types",
    "possession",
    "xg",
)


def supported_stats(level: str) -> tuple[str, ...]:
    try:
        return FBREF_CAPABILITIES[level]
    except KeyError as exc:
        raise ValueError(
            f"Unknown FBref capability level {level!r}; "
            f"expected one of {sorted(FBREF_CAPABILITIES)}"
        ) from exc


def normalize_stat_type(level: str, stat_type: str) -> str:
    value = str(stat_type).strip().lower().replace("-", "_")
    return FBREF_STAT_ALIASES.get(level, {}).get(value, value)


def validate_requested_stats(
    level: str,
    requested: Sequence[str] | None,
) -> list[str]:
    allowed = supported_stats(level)
    if requested is None:
        return list(allowed)
    normalized = list(
        dict.fromkeys(normalize_stat_type(level, value) for value in requested)
    )
    unsupported = [value for value in normalized if value not in allowed]
    if unsupported:
        raise ValueError(
            f"Unsupported FBref {level} statistic(s): {unsupported}. "
            f"Supported values: {list(allowed)}"
        )
    return [value for value in allowed if value in normalized]


def capability_document() -> dict[str, Any]:
    return {
        "provider": "fbref",
        "contract_version": 1,
        "capabilities": {
            level: list(values) for level, values in FBREF_CAPABILITIES.items()
        },
        "aliases": FBREF_STAT_ALIASES,
        "historical_unavailable": list(HISTORICAL_UNAVAILABLE_STATS),
    }


def write_capability_document(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(capability_document(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True)
class CoverageRecord:
    level: str
    stat_type: str
    status: str
    rows: int
    columns: tuple[str, ...]
    non_null_fraction: float | None
    schema_hash: str | None
    output_path: str | None


def _flatten_column(column: Any) -> str:
    if isinstance(column, tuple):
        return ".".join(str(part) for part in column if str(part))
    return str(column)


def schema_hash(columns: Iterable[Any]) -> str:
    payload = "\n".join(_flatten_column(column) for column in columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def coverage_record(
    *,
    level: str,
    stat_type: str,
    status: str,
    frame: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> CoverageRecord:
    if frame is None:
        return CoverageRecord(
            level=level,
            stat_type=stat_type,
            status=status,
            rows=0,
            columns=(),
            non_null_fraction=None,
            schema_hash=None,
            output_path=str(output_path) if output_path is not None else None,
        )
    columns = tuple(_flatten_column(column) for column in frame.columns)
    cells = int(frame.shape[0] * frame.shape[1])
    non_null = int(frame.notna().sum().sum()) if cells else 0
    return CoverageRecord(
        level=level,
        stat_type=stat_type,
        status=status,
        rows=int(len(frame)),
        columns=columns,
        non_null_fraction=(non_null / cells) if cells else None,
        schema_hash=schema_hash(columns),
        output_path=str(output_path) if output_path is not None else None,
    )


def coverage_from_outputs(
    *,
    output_dir: Path,
    statuses: Mapping[str, Mapping[str, Sequence[str]]],
    layout: str = "folders",
) -> list[CoverageRecord]:
    records: list[CoverageRecord] = []
    for level, by_status in statuses.items():
        for status, stat_types in by_status.items():
            for stat_type in stat_types:
                candidates = (
                    [
                        output_dir / level / f"{stat_type}.csv",
                        output_dir / f"{level}_{stat_type}.csv",
                    ]
                    if layout == "folders"
                    else [
                        output_dir / f"{level}_{stat_type}.csv",
                        output_dir / level / f"{stat_type}.csv",
                    ]
                )
                output_path = next((path for path in candidates if path.is_file()), None)
                frame: pd.DataFrame | None = None
                if output_path is not None:
                    try:
                        frame = pd.read_csv(output_path)
                    except Exception:
                        frame = None
                records.append(
                    coverage_record(
                        level=level,
                        stat_type=stat_type,
                        status=status,
                        frame=frame,
                        output_path=output_path,
                    )
                )
    return records


def write_coverage_manifest(
    path: Path,
    *,
    league: str,
    season: str,
    records: Sequence[CoverageRecord],
    extras: Mapping[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": "fbref",
        "contract_version": 1,
        "league": league,
        "season": season,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capabilities": capability_document()["capabilities"],
        "records": [asdict(record) for record in records],
        "extras": dict(extras or {}),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
