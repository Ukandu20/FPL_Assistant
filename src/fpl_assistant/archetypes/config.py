from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "fpl_archetypes_v1.json"


@dataclass(frozen=True)
class ArchetypeConfig:
    values: Mapping[str, Any]
    source_path: Path

    @property
    def model_version(self) -> str:
        return str(self.values["model_version"])

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, Mapping):
            raise KeyError(f"Missing configuration section: {name}")
        return value


def _validate_probability(value: object, field: str) -> None:
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise ValueError(f"{field} must be between 0 and 1")


def validate_config(values: Mapping[str, Any]) -> None:
    required = {
        "model_version",
        "temporal_weights",
        "confidence",
        "usage",
        "team_ratings",
        "components",
        "provider_fields",
        "deferred_components",
        "validation",
    }
    missing = required - set(values)
    if missing:
        raise ValueError(f"Archetype configuration missing: {sorted(missing)}")

    temporal = values["temporal_weights"]
    if set(temporal) != {"previous_season", "earlier_current", "recent"}:
        raise ValueError("temporal_weights must define the three catalogue windows")
    if abs(sum(float(value) for value in temporal.values()) - 1.0) > 1e-9:
        raise ValueError("temporal_weights must sum to 1")

    confidence = values["confidence"]
    confidence_total = sum(
        float(confidence[name])
        for name in (
            "minutes_weight",
            "appearances_weight",
            "context_weight",
            "precision_weight",
            "data_quality_weight",
        )
    )
    if abs(confidence_total - 1.0) > 1e-9:
        raise ValueError("confidence weights must sum to 1")

    for component_id, component in values["components"].items():
        weights_by_position = component.get("weights_by_position")
        collections = weights_by_position.values() if weights_by_position else [component["weights"]]
        for weights in collections:
            if abs(sum(float(weight) for weight in weights.values()) - 1.0) > 1e-9:
                raise ValueError(f"{component_id} component weights must sum to 1")

    _validate_probability(values["validation"]["minimum_fold_win_rate"], "minimum_fold_win_rate")
    deferred = {str(item).upper() for item in values["deferred_components"]}
    if not {"SWEEPER", "DISTRIBUTOR"}.issubset(deferred):
        raise ValueError("V1 must keep Sweeper and Distributor deferred")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ArchetypeConfig:
    source = Path(path)
    values = json.loads(source.read_text(encoding="utf-8"))
    validate_config(values)
    return ArchetypeConfig(values=values, source_path=source)


__all__ = ["ArchetypeConfig", "DEFAULT_CONFIG_PATH", "load_config", "validate_config"]
