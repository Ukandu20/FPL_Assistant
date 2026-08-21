from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MinutesV2Config:
    architecture_version: str
    feature_version: str
    registry_root: Path
    artifact_root: Path
    shadow_root: Path
    canonical_label_seasons: tuple[str, ...]
    diagnostic_only_seasons: tuple[str, ...]
    forward_season: str
    trusted_starter_sources: tuple[str, ...]
    untrusted_starter_sources: tuple[str, ...]
    calibration_gws: int = 6
    evaluation_gws: int = 6
    step_gws: int = 6
    final_holdout_gws: int = 6
    calibration_min_rows: int = 200
    calibration_min_positive: int = 30
    calibration_min_negative: int = 30
    calibration_max_fold_brier_deterioration: float = 0.002
    random_seed: int = 20260821
    bootstrap_iterations: int = 2000
    bootstrap_confidence: float = 0.95
    acceptance_margins: dict[str, float] = field(default_factory=dict)
    extensions: dict[str, bool] = field(default_factory=dict)
    source_path: Path | None = None

    def validate(self) -> None:
        if self.architecture_version != "minutes/v2.0":
            raise ValueError("Canonical implementation requires architecture_version=minutes/v2.0")
        if set(self.canonical_label_seasons) & set(self.diagnostic_only_seasons):
            raise ValueError("Canonical and diagnostic-only seasons must be disjoint")
        if any(self.extensions.values()):
            raise ValueError("V2.1+ extensions must remain disabled in canonical V2.0")
        for key in ("calibration_gws", "evaluation_gws", "step_gws", "final_holdout_gws"):
            if getattr(self, key) != 6:
                raise ValueError(f"Hardened V2.0 requires {key}=6")


def load_config(path: str | Path = "config/minutes_v2.json") -> MinutesV2Config:
    source = Path(path)
    raw: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    for key in (
        "registry_root", "artifact_root", "shadow_root",
    ):
        raw[key] = Path(raw[key])
    for key in (
        "canonical_label_seasons", "diagnostic_only_seasons",
        "trusted_starter_sources", "untrusted_starter_sources",
    ):
        raw[key] = tuple(raw[key])
    cfg = MinutesV2Config(**raw, source_path=source.resolve())
    cfg.validate()
    return cfg
