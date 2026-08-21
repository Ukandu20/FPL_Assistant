from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import MinutesV2Config


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def code_version() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip())
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def deterministic_run_id(config: MinutesV2Config, prediction_cutoff: str, input_hashes: dict[str, object]) -> str:
    payload = json.dumps({
        "architecture": config.architecture_version,
        "feature_version": config.feature_version,
        "prediction_cutoff": prediction_cutoff,
        "inputs": input_hashes,
        "seed": config.random_seed,
    }, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def save_training_artifacts(
    artifact_dir: Path,
    models: Any,
    calibrators: Any,
    model_card: dict[str, object],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "models.pkl").write_bytes(pickle.dumps(models, protocol=pickle.HIGHEST_PROTOCOL))
    (artifact_dir / "calibrators.pkl").write_bytes(pickle.dumps(calibrators, protocol=pickle.HIGHEST_PROTOCOL))
    card = dict(model_card)
    card.setdefault("creation_timestamp", datetime.now(timezone.utc).isoformat())
    card.setdefault("code_version", code_version())
    (artifact_dir / "model_card.json").write_text(
        json.dumps(card, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8"
    )


def load_training_artifacts(artifact_dir: Path) -> tuple[Any, Any, dict[str, object]]:
    models = pickle.loads((artifact_dir / "models.pkl").read_bytes())
    calibrators = pickle.loads((artifact_dir / "calibrators.pkl").read_bytes())
    card = json.loads((artifact_dir / "model_card.json").read_text(encoding="utf-8"))
    return models, calibrators, card


def write_prediction_artifact(frame: pd.DataFrame, output: Path, metadata: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8"
    )
