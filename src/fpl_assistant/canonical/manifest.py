from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


FULL_PROVIDERS = frozenset({"fpl", "understat", "clubelo", "fbref", "whoscored"})
STANDARD_PROVIDERS = frozenset({"fpl", "understat", "clubelo"})
MINIMUM_PROVIDERS = frozenset({"fpl", "clubelo"})


def determine_run_mode(provider_status: Mapping[str, Any]) -> str:
    """Return the strongest supported operating mode for available providers."""

    available = {
        provider
        for provider, status in provider_status.items()
        if status is True
        or str(status).lower() in {"ok", "available", "complete", "success"}
    }
    if FULL_PROVIDERS <= available:
        return "full"
    if STANDARD_PROVIDERS <= available:
        return "standard"
    if MINIMUM_PROVIDERS <= available:
        return "minimum"
    missing = sorted(MINIMUM_PROVIDERS - available)
    raise ValueError(
        "Insufficient provider coverage for a minimum run; missing "
        + ", ".join(missing)
    )


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    sha256: str
    bytes: int


@dataclass
class RunManifest:
    run_id: str
    mode: str
    started_at: str
    completed_at: str | None = None
    git_commit: str | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    provider_status: dict[str, Any] = field(default_factory=dict)
    contracts: dict[str, str] = field(default_factory=dict)
    feature_versions: dict[str, str] = field(default_factory=dict)
    inputs: list[ArtifactReference] = field(default_factory=list)
    outputs: list[ArtifactReference] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        mode: str,
        configuration: Mapping[str, Any] | None = None,
    ) -> "RunManifest":
        if mode not in {"full", "standard", "minimum"}:
            raise ValueError("Run mode must be full, standard, or minimum.")
        return cls(
            run_id=run_id,
            mode=mode,
            started_at=datetime.now(timezone.utc).isoformat(),
            git_commit=_git_commit(),
            configuration=dict(configuration or {}),
        )

    def add_artifact(self, path: Path, *, output: bool) -> None:
        resolved = path.resolve()
        reference = ArtifactReference(
            path=str(resolved),
            sha256=file_checksum(resolved),
            bytes=resolved.stat().st_size,
        )
        target = self.outputs if output else self.inputs
        target.append(reference)

    def finish(self) -> None:
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path
