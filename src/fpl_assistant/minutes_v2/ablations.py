from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationSpec:
    name: str
    feature_overrides: dict[str, list[str]]
    canonical: bool = False


ABLATIONS = {
    "played_last": AblationSpec("played_last", {head: ["played_last"] for head in ("start", "start_minutes", "cameo", "cameo_minutes", "p60")}),
    "long_gap14": AblationSpec("long_gap14", {head: ["long_gap14"] for head in ("start", "start_minutes", "cameo", "cameo_minutes", "p60")}),
    "fdr": AblationSpec("fdr", {head: ["fdr"] for head in ("start", "start_minutes", "cameo", "cameo_minutes", "p60")}),
    "team_rot3": AblationSpec("team_rot3", {head: ["team_rot3"] for head in ("start", "start_minutes", "cameo", "cameo_minutes", "p60")}),
}


def get_ablation(name: str | None) -> AblationSpec | None:
    if not name:
        return None
    if name not in ABLATIONS:
        raise ValueError(f"Unknown isolated V2.0 ablation {name!r}; choose from {sorted(ABLATIONS)}")
    return ABLATIONS[name]
