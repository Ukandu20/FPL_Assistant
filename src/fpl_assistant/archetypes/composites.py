from __future__ import annotations

from collections.abc import Mapping


OUTFIELD_LABELS: dict[str, dict[frozenset[str], str]] = {
    "FWD": {
        frozenset("G"): "Finisher",
        frozenset("C"): "Creative Forward",
        frozenset("D"): "Pressing Forward",
        frozenset(("G", "C")): "Complete Forward",
        frozenset(("G", "D")): "Pressing Finisher",
        frozenset(("C", "D")): "Pressing Link Forward",
        frozenset(("G", "C", "D")): "Complete Two-Way Forward",
    },
    "MID": {
        frozenset("G"): "Goal-Scoring Midfielder",
        frozenset("C"): "Playmaker",
        frozenset("D"): "Ball-Winning Midfielder",
        frozenset(("G", "C")): "Attacking Playmaker",
        frozenset(("G", "D")): "Box-to-Box Midfielder",
        frozenset(("C", "D")): "Deep-Lying Playmaker",
        frozenset(("G", "C", "D")): "Complete Midfielder",
    },
    "DEF": {
        frozenset("G"): "Goal-Threat Defender",
        frozenset("C"): "Creative Defender",
        frozenset("D"): "Defensive Stopper",
        frozenset(("G", "C")): "Attacking Defender",
        frozenset(("G", "D")): "Two-Way Defender",
        frozenset(("C", "D")): "Defensive Creator",
        frozenset(("G", "C", "D")): "Complete Defender",
    },
}


def resolve_production_composite(
    position: str,
    components: Mapping[str, Mapping[str, object]],
) -> dict[str, object] | None:
    normalized = position.upper()
    if normalized == "GK":
        normalized = "GKP"
    if normalized == "GKP":
        saves = components.get("SAVES_MACHINE", {})
        if not bool(saves.get("active", False)):
            return None
        return {
            "id": "PRODSTYLE_GKP_S",
            "display_name": "Shot Stopper",
            "score": float(saves["score"]),
            "confidence": float(saves["confidence"]),
            "components": ["SAVES_MACHINE"],
        }
    if normalized not in OUTFIELD_LABELS:
        raise ValueError(f"Unsupported FPL position: {position}")
    code_to_id = {"G": "GOAL_THREAT", "C": "CREATOR", "D": "DEFENSIVE_ENGINE"}
    active_codes = frozenset(
        code for code, component_id in code_to_id.items()
        if bool(components.get(component_id, {}).get("active", False))
    )
    if not active_codes:
        return None
    ordered_codes = [code for code in ("G", "C", "D") if code in active_codes]
    required_ids = [code_to_id[code] for code in ordered_codes]
    label = OUTFIELD_LABELS[normalized][active_codes]
    return {
        "id": f"PRODSTYLE_{normalized}_{'_'.join(ordered_codes)}",
        "display_name": label,
        "score": min(float(components[item]["score"]) for item in required_ids),
        "confidence": min(float(components[item]["confidence"]) for item in required_ids),
        "components": required_ids,
    }


__all__ = ["OUTFIELD_LABELS", "resolve_production_composite"]
