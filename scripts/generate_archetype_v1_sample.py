"""Generate a deterministic four-position V1 archetype example artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fpl_assistant.archetypes.persistence import persist_snapshot
from fpl_assistant.archetypes.pipeline import build_archetype_snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AS_OF = "2026-06-03T00:00:00Z"


def sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    values: list[dict[str, object]] = []
    for position_index, position in enumerate(("FWD", "MID", "DEF", "GKP")):
        for peer in range(1, 6):
            player_id = f"sample-{position.lower()}-{peer}"
            strength = peer / 5
            for appearance in range(24):
                is_home = appearance % 2 == 0
                difficulty = (appearance % 8) / 7
                base = 1.0 + strength * 2.0
                points = 10 if peer == 5 and appearance % 5 == 0 else int(round(base + 2))
                rows.append(
                    {
                        "match_id": f"{player_id}-m{appearance + 1}",
                        "player_id": player_id, "team_id": f"team-{position_index}-{peer}",
                        "season": "2025-2026", "gameweek": appearance + 1,
                        "kickoff_utc": pd.Timestamp("2025-08-01", tz="UTC") + pd.Timedelta(days=7 * appearance),
                        "fpl_position": position, "minutes": 90, "started": True,
                        "named_on_bench": False, "availability_status": "available",
                        "availability_reason": None, "is_home": is_home,
                        "venue": "Home" if is_home else "Away",
                        "npxg": 0.08 * base if position != "GKP" else None,
                        "non_penalty_goals": 0.04 * base if position != "GKP" else None,
                        "shots_in_box": 0.6 * base if position != "GKP" else None,
                        "shots_on_target": 0.3 * base if position != "GKP" else None,
                        "xa": 0.06 * base if position != "GKP" else None,
                        "key_passes": 0.5 * base if position != "GKP" else None,
                        "big_chances_created": 0.1 * base if position != "GKP" else None,
                        "shot_creating_actions": 0.8 * base if position != "GKP" else None,
                        "tackles_won": 0.6 * base if position != "GKP" else None,
                        "interceptions": 0.4 * base if position != "GKP" else None,
                        "clearances": 0.5 * base if position != "GKP" else None,
                        "blocks": 0.25 * base if position != "GKP" else None,
                        "recoveries": 0.8 * base if position != "GKP" else None,
                        "saves": 1.2 * base if position == "GKP" else None,
                        "shots_on_target_faced": 1.8 * base if position == "GKP" else None,
                        "goals_conceded": max(0.0, 2.0 - strength) if position in {"DEF", "GKP"} else None,
                        "post_shot_xg": (2.2 - strength / 2) if position == "GKP" else None,
                        "penalties_saved": 1 if position == "GKP" and peer == 5 and appearance == 0 else 0 if position == "GKP" else None,
                        "penalties_faced": 1 if position == "GKP" and appearance == 0 else 0 if position == "GKP" else None,
                        "clean_sheet": bool(position in {"DEF", "GKP"} and appearance % max(2, 7 - peer) == 0),
                        "xga": 1.8 - strength / 2 if position in {"DEF", "GKP"} else None,
                        "team_defence_index_pre_match": 0.8 + strength * 0.4 if position in {"DEF", "GKP"} else None,
                        "production_response": base - (0.15 * peer * difficulty) + (0.10 if is_home else 0),
                        "matchup_difficulty": difficulty,
                        "fpl_points": points, "return_event": points >= 5,
                        "yellow_cards": 1 if appearance < peer else 0,
                        "second_yellow_cards": 0, "red_cards": 0,
                    }
                )
            values.append(
                {
                    "player_id": player_id, "fpl_position": position,
                    "price": 40 + peer * 10,
                    "historical_value_over_replacement": peer * 2.0,
                    "expected_next5_value_over_replacement": peer * 1.5,
                    "historical_minutes": 2160, "historical_appearances": 24,
                    "expected_next5_minutes": 350,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(values)


def main() -> None:
    player_matches, player_values = sample_inputs()
    result = build_archetype_snapshot(
        player_matches, as_of=AS_OF, current_season="2025-2026",
        player_values=player_values,
    )
    output_root = PROJECT_ROOT / "artifacts" / "archetypes" / "1.0.0" / "examples"
    target = persist_snapshot(
        result.archetypes, result.team_ratings,
        output_root=output_root, snapshot_date=AS_OF, model_version="1.0.0",
        evidence_tables=result.evidence_tables,
    )
    representatives = [f"sample-{position}-5" for position in ("fwd", "mid", "def", "gkp")]
    sample = result.archetypes[
        result.archetypes["player_id"].isin(representatives)
        & (result.archetypes["active_label"] | result.archetypes["family"].eq("Usage"))
    ].to_dict("records")
    (target / "representative_players.json").write_text(
        json.dumps(sample, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(target)


if __name__ == "__main__":
    main()
