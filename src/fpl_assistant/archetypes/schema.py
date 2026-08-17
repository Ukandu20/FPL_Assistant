from __future__ import annotations

from fpl_assistant.canonical.contracts import ColumnContract, TableContract


ARCHETYPE_PLAYER_MATCH_CONTRACT = TableContract(
    name="archetype_player_match",
    version="1.0.0",
    key=("match_id", "player_id"),
    columns={
        "match_id": ColumnContract(nullable=False),
        "player_id": ColumnContract(nullable=False),
        "team_id": ColumnContract(nullable=False),
        "season": ColumnContract(nullable=False),
        "kickoff_utc": ColumnContract(nullable=False),
        "fpl_position": ColumnContract(nullable=False, allowed=("GKP", "DEF", "MID", "FWD")),
        "minutes": ColumnContract(nullable=False, minimum=0, maximum=120),
        "started": ColumnContract(),
        "availability_status": ColumnContract(),
        "fpl_points": ColumnContract(),
        "price": ColumnContract(minimum=0),
        "npxg": ColumnContract(minimum=0),
        "non_penalty_goals": ColumnContract(minimum=0),
        "xa": ColumnContract(minimum=0),
        "shots_on_target": ColumnContract(minimum=0),
        "shots_in_box": ColumnContract(minimum=0),
        "key_passes": ColumnContract(minimum=0),
        "big_chances_created": ColumnContract(minimum=0),
        "shot_creating_actions": ColumnContract(minimum=0),
        "tackles_won": ColumnContract(minimum=0),
        "interceptions": ColumnContract(minimum=0),
        "clearances": ColumnContract(minimum=0),
        "blocks": ColumnContract(minimum=0),
        "recoveries": ColumnContract(minimum=0),
        "clean_sheet": ColumnContract(),
        "goals_conceded": ColumnContract(minimum=0),
        "xga": ColumnContract(minimum=0),
        "saves": ColumnContract(minimum=0),
        "shots_on_target_faced": ColumnContract(minimum=0),
        "post_shot_xg": ColumnContract(minimum=0),
        "penalties_saved": ColumnContract(minimum=0),
        "penalties_faced": ColumnContract(minimum=0),
        "yellow_cards": ColumnContract(minimum=0),
        "second_yellow_cards": ColumnContract(minimum=0),
        "red_cards": ColumnContract(minimum=0),
        "production_response": ColumnContract(),
        "matchup_difficulty": ColumnContract(),
        "return_event": ColumnContract(),
    },
)


ARCHETYPE_OUTPUT_CONTRACT = TableContract(
    name="player_archetype_snapshot",
    version="1.0.0",
    key=("player_id", "snapshot_date", "archetype_id", "model_version"),
    columns={
        "player_id": ColumnContract(nullable=False),
        "snapshot_date": ColumnContract(nullable=False),
        "archetype_id": ColumnContract(nullable=False),
        "display_name": ColumnContract(nullable=False),
        "family": ColumnContract(nullable=False),
        "score_0_100": ColumnContract(minimum=0, maximum=100),
        "active_label": ColumnContract(nullable=False),
        "confidence_0_1": ColumnContract(nullable=False, minimum=0, maximum=1),
        "confidence_band": ColumnContract(nullable=False, allowed=("Insufficient", "Low", "Medium", "High")),
        "trend": ColumnContract(),
        "status": ColumnContract(nullable=False),
        "evidence_minutes": ColumnContract(nullable=False, minimum=0),
        "eligible_appearances": ColumnContract(nullable=False, minimum=0),
        "component_scores": ColumnContract(),
        "missing_data_flags": ColumnContract(),
        "model_version": ColumnContract(nullable=False),
    },
)


__all__ = ["ARCHETYPE_OUTPUT_CONTRACT", "ARCHETYPE_PLAYER_MATCH_CONTRACT"]
