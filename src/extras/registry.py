# registry.py
#!/usr/bin/env python3
"""
scripts/build_player_registry.py & build_team_registry.py
"""
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.registry_updater import update_player_registry, update_team_registry
from src.conflict_resolver import (
    detect_player_conflicts, print_player_conflicts,
    detect_team_conflicts, print_team_conflicts
)


def main():
    season = "2024_25"

    # Player registry
    player_path = Path(f"data/processed/players/fbref/{season}/standard_stats.csv")
    df = pd.read_csv(player_path)
    df.rename(columns={'player': 'player_name', 'born': 'dob', 'squad': 'club'}, inplace=True)
    names = df['player_name'].astype(str).str.strip()
    df['first_name'] = names.str.split().str[0]
    df['last_name']  = names.str.split().str[-1]

    # Manual club‐code map (shared)
    club_code_map = {
        "Arsenal":       "ARS",
        "Aston Villa":   "AVL",
        "Bournemouth":   "BOU",
        "Brentford":     "BRE",
        "Brighton":      "BHA",
        "Chelsea":       "CHE",
        "Crystal Palace":"CRY",
        "Everton":       "EVE",
        "Fulham":        "FUL",
        "Ipswich Town":  "IPS",
        "Leicester City":"LEI",
        "Liverpool":     "LIV",
        "Manchester City":"MCI",
        "Manchester Utd":"MUN",
        "Newcastle Utd": "NEW",
        "Nott'ham Forest": "NFO",
        "Southampton":   "SOU",
        "Tottenham":     "TOT",
        "West Ham":      "WHU",
        "Wolves":        "WOL",
        # …add others as needed…
    }
    df['club_code'] = df['club'].map(club_code_map).fillna("UNK")

    registry_p = update_player_registry(df, season)
    pc = detect_player_conflicts(registry_p)
    print_player_conflicts(pc)

    # Team registry
    team_path = Path(f"data/processed/teams/fbref/{season}/squad_standard_stats.csv")
    tdf = pd.read_csv(team_path)
    tdf.rename(columns={'squad': 'team_name'}, inplace=True)
    tdf['season'] = season
    tdf['club_code'] = tdf['team_name'].map(club_code_map).fillna("UNK")

    registry_t = update_team_registry(tdf, season)
    tc = detect_team_conflicts(registry_t)
    print_team_conflicts(tc)

    print(f"\n✅ Player registry: registry/players/{season}/database.csv ({len(registry_p)} rows)")
    print(f"✅ Team registry:   registry/teams/{season}/database.csv ({len(registry_t)} rows)")

if __name__ == "__main__":
    main()

