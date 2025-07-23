# conflict_resolver.py
from collections import defaultdict

def detect_player_conflicts(registry_df):
    name_map = defaultdict(list)
    for _, row in registry_df.iterrows():
        key = str(row['player_name']).lower().strip()
        name_map[key].append((row['player_id'], row['dob'], row['club']))
    return { name: entries for name, entries in name_map.items() if len({e[1] for e in entries}) > 1 }

def print_player_conflicts(conflicts):
    for name, entries in conflicts.items():
        print(f"\n⚠️ Player conflict for '{name}':")
        for pid, dob, club in entries:
            print(f" - ID: {pid}, DOB: {dob}, Club: {club}")


def detect_team_conflicts(registry_df):
    season_map = defaultdict(list)
    for _, row in registry_df.iterrows():
        key = str(row['team_name']).lower().strip()
        season_map[key].append(row['season'])
    return { team: seasons for team, seasons in season_map.items() if len(seasons) > 1 }

def print_team_conflicts(conflicts):
    for team, seasons in conflicts.items():
        print(f"\n⚠️ Team conflict for '{team}': appears in seasons {seasons}")
