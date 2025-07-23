# registry_updater.py
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.id_generator import generate_player_id, generate_team_id


def update_player_registry(player_df: pd.DataFrame, season: str, registry_root: str = "registry/players") -> pd.DataFrame:
    """
    Build a fresh player registry for the given season, overwriting any existing file.
    """
    out_dir = Path(registry_root) / season
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "database.csv"

    # Generate stable player_id
    df = player_df.copy()
    df['player_id'] = df.apply(
        lambda row: generate_player_id(
            row.get('player_name', ''),
            str(int(row.get('dob'))) if pd.notna(row.get('dob')) else '',
            '', ''
        ),
        axis=1
    )

    # Ensure dob is Int64
    df['dob'] = pd.to_numeric(df['dob'], errors='coerce').round(0).astype('Int64')

    # Select and dedupe
    registry = df[[
        'player_id', 'player_name', 'first_name', 'last_name',
        'nation', 'club', 'club_code', 'dob'
    ]].drop_duplicates(subset=['player_id'])

    registry.to_csv(path, index=False)
    return registry


def update_team_registry(team_df: pd.DataFrame, season: str, registry_root: str = "registry/teams") -> pd.DataFrame:
    """
    Build a fresh team registry for the given season, overwriting any existing file.
    """
    out_dir = Path(registry_root) / season
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "database.csv"

    df = team_df.copy()
    df['team_id'] = df.apply(
        lambda row: generate_team_id(
            row.get('team_name', ''), season
        ),
        axis=1
    )

    registry = df[['team_id', 'team_name', 'season', 'club_code']].drop_duplicates(subset=['team_id'])

    registry.to_csv(path, index=False)
    return registry

