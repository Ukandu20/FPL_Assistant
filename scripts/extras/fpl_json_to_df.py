#!/usr/bin/env python3
"""
transform_fpl_jsons_to_dataframes.py
===================================
Load saved FPL JSON files into pandas DataFrames for analysis.

Directory structure assumed:
  data/raw/fpl/<season>/bootstrap-static.json
  data/raw/fpl/<season>/fixtures.json
  data/raw/fpl/<season>/events.json
  data/raw/fpl/<season>/element_types.json
  data/raw/fpl/<season>/gameweek_live/event_<id>_live.json
  data/raw/fpl/<season>/player_summaries/<player_id>.json

Usage:
    from transform_fpl_jsons_to_dataframes import (
        load_bootstrap,
        load_fixtures,
        load_events,
        load_element_types,
        load_gameweek_live,
        load_player_history,
        build_player_table,
    )

Functions will return pandas DataFrames.
"""
import json
from pathlib import Path
import pandas as pd
from pandas import json_normalize


def load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


from typing import Tuple

def load_bootstrap(season_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load players, teams, and element_types tables from bootstrap-static.json"""
    data = load_json(season_dir / 'bootstrap-static.json')
    players = json_normalize(data['elements'])
    teams = json_normalize(data['teams'])
    element_types = json_normalize(data['element_types'])
    # Rename for clarity
    teams = teams.rename(columns={'id':'team_id'})
    element_types = element_types.rename(columns={'id':'element_type'})
    return players, teams, element_types


def load_fixtures(season_dir: Path) -> pd.DataFrame:
    """Load fixture list"""
    fixtures = load_json(season_dir / 'fixtures.json')
    return json_normalize(fixtures)


def load_events(season_dir: Path) -> pd.DataFrame:
    """Load gameweek metadata"""
    events = load_json(season_dir / 'events.json')
    return json_normalize(events)


def load_element_types(season_dir: Path) -> pd.DataFrame:
    """Load element types (duplicate of bootstrap, if separate file)"""
    et = load_json(season_dir / 'element_types.json')
    return json_normalize(et)


def load_gameweek_live(season_dir: Path) -> pd.DataFrame:
    """Flatten all per-GW live stats into one DataFrame"""
    live_dir = season_dir / 'gameweek_live'
    rows = []
    for f in live_dir.glob('event_*_live.json'):
        data = load_json(f)
        event_id = data.get('event') or f.stem.split('_')[1]
        stats = data.get('elements', [])
        df = json_normalize(stats)
        df['event'] = event_id
        rows.append(df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def load_player_history(season_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten all player history (per-GW and past seasons) into DataFrames"""
    players_dir = season_dir / 'player_summaries'
    history_rows, past_rows = [], []
    for f in players_dir.glob('*.json'):
        data = load_json(f)
        player_id = data.get('id')
        # history: per-GW
        for rec in data.get('history', []):
            rec['player_id'] = player_id
            history_rows.append(rec)
        # history_past: season totals
        for rec in data.get('history_past', []):
            rec['player_id'] = player_id
            past_rows.append(rec)
    df_hist = pd.DataFrame(history_rows)
    df_past = pd.DataFrame(past_rows)
    return df_hist, df_past


def build_player_table(players: pd.DataFrame, teams: pd.DataFrame, element_types: pd.DataFrame) -> pd.DataFrame:
    """Merge players with team and position info"""
    df = players.merge(teams[['team_id','name']], left_on='team', right_on='team_id', how='left')
    df = df.merge(element_types[['element_type','singular_name']], left_on='element_type', right_on='element_type', how='left')
    return df.rename(columns={'name':'team_name','singular_name':'position'})
