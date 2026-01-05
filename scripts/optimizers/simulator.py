#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build MC sim_input Parquets/CSVs for one or many candidate rosters over a GW horizon.

Inputs accept native column names (e.g., opponent_id, gw_orig). We resolve aliases per table
and only create small *views* with canonical names for joins/output. Source DataFrames are not mutated.

Output schema (per row; consumed by mc.py):
[
  'gw','fixture_id','team_id','opp_team_id','player_id','player','pos',
  'xg_share','xa_share','lambda_goals_for','lambda_goals_against',
  'pred_minutes','p60','pred_saves_mean','is_start_xi','bench_order','is_bench_gk',
  'is_captain','is_vice'
]

USAGE (single candidate):
py -m scripts.optimizers.simulator ^
  --team-state data/state/team_state.json ^
  --minutes data/predictions/minutes/2025-2026/GW4_6.csv ^
  --goals-assists data/predictions/goals_assists/2025-2026/GW4_6.csv ^
  --defense data/predictions/defense/2025-2026/GW4_6.csv ^
  --saves data/predictions/saves/2025-2026/GW4_6.csv ^
  --fixtures data/processed/registry/fixtures/2025-2026/fixture_calendar.csv ^
  --season 2025-2026 --gws 4,5,6 ^
  --out-dir data/decisions/candidates/A0/hold ^
  --out-format parquet

USAGE (batch: many candidates at once; candidate label inferred from parent dir name):
py -m scripts.optimizers.simulator ^
  --team-state-glob data/state/candidates/*/team_state.json ^
  --candidate-subdir hold ^
  --minutes data/predictions/minutes/2025-2026/GW4_6.csv ^
  --goals-assists data/predictions/goals_assists/2025-2026/GW4_6.csv ^
  --defense data/predictions/defense/2025-2026/GW4_6.csv ^
  --saves data/predictions/saves/2025-2026/GW4_6.csv ^
  --fixtures data/processed/registry/fixtures/2025-2026/fixture_calendar.csv ^
  --season 2025-2026 --gws 4,5,6 ^
  --out-dir data/decisions/candidates ^
  --out-format parquet
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ----------------------------- Constants -------------------------------------

REQ_OUT_COLS = [
    'gw','fixture_id','team_id','opp_team_id','player_id','player','pos',
    'xg_share','xa_share','lambda_goals_for','lambda_goals_against',
    'pred_minutes','p60','pred_saves_mean','is_start_xi','bench_order','is_bench_gk',
    'is_captain','is_vice'
]

POS_SET = {'GK','DEF','MID','FWD'}

# Aliases accepted from upstream tables (inputs)
ALIASES: Dict[str, List[str]] = {
    "season": ["season", "szn"],
    "gw": ["gw", "gw_orig", "GW", "round", "gameweek"],
    "team_id": ["team_id", "team", "team_code", "squad_id"],
    "opp_team_id": ["opp_team_id", "opponent_id", "opp_id", "opp"],
    "fixture_id": ["fixture_id", "fbref_id", "match_id"],
    "player_id": ["player_id", "id", "element"],
    "player": ["player", "name"],
    "pos": ["pos", "position", "element_type"],
    "pred_minutes": ["pred_minutes", "minutes_mean", "min_pred"],
    "p60": ["p60", "prob_60", "p_start", "p_started"],
    "pred_saves_mean": ["pred_saves_mean", "saves_mean"],
    "pred_goals_mean": ["pred_goals_mean", "goals_mean", "xg_mean"],
    "pred_assists_mean": ["pred_assists_mean", "assists_mean", "xa_mean"],
    "lambda_goals_for": ["lambda_goals_for", "team_exp_goals_for"],
    "lambda_goals_against": ["lambda_goals_against", "team_exp_goals_against", "exp_gc"],
}

# ----------------------------- Helpers ---------------------------------------

def read_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_team_state(path: Path) -> pd.DataFrame:
    js = read_json(path)
    squad = js.get('squad') or js.get('players') or []
    if not squad:
        raise ValueError(f"team_state JSON missing 'squad'/'players' array: {path}")
    df = pd.DataFrame(squad)
    # Map common fields -> internal names (only inside this small DF)
    if 'player' not in df.columns and 'name' in df.columns:
        df = df.rename(columns={'name':'player'})
    if 'pos' not in df.columns and 'element_type' in df.columns:
        df = df.rename(columns={'element_type':'pos'})
    if 'team_id' not in df.columns and 'team' in df.columns:
        df = df.rename(columns={'team':'team_id'})
    need = {'player_id','player','pos'}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"team_state must include columns {need}, found {set(df.columns)} at {path}")
    # normalize types and casing
    df['player_id'] = df['player_id'].astype(str)
    df['player'] = df['player'].astype(str)
    df['pos'] = df['pos'].astype(str).str.upper().str[:3]
    for c in ['is_start_xi','bench_order','is_bench_gk']:
        if c not in df.columns:
            df[c] = np.nan
    return df

def parse_gws(gws_str: str) -> List[int]:
    parts = [p.strip() for p in gws_str.split(',') if p.strip()]
    gws = [int(p) for p in parts]
    if not gws:
        raise ValueError("No GWs parsed from --gws")
    return gws

def pick_col(df: pd.DataFrame, logical: str) -> str:
    for cand in ALIASES.get(logical, []):
        if cand in df.columns:
            return cand
    raise KeyError(f"Missing column for '{logical}'. Looked for aliases {ALIASES.get(logical)} in {list(df.columns)}")

def select_view(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Return a small view with canonical names for keys in `mapping`.
    mapping: logical_name -> alias_to_use_in_df
    """
    cols_in = [mapping[k] for k in mapping]
    ren_map = {mapping[k]: k for k in mapping}
    out = df[cols_in].rename(columns=ren_map).copy()
    # normalize some canonical dtypes
    for col in ['team_id','opp_team_id','fixture_id','player_id']:
        if col in out.columns:
            out[col] = out[col].astype(str)
    if 'pos' in out.columns:
        out['pos'] = out['pos'].astype(str).str.upper().str[:3]
    return out

def ensure_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    if 'season' in df.columns:
        return df[df['season'] == season].copy()
    d = df.copy()
    d['season'] = season
    return d

def require_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

def ensure_col(df: pd.DataFrame, col: str, default):
    """Ensure column exists; if missing create with default; else fillna(default)."""
    if col not in df.columns:
        df[col] = default
    else:
        df[col] = df[col].fillna(default)
    return df

def derive_lineup(roster: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    df = roster.merge(
        metrics[['player_id','p60','pred_minutes','pos']],
        on=['player_id','pos'], how='left'
    )
    df['__score'] = df['p60'].clip(0,1).fillna(0) * df['pred_minutes'].clip(0,95).fillna(0)

    # If any is_start_xi provided, respect them; else infer XI
    if df['is_start_xi'].notna().any():
        df['is_start_xi'] = df['is_start_xi'].fillna(False).astype(bool)
    else:
        # Pick XI: 1 GK, then top 10 outfield by score
        gks = df[df['pos']=='GK'].sort_values('__score', ascending=False)
        start_ids = set(gks['player_id'].head(1).tolist())
        outfield = df[df['pos']!='GK'].sort_values('__score', ascending=False)
        start_ids.update(outfield['player_id'].head(10).tolist())
        df['is_start_xi'] = df['player_id'].isin(start_ids)

    # is_bench_gk
    if df['is_bench_gk'].notna().any():
        df['is_bench_gk'] = df['is_bench_gk'].fillna(False).astype(bool)
    else:
        df['is_bench_gk'] = False
        if (df['pos']=='GK').any():
            gk_non_xi = df[(df['pos']=='GK') & (~df['is_start_xi'])]
            if not gk_non_xi.empty:
                df.loc[df['player_id'].isin(gk_non_xi['player_id']), 'is_bench_gk'] = True

    # bench_order (outfield bench ordered by score, highest = bench1)
    if df['bench_order'].notna().any():
        df['bench_order'] = df['bench_order'].fillna(0).astype(int)
    else:
        bench_out = df[(~df['is_start_xi']) & (df['pos']!='GK')].sort_values('__score', ascending=False)
        for i, pid in enumerate(bench_out['player_id'].tolist(), 1):
            df.loc[df['player_id']==pid, 'bench_order'] = i
        df['bench_order'] = df['bench_order'].fillna(0).astype(int)

    df['is_captain'] = False
    df['is_vice'] = False
    return df.drop(columns=['__score'])

def write_out(df: pd.DataFrame, base: Path, gw: int, fmt: str):
    base.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        path = base.with_name(f"gw{gw}.csv")
        df.to_csv(path, index=False, encoding='utf-8')
    else:
        path = base.with_name(f"gw{gw}.parquet")
        try:
            df.to_parquet(path, index=False)
        except Exception as e:
            raise SystemExit(f"to_parquet failed ({e}). Install `pyarrow` or use --out-format csv.")
    print(f"Wrote {path} ({len(df)} rows)")

# -------------------------- Builders -----------------------------------------

def build_from_components(
    season: str,
    gws: List[int],
    roster: pd.DataFrame,
    minutes_df: pd.DataFrame,
    ga_df: pd.DataFrame,
    def_df: pd.DataFrame,
    saves_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    out_dir: Path,
    out_format: str
):
    # --- Seasonize inputs (don’t mutate originals) ---
    minutes_df  = ensure_season(minutes_df, season)
    ga_df       = ensure_season(ga_df, season)
    def_df      = ensure_season(def_df, season)
    saves_df    = ensure_season(saves_df, season) if not saves_df.empty else saves_df
    fixtures_df = ensure_season(fixtures_df, season)

    # --- Canonical views (using your native names via alias resolution) ---

    # Fixtures view
    fx_map = {
        'season':      pick_col(fixtures_df, 'season'),
        'gw':          pick_col(fixtures_df, 'gw'),           # accepts gw_orig
        'team_id':     pick_col(fixtures_df, 'team_id'),
        'opp_team_id': pick_col(fixtures_df, 'opp_team_id'),  # accepts opponent_id
        'fixture_id':  pick_col(fixtures_df, 'fixture_id'),   # accepts fbref_id
    }
    fixtures_v = select_view(fixtures_df, fx_map)

    # Minutes view (no 'player' here; we bring 'player' from roster)
    try:
        m_map = {
            'season':       pick_col(minutes_df, 'season'),
            'gw':           pick_col(minutes_df, 'gw'),
            'team_id':      pick_col(minutes_df, 'team_id'),
            'player_id':    pick_col(minutes_df, 'player_id'),
            'pos':          pick_col(minutes_df, 'pos'),
            'pred_minutes': pick_col(minutes_df, 'pred_minutes'),
            'p60':          pick_col(minutes_df, 'p60'),
        }
    except KeyError as e:
        raise ValueError(f"minutes table missing required alias: {e}")
    minutes_v = select_view(minutes_df, m_map)
    minutes_v['p60'] = minutes_v['p60'].astype(float).clip(0, 1)

    # GA view
    ga_map = {
        'season':            pick_col(ga_df, 'season'),
        'gw':                pick_col(ga_df, 'gw'),
        'team_id':           pick_col(ga_df, 'team_id'),
        'player_id':         pick_col(ga_df, 'player_id'),
        'pred_goals_mean':   pick_col(ga_df, 'pred_goals_mean'),
        'pred_assists_mean': pick_col(ga_df, 'pred_assists_mean'),
    }
    ga_v = select_view(ga_df, ga_map)

    # Defense lambdas (team-level)
    def_has_any = any(c in def_df.columns for c in (ALIASES['lambda_goals_for'] + ALIASES['lambda_goals_against']))
    if def_has_any:
        d_map = {
            'season':  pick_col(def_df, 'season'),
            'gw':      pick_col(def_df, 'gw'),
            'team_id': pick_col(def_df, 'team_id'),
        }
        for cand in ALIASES['lambda_goals_for']:
            if cand in def_df.columns:
                d_map['lambda_goals_for'] = cand
                break
        for cand in ALIASES['lambda_goals_against']:
            if cand in def_df.columns:
                d_map['lambda_goals_against'] = cand
                break
        defense_v = select_view(def_df, d_map)
    else:
        defense_v = pd.DataFrame(columns=['season','gw','team_id'])

    # Saves view (GK-only). It can be empty; that’s fine.
    if not saves_df.empty:
        s_map = {
            'season':          pick_col(saves_df, 'season'),
            'gw':              pick_col(saves_df, 'gw'),
            'player_id':       pick_col(saves_df, 'player_id'),
            'pred_saves_mean': pick_col(saves_df, 'pred_saves_mean'),
        }
        saves_v = select_view(saves_df, s_map)
        saves_small = saves_v[['season','gw','player_id','pred_saves_mean']].drop_duplicates()
    else:
        saves_small = pd.DataFrame(columns=['season','gw','player_id','pred_saves_mean'])

    # --- Precompute team-level λ table from defense (or fallbacks) ---
    teams = fixtures_v[['season','gw','team_id','opp_team_id','fixture_id']].copy()

    if {'lambda_goals_for','lambda_goals_against'}.issubset(defense_v.columns):
        lam = defense_v[['season','gw','team_id','lambda_goals_for','lambda_goals_against']].drop_duplicates(subset=['season','gw','team_id'])
        teams = teams.merge(lam, on=['season','gw','team_id'], how='left')
        if teams['lambda_goals_against'].isna().any():
            opp = teams[['season','gw','team_id','lambda_goals_for']].rename(columns={
                'team_id':'opp_team_id','lambda_goals_for':'lambda_goals_against_opp'
            })
            teams = teams.merge(opp, on=['season','gw','opp_team_id'], how='left')
            teams['lambda_goals_against'] = teams['lambda_goals_against'].fillna(teams['lambda_goals_against_opp'])
            teams = teams.drop(columns=['lambda_goals_against_opp'])
    elif 'lambda_goals_against' in defense_v.columns and 'lambda_goals_for' not in defense_v.columns:
        # exp_gc style: have against; derive for from opponent's against
        lam_me = defense_v[['season','gw','team_id','lambda_goals_against']]
        teams = teams.merge(lam_me, on=['season','gw','team_id'], how='left')
        lam_opp = lam_me.rename(columns={'team_id':'opp_team_id','lambda_goals_against':'lambda_goals_for'})
        teams = teams.merge(lam_opp, on=['season','gw','opp_team_id'], how='left')
    else:
        # Fallback from GA totals
        ga_tot = ga_v.groupby(['season','gw','team_id'], as_index=False)[['pred_goals_mean']].sum()
        ga_tot = ga_tot.rename(columns={'pred_goals_mean':'lambda_goals_for'})
        teams = teams.merge(ga_tot, on=['season','gw','team_id'], how='left')
        lam_opp = ga_tot.rename(columns={'team_id':'opp_team_id','lambda_goals_for':'lambda_goals_against'})
        teams = teams.merge(lam_opp, on=['season','gw','opp_team_id'], how='left')

    # Team totals for GA shares
    ga_tot_team = ga_v.groupby(['season','gw','team_id'], as_index=False)[['pred_goals_mean','pred_assists_mean']].sum()
    ga_tot_team = ga_tot_team.rename(columns={'pred_goals_mean':'team_goals_mean_total','pred_assists_mean':'team_assists_mean_total'})

    # --- Loop per GW and write parquet/csv ---
    base = Path(out_dir) / "gw0.dummy"  # only for naming; replaced in writer
    for gw in gws:
        m = minutes_v[(minutes_v['season']==season) & (minutes_v['gw']==gw)].copy()

        # attach roster 'player' (trust roster naming) by player_id + pos
        m = m.merge(roster[['player_id','player','pos']], on=['player_id','pos'], how='inner')

        if m.empty:
            raise ValueError(f"No minutes rows for roster players in GW{gw}. Check inputs/aliases.")

        # Join fixtures + lambdas
        m = m.merge(teams[teams['gw']==gw], on=['season','gw','team_id'], how='left')
        require_cols(m, ['opp_team_id','fixture_id','lambda_goals_for','lambda_goals_against'], f'minutes+teams GW{gw}')

        # Per-player GA
        ga = ga_v[(ga_v['season']==season) & (ga_v['gw']==gw)][['player_id','team_id','pred_goals_mean','pred_assists_mean']]
        m = m.merge(ga, on=['player_id','team_id'], how='left')

        # Team totals for shares
        tot = ga_tot_team[ga_tot_team['gw']==gw][['team_id','team_goals_mean_total','team_assists_mean_total']]
        m = m.merge(tot, on='team_id', how='left')

        # Shares
        m['xg_share'] = (m['pred_goals_mean'].clip(lower=0).fillna(0) /
                         m['team_goals_mean_total'].replace(0, np.nan))
        m['xa_share'] = (m['pred_assists_mean'].clip(lower=0).fillna(0) /
                         m['team_assists_mean_total'].replace(0, np.nan))
        m['xg_share'] = m['xg_share'].fillna(0.0)
        m['xa_share'] = m['xa_share'].fillna(0.0)

        # Owned coverage
        cov = m.groupby('team_id', as_index=False)[['xg_share','xa_share']].sum().rename(columns={'xg_share':'xg_cov','xa_share':'xa_cov'})
        m = m.merge(cov, on='team_id', how='left')

        # Scale λ_for by owned xG coverage
        m['lambda_goals_for'] = (m['lambda_goals_for'] * m['xg_cov'].clip(0, 1.0)).fillna(0.0)

        # Saves (safe even if saves_small is empty)
        if not saves_small.empty:
            m = m.merge(saves_small[saves_small['gw']==gw], on=['season','gw','player_id'], how='left')
        m = ensure_col(m, 'pred_saves_mean', 0.0)

        # Lineup meta
        metrics = m[['player_id','pos','pred_minutes','p60']].drop_duplicates()
        roster_lineup = derive_lineup(roster[['player_id','player','pos','is_start_xi','bench_order','is_bench_gk']], metrics)

        out = m.merge(roster_lineup[['player_id','is_start_xi','bench_order','is_bench_gk']], on='player_id', how='left')

        out_df = out[['gw','fixture_id','team_id','opp_team_id','player_id','player','pos',
                      'xg_share','xa_share','lambda_goals_for','lambda_goals_against',
                      'pred_minutes','p60','pred_saves_mean','is_start_xi','bench_order','is_bench_gk']].copy()

        out_df['is_captain'] = False
        out_df['is_vice'] = False

        out_df['gw'] = out_df['gw'].astype(int)
        for c in ['is_start_xi','is_bench_gk','is_captain','is_vice']:
            out_df[c] = out_df[c].fillna(False).astype(bool)
        out_df['bench_order'] = out_df['bench_order'].fillna(0).astype(int)

        missing = [c for c in REQ_OUT_COLS if c not in out_df.columns]
        if missing:
            raise ValueError(f"Internal error: missing output cols {missing}")

        write_out(out_df, base, gw, out_format)

    print("Done. Sim inputs ready.")

def build_from_optimizer_input(
    season: str,
    gws: List[int],
    roster: pd.DataFrame,
    opt_df: pd.DataFrame,
    out_dir: Path,
    out_format: str
):
    opt_df = ensure_season(opt_df, season)

    # Canonical view (NO 'player'; we will source name from roster)
    opt_map = {
        'season':            pick_col(opt_df, 'season'),
        'gw':                pick_col(opt_df, 'gw'),
        'team_id':           pick_col(opt_df, 'team_id'),
        'opp_team_id':       pick_col(opt_df, 'opp_team_id'),
        'fixture_id':        pick_col(opt_df, 'fixture_id'),
        'player_id':         pick_col(opt_df, 'player_id'),
        'pos':               pick_col(opt_df, 'pos'),
        'pred_minutes':      pick_col(opt_df, 'pred_minutes'),
        'p60':               pick_col(opt_df, 'p60'),
        'pred_saves_mean':   pick_col(opt_df, 'pred_saves_mean'),
        'pred_goals_mean':   pick_col(opt_df, 'pred_goals_mean'),
        'pred_assists_mean': pick_col(opt_df, 'pred_assists_mean'),
        'lambda_goals_for':  pick_col(opt_df, 'lambda_goals_for'),
        'lambda_goals_against': pick_col(opt_df, 'lambda_goals_against'),
    }
    df = select_view(opt_df, opt_map)
    df['p60'] = df['p60'].astype(float).clip(0,1)

    need = ['season','gw','player_id','team_id','opp_team_id','fixture_id','pos',
            'pred_minutes','p60','pred_saves_mean','pred_goals_mean','pred_assists_mean',
            'lambda_goals_for','lambda_goals_against']
    require_cols(df, need, 'optimizer_input')

    base = Path(out_dir) / "gw0.dummy"
    for gw in gws:
        sub = df[df['gw']==gw].copy()
        # bring reliable player display name from roster
        sub = sub.merge(roster[['player_id','player','pos']], on=['player_id','pos'], how='inner')
        if sub.empty:
            raise ValueError(f"No optimizer_input rows for roster players in GW{gw}.")

        tot = sub.groupby(['team_id'], as_index=False)[['pred_goals_mean','pred_assists_mean']].sum().rename(
            columns={'pred_goals_mean':'team_goals_mean_total','pred_assists_mean':'team_assists_mean_total'}
        )
        sub = sub.merge(tot, on='team_id', how='left')

        sub['xg_share'] = (sub['pred_goals_mean'].clip(lower=0) / sub['team_goals_mean_total'].replace(0, np.nan)).fillna(0.0)
        sub['xa_share'] = (sub['pred_assists_mean'].clip(lower=0) / sub['team_assists_mean_total'].replace(0, np.nan)).fillna(0.0)

        cov = sub.groupby('team_id', as_index=False)['xg_share'].sum().rename(columns={'xg_share':'xg_cov'})
        sub = sub.merge(cov, on='team_id', how='left')
        sub['lambda_goals_for'] = (sub['lambda_goals_for'] * sub['xg_cov'].clip(0,1.0)).fillna(0.0)

        metrics = sub[['player_id','pos','pred_minutes','p60']].drop_duplicates()
        roster_lineup = derive_lineup(roster[['player_id','player','pos','is_start_xi','bench_order','is_bench_gk']], metrics)

        out_df = sub[['gw','fixture_id','team_id','opp_team_id','player_id','player','pos',
                      'xg_share','xa_share','lambda_goals_for','lambda_goals_against',
                      'pred_minutes','p60','pred_saves_mean']].copy()
        out_df = out_df.merge(roster_lineup[['player_id','is_start_xi','bench_order','is_bench_gk']], on='player_id', how='left')
        out_df['is_captain'] = False
        out_df['is_vice'] = False

        for c in ['is_start_xi','is_bench_gk','is_captain','is_vice']:
            out_df[c] = out_df[c].fillna(False).astype(bool)
        out_df['bench_order'] = out_df['bench_order'].fillna(0).astype(int)

        write_out(out_df, base, gw, out_format)

    print("Done. Sim inputs ready.")

# ------------------------------ Runner ---------------------------------------

def run_single_candidate(
    team_state_path: Path,
    season: str,
    gws: List[int],
    out_dir: Path,
    out_format: str,
    args
):
    roster = load_team_state(team_state_path)
    if not set(roster['pos']).issubset(POS_SET):
        raise SystemExit(f"Unexpected positions in team_state ({team_state_path}): {sorted(set(roster['pos']) - POS_SET)}")

    use_components = all([args.minutes, args.goals_assists, args.defense, args.fixtures])
    if use_components:
        def rd(p):
            pth = Path(p)
            if pth.suffix.lower() in {'.csv', '.txt'}:
                return pd.read_csv(pth)
            else:
                return pd.read_parquet(pth)
        minutes_df  = rd(args.minutes)
        ga_df       = rd(args.goals_assists)
        def_df      = rd(args.defense)
        saves_df    = rd(args.saves) if args.saves else pd.DataFrame(columns=['season','gw','player_id','pred_saves_mean'])
        fixtures_df = rd(args.fixtures)

        build_from_components(
            season=season, gws=gws, roster=roster,
            minutes_df=minutes_df, ga_df=ga_df, def_df=def_df, saves_df=saves_df,
            fixtures_df=fixtures_df, out_dir=out_dir, out_format=out_format
        )
    else:
        if not args.optimizer_input:
            raise SystemExit("Provide either component tables (--minutes, --goals-assists, --defense, --fixtures) or --optimizer-input.")
        opt_df = pd.read_parquet(args.optimizer_input)
        build_from_optimizer_input(
            season=season, gws=gws, roster=roster, opt_df=opt_df, out_dir=out_dir, out_format=out_format
        )

# ------------------------------ Main -----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build sim_input Parquets/CSVs for MC from roster + forecasts")
    # Single or batch
    ap.add_argument("--team-state", help="Path to team_state JSON (single run)")
    ap.add_argument("--team-state-glob", help="Glob of team_state JSONs for batch mode, e.g. data/state/candidates/*/team_state.json")
    ap.add_argument("--candidate-subdir", default="", help="Optional subdir appended inside each candidate folder, e.g., 'hold' or 'one_ft'")

    ap.add_argument("--season", required=True, help="Season string, e.g., 2025-2026")
    ap.add_argument("--gws", required=True, help="Comma-separated GWs, e.g., 4,5,6")
    ap.add_argument("--out-dir", required=True, help="Output ROOT (batch) or directory (single) for sim_input files")
    ap.add_argument("--out-format", choices=["parquet","csv"], default="parquet", help="Output file format (default: parquet)")

    # Component tables (shared across candidates)
    ap.add_argument("--minutes", help="CSV/Parquet for minutes forecast")
    ap.add_argument("--goals-assists", help="CSV/Parquet for goals & assists forecast")
    ap.add_argument("--defense", help="CSV/Parquet for defense/team λ forecasts")
    ap.add_argument("--saves", help="CSV/Parquet for GK saves forecast")
    ap.add_argument("--fixtures", help="CSV/Parquet for fixtures meta")

    # Fallback
    ap.add_argument("--optimizer-input", help="Parquet with rich fields (see module docstring)")

    args = ap.parse_args()

    if not args.team_state and not args.team_state_glob:
        raise SystemExit("Provide either --team-state (single) or --team-state-glob (batch).")

    season = args.season
    gws = parse_gws(args.gws)
    out_root = Path(args.out_dir)
    out_format = args.out_format

    if args.team_state_glob:
        # Batch mode
        paths = sorted(glob.glob(args.team_state_glob))
        if not paths:
            raise SystemExit(f"No team_state files matched glob: {args.team_state_glob}")
        print(f"Batch: found {len(paths)} candidates from glob.")

        for ts in paths:
            ts_path = Path(ts)
            cand_label = ts_path.parent.name or ts_path.stem
            cand_out = out_root / cand_label
            if args.candidate_subdir:
                cand_out = cand_out / args.candidate_subdir
            print(f"[{cand_label}] team_state={ts_path} -> out_dir={cand_out}")
            run_single_candidate(
                team_state_path=ts_path,
                season=season,
                gws=gws,
                out_dir=cand_out,
                out_format=out_format,
                args=args
            )
    else:
        # Single-candidate mode (backward compatible)
        run_single_candidate(
            team_state_path=Path(args.team_state),
            season=season,
            gws=gws,
            out_dir=out_root,
            out_format=out_format,
            args=args
        )

if __name__ == "__main__":
    main()
