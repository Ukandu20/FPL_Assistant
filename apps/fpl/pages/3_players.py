#import all libraries

from altair import Orientation
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

import plotly as py
import plotly.express as px
import plotly.graph_objects as go

import json





import streamlit as st

#initialize and setup the streamlit app
st.set_page_config(
    page_title= "Player Analysis & Deep Dive",
    layout="wide",
    )
#st.title("Player Analysis & Deep Dive")

#import the data
pd.set_option("display.max_columns", None)

#match data

defending = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_match/defense.csv")
passing = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_match/passing.csv")
misc =  pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_match/misc.csv")
keepers= pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_match/keepers.csv")
pass_type = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_match/passing_types.csv")
possession = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_match/possession.csv")
summary = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_match/summary.csv")
schedule = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/team_match/schedule.csv")
with open("data/processed/registry/prices/2025-2026.json", "r", encoding="utf-8") as f:
    prices = json.load(f)


#season data
szn_defending = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_season/defense.csv")
szn_passing = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_season/passing.csv")
szn_misc = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_season/misc.csv")
szn_gca = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_season/goal_shot_creation.csv")
szn_shooting = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_season/shooting.csv")
szn_possession = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_season/possession.csv")
szn_pass_type = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_season/passing_types.csv")
szn_standards = pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_season/standard.csv")
szn_keepers= pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_season/keeper.csv")
szn_keeper_adv= pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_season/keeper_adv.csv")
szn_playing_time= pd.read_csv("data/processed/fbref/ENG-Premier League/2025-2026/player_season/playing_time.csv")

#fpl data
fpl = pd.read_csv("data/processed/fpl/2025-2026/gws/merged_gws.csv")

#Create dataframes
defending_df = pd.DataFrame(defending)
passing_df = pd.DataFrame(passing)
misc_df = pd.DataFrame(misc)
keepers_df = pd.DataFrame(keepers)
pass_type_df = pd.DataFrame(pass_type)
possession_df = pd.DataFrame(possession)
summary_df = pd.DataFrame(summary)
schedule_df = pd.DataFrame(schedule)
szn_defending_df = pd.DataFrame(szn_defending)
szn_passing_df = pd.DataFrame(szn_passing)
szn_misc_df = pd.DataFrame(szn_misc)
szn_gca_df = pd.DataFrame(szn_gca)
szn_shooting_df = pd.DataFrame(szn_shooting)
szn_possession_df = pd.DataFrame(szn_possession)
szn_pass_type_df = pd.DataFrame(szn_pass_type)
szn_standards_df = pd.DataFrame(szn_standards)
szn_keepers_df = pd.DataFrame(szn_keepers)
szn_keeper_adv_df = pd.DataFrame(szn_keeper_adv)
szn_playing_time_df = pd.DataFrame(szn_playing_time)
fpl_df = pd.DataFrame(fpl)

# Rename columns for clarity
defending_df.rename(columns={'tkl_tkl':'tkl_pct', 'def_3rd': 'tkl_def_3rd', 'mid_3rd': 'tkl_mid_3rd',  'att_3rd': 'tkl_att_3rd', 'att':'drb_chl', 'sh': 'sh_blk'}, inplace=True)
passing_df.rename(columns={'cmp_cmp':'cmp_pct', 'att':'pass_att','1_3':'1_3pass','totdist':'totpassdist', 'prgdist':'prgpassdist'}, inplace=True)
misc_df.rename(columns={'won_won':'arlw_pct', 'won': 'arlw', 'lost':'arl_lost', 'recov':'recoveries'}, inplace=True)
pass_type_df.rename(columns={'in':'ck_in','out': 'ck_out','str': 'ck_str','off': 'pass_offside','blocks': 'pass_blocked', 'live': 'pass_live'}, inplace=True)
possession_df.rename(columns={'live':'touch_live','succ_succ':'succ_pct', 'def_pen': 'touch_def_pen', 'def_3rd': 'touch_def_3rd', 'mid_3rd': 'touch_mid_3rd',  'att_3rd': 'touch_att_3rd', 'att_pen': 'touch_att_pen', 'tkld': 'to_tkld', 'tkld_tkld':'to_tkld_pct', '1_3':'1_3carry', 'att': 'to_att', 'succ': 'to_succ','totdist':'totcarrydist', 'prgdist':'prgcarrydist'}, inplace=True)


# Merge match data
merged = defending_df.merge(
    passing_df[['player', 'player_id', 'game_id', 'cmp', 'pass_att', 'cmp_pct', 'totpassdist', 'prgpassdist',
    'short_cmp', 'att_att', 'medium_cmp', 'medium_att', 'long_cmp',
    'long_att', 'ast', 'xag', 'xa', 'kp', '1_3pass', 'ppa', 'crspa', 'prgp']], on=["player", 'player_id', 'game_id'], how="left",
    validate="many_to_one"  # IMPORTANT: fails fast if you accidentally have duplicates
)

merged = merged.merge(
    misc_df[['player', 'player_id', 'game_id','crdy', 'crdr', '2crdy', 'fls', 'fld', 'off',
        'pkwon', 'pkcon', 'og', 'arlw', 'arl_lost', 'recoveries',
        'arlw_pct']], on=["player", 'player_id', 'game_id'], how="left", validate="many_to_one"
)

merged = merged.merge(
    summary_df[['player', 'player_id', 'game_id','gls', 'pk', 'pkatt', 'sh', 'sot',
    'xg', 'npxg']], on=["player", 'player_id', 'game_id'], how="left", validate="many_to_one"
)

merged = merged.merge(
    possession_df[['player', 'player_id', 'game_id','touches', 'touch_def_pen', 'touch_def_3rd', 'touch_mid_3rd',
    'touch_att_3rd', 'touch_att_pen', 'touch_live', 'to_att', 'to_succ', 'succ_pct', 'to_tkld',
    'to_tkld_pct', 'carries', 'totcarrydist', 'prgcarrydist', 'prgc', '1_3carry', 'cpa',
    'mis', 'dis', 'rec', 'prgr']], on=["player", 'player_id', 'game_id'], how="left", validate="many_to_one"
)

merged = merged.merge(
    pass_type_df[['player', 'player_id', 'game_id','pass_live', 'dead', 'fk', 'tb', 'sw', 'crs',
    'ti', 'ck', 'ck_in', 'ck_out', 'ck_str', 'pass_offside', 'pass_blocked']], on=["player", 'player_id', 'game_id'], how="left", validate="many_to_one"
)

#Add GW info
sched = schedule_df.copy()
sched["gw"] = (
    sched["round"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .astype("Int64")
)

#add cleansheet info (robust: handle alternative column names or score parsing)
if "ga" in sched.columns:
    sched["cs"] = sched["ga"] == 0
else:
    alt_cols = ["goals_against", "goals_allowed", "GA", "away_goals", "a"]
    found = next((c for c in alt_cols if c in sched.columns), None)
    if found:
        sched["cs"] = sched[found] == 0
    else:
        # try to parse a score column like "score" or "result" with format "home-away"
        score_col = next((c for c in ["score", "result", "final_score", "scoreline"] if c in sched.columns), None)
        if score_col:
            def _opp_goals(row):
                try:
                    parts = str(row[score_col]).split("-")
                    if len(parts) != 2:
                        return np.nan
                    home_goals = int(parts[0])
                    away_goals = int(parts[1])
                    is_home = row.get("is_home", True)
                    return away_goals if is_home else home_goals
                except Exception:
                    return np.nan
            sched["__opp_ga__"] = sched.apply(_opp_goals, axis=1)
            sched["cs"] = sched["__opp_ga__"] == 0
            sched.drop(columns="__opp_ga__", inplace=True)
        else:
            # fallback: mark as False if we cannot determine goals against
            sched["cs"] = False

gw_map = (
    sched[["game_id", "gw", "cs"]]
    .drop_duplicates(subset=["game_id"], keep="first")
)

merged = merged.merge(gw_map, on="game_id", how="left", validate="many_to_one")


merged.fillna(0, inplace=True)
merged['defcon'] = (merged['tkl_int'] + merged['blocks'] + merged['clr']) 


def _normalize_prices(prices_raw: dict) -> dict[str, dict[int, float]]:
    out = {}
    for pid, gw_map in (prices_raw or {}).items():
        if not isinstance(gw_map, dict):
            continue
        clean = {}
        for gw, p in gw_map.items():
            try:
                clean[int(gw)] = float(p)
            except Exception:
                continue
        if clean:
            out[str(pid)] = clean
    return out

def _season_latest_gw(prices_map: dict[str, dict[int, float]]) -> int | None:
    all_gws = [gw for m in prices_map.values() for gw in m.keys()]
    return max(all_gws) if all_gws else None

def _current_price(prices_map: dict[str, dict[int, float]], player_id: str, season_latest_gw: int) -> float | None:
    m = prices_map.get(str(player_id))
    if not m:
        return None
    # rule: price at global latest gw else player latest
    return m.get(season_latest_gw, m[max(m.keys())])



#merge season data
szn_defending_df.rename(columns={'tkl_tkl':'tkl_pct', 'def_3rd': 'tkl_def_3rd', 'mid_3rd': 'tkl_mid_3rd',  'att_3rd': 'tkl_att_3rd', 'att':'drb_chl', 'sh': 'sh_blk', 'pass': 'pass_blk', 'lost': 'chall_lost'}, inplace=True)
szn_passing_df.rename(columns={'cmp_cmp':'cmp_pct', 'att':'pass_att','1_3':'1_3pass','totdist':'totpassdist', 'prgdist':'prgpassdist','cmp':'pass_cmp'}, inplace=True)
szn_misc_df.rename(columns={'won_won':'arlw_pct', 'won': 'arlw', 'lost':'arl_lost'}, inplace=True)
szn_pass_type_df.rename(columns={'live': 'pass_live','in':'ck_in','out': 'ck_out','str': 'ck_str','off': 'pass_offside','blocks': 'pass_blocked', 'fk': 'fk_pass'}, inplace=True)
szn_gca_df.rename(columns={'passlive': 'sca_passlive', 'passdead': 'sca_passdead', 'to': 'sca_to', 'sh': 'sca_sh', 'fld': 'sca_fld', 'def':'sca_def', 'passlive_passlive': 'gca_passlive', 'passdead_passdead': 'gca_passdead', 'to_to': 'gca_to', 'sh_sh': 'gca_sh', 'fld_fld': 'gca_fld', 'def_def': 'gca_def'}, inplace=True)
szn_possession_df.rename(columns={'live': 'touch_live','succ_succ':'succ_pct', 'def_pen': 'touch_def_pen', 'def_3rd': 'touch_def_3rd', 'mid_3rd': 'touch_mid_3rd',  'att_3rd': 'touch_att_3rd', 'att_pen': 'touch_att_pen', 'tkld': 'to_tkld', 'tkld_tkld':'to_tkld_pct', '1_3':'1_3carry', 'att': 'to_att', 'succ': 'to_succ','totdist':'totcarrydist', 'prgdist':'prgcarrydist'}, inplace=True)
szn_shooting_df.rename(columns={'sot_sot':'sot_pct', 'fk': 'fk_shot'}, inplace=True)
szn_standards_df.rename(columns={'gls_gls':'gls90', 'ast_ast':'ast90', 'g_a_g_a':'g_a90', 'g_pk_g_pk':'g_pk90', 'g_a_pk':'g_a_pk90', 'xg_xg':'xg90', 'xag_xag':'xag90', 'xg_xag':'xg_xag90', 'npxg_npxg':'npxg90', 'npxg_xag_npxg_xag':'npxg_xag90'}, inplace=True)
szn_playing_time_df.rename(columns={'min_min':'min_pct'}, inplace=True)

# Build deduplicated right-side subsets to guarantee unique keys (player_id)
passing_subset = szn_passing_df[['player', 'player_id', 'pass_cmp', 'pass_att', 'cmp_pct', 'totpassdist', 'prgpassdist',
        'short_cmp', 'att_att', 'medium_cmp', 'medium_att', 'long_cmp',
        'long_att', 'xag', 'xa', 'kp', '1_3pass', 'ppa', 'crspa', 'prgp']].drop_duplicates(subset=['player_id'])

misc_subset = szn_misc_df[['player', 'player_id','crdy', 'crdr', '2crdy', 'fls', 'fld', 'off',
        'pkwon', 'pkcon', 'og', 'arlw', 'arl_lost',
        'arlw_pct']].drop_duplicates(subset=['player_id'])

possession_subset = szn_possession_df[['player', 'player_id','touches', 'touch_def_pen', 'touch_def_3rd', 'touch_mid_3rd',
        'touch_att_3rd', 'touch_att_pen', 'touch_live', 'to_att', 'to_succ', 'succ_pct', 'to_tkld',
        'to_tkld_pct', 'carries', 'totcarrydist', 'prgcarrydist', 'prgc', '1_3carry', 'cpa',
        'mis', 'dis', 'rec', 'prgr']].drop_duplicates(subset=['player_id'])

pass_type_subset = szn_pass_type_df[['player', 'player_id', 'pass_live', 'dead', 'fk_pass', 'tb', 'sw', 'crs',
        'ti', 'ck', 'ck_in', 'ck_out', 'ck_str', 'pass_offside', 'pass_blocked']].drop_duplicates(subset=['player_id'])

gca_subset = szn_gca_df[['player', 'player_id', 'sca', 'sca90','sca_passlive','sca_passdead','sca_to','gca', 'gca90','sca_sh', 'sca_fld','sca_def',
        'gca_passlive','gca_passdead','gca_to','gca_sh','gca_fld','gca_def']].drop_duplicates(subset=['player_id'])

shooting_subset = szn_shooting_df[['player', 'player_id', 'gls', 'sh', 'sot', 'sot_pct', 'sh_90', 'sot_90', 'g_sh',
    'g_sot', 'dist', 'fk_shot', 'pk', 'pkatt', 'xg', 'npxg', 'npxg_sh', 'g_xg',
    'np_g_xg']].drop_duplicates(subset=['player_id'])

standards_subset = szn_standards_df[['player', 'player_id','g_a', 'g_pk','gls90', 'ast','ast90', 'g_a90', 'g_pk90', 'xg90', 'xag90', 'npxg90', 'npxg90', 'npxg_xag']].drop_duplicates(subset=['player_id'])

playtime_subset = szn_playing_time_df[['player', 'player_id', 'mp', 'min', 'mn_mp', 'min_pct', 'starts', 'mn_start', 'compl',
        'subs', 'mn_sub', 'unsub', 'ppm', 'ong', 'onga', 'onxg', 'onxga']].drop_duplicates(subset=['player_id'])

# Merge using deduplicated subsets; keeps validate to catch unexpected duplicates on left
szn_merged = szn_defending_df.merge(
    passing_subset, on=["player", 'player_id'], how="left", validate="many_to_one"
)

szn_merged = szn_merged.merge(
    misc_subset, on=["player", 'player_id'], how="left", validate="many_to_one"
)

szn_merged = szn_merged.merge(
    possession_subset, on=["player", 'player_id'], how="left", validate="many_to_one"
)

szn_merged = szn_merged.merge(
    pass_type_subset, on=["player", 'player_id'], how="left", validate="many_to_one"
)

szn_merged = szn_merged.merge(
    gca_subset, on=["player", 'player_id'], how="left", validate="many_to_one"
)

szn_merged = szn_merged.merge(
    shooting_subset, on=["player", 'player_id'], how="left", validate="many_to_one"
)

szn_merged = szn_merged.merge(
    standards_subset, on=["player", 'player_id'], how="left", validate="many_to_one"
)

szn_merged = szn_merged.merge(
    playtime_subset, on=["player", 'player_id'], how="left", validate="many_to_one"
)

szn_merged.fillna(0, inplace=True)
szn_merged['defcon'] = (szn_merged['tkl_int'] + szn_merged['blocks'] + szn_merged['clr']) / szn_merged['90s']


szn_playing_time_df.rename(columns={'min':'szn_mins'}, inplace=True)
szn_playing_time_df.fillna(0, inplace=True)

merged = merged.merge(
    szn_playing_time_df[['player', 'team', 'player_id', 'szn_mins']], on=['player', 'team','player_id'], how='left', validate="many_to_one"
)

missing = merged['szn_mins'].isna()
fallback_minutes = (
    merged.groupby("player_id")["min"]
    .sum()
)

# 3) fill only missing szn_mins using the fallback mapping
merged.loc[missing, "szn_mins"] = (
    merged.loc[missing, "player_id"]
        .map(fallback_minutes)
)


#Calculate total points a

collections = [
    # entity / labels (keep, don’t sum)
    "league", "season",
    "nation",
    "pos", "fpl_pos",
    "jersey_number", "age",
    "team_id", "opponent_id",
    "game", "game_id", "game_date",
    "round",
    "home", "away",
    "is_home", "is_away",
    "is_relegated"
]

meta_data = [
    # match context (filters/joins; don’t sum)
    "player", "position","player_id","team",
]

counts = [
    # minutes (denominator inputs)
    "min",

    # defense / disruptions
    "tkl", "tklw", "tkl_def_3rd", "tkl_mid_3rd", "tkl_att_3rd",
    "drb_chl", "challenges_tkl", "lost",
    "blocks", "sh_blk",
    "pass", "int", "tkl_int", "clr", "err",
    "defcon", 'recoveries', "cs",

    # passing creation + volume
    "cmp", "pass_att",
    "short_cmp", "att_att",
    "medium_cmp", "medium_att",
    "long_cmp", "long_att",
    "kp", "1_3pass", "ppa", "crspa", "prgp",
    "ast",

    # discipline / events
    "crdy", "crdr", "2crdy", "fls", "fld", "off",
    "pkwon", "pkcon", "og",
    "arlw", "arl_lost",

    # shooting + expected
    "gls", "pk", "pkatt", "sh", "sot",
    "xg", "npxg", "xa", "xag",

    # touches / carrying / receiving
    "touches",
    "touch_def_pen", "touch_def_3rd", "touch_mid_3rd",
    "touch_att_3rd", "touch_att_pen", "touch_live",
    "to_att", "to_succ", "to_tkld",
    "carries", "prgc", "1_3carry", "cpa",
    "mis", "dis", "rec", "prgr",

    # pass types
    "pass_live", "dead", "fk", "tb", "sw", "crs",
    "ti", "ck", "ck_in", "ck_out", "ck_str",
    "pass_offside", "pass_blocked",

    # distances (special: either sum-only or per90 optional)
    "totpassdist", "prgpassdist", "totcarrydist", "prgcarrydist"
]

others = [
    # rates / pct (don’t per90 or sum)
    "tkl_pct", "cmp_pct", "arlw_pct", "succ_pct", "to_tkld_pct",

    # season carryover (don’t per90)
    "szn_mins"
]

def totals_90s(df: pd.DataFrame, group_cols: list[str], count_cols: list[str]) -> pd.DataFrame:
    """
    Group by group_cols and sum count_cols.
    """
    # safety checks (fail fast, better error messages)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame, got {type(df)}")

    missing_g = [c for c in group_cols if c not in df.columns]
    missing_c = [c for c in count_cols if c not in df.columns]
    if missing_g:
        raise KeyError(f"Missing group columns: {missing_g}")
    if missing_c:
        raise KeyError(f"Missing count columns: {missing_c}")

    agg_map = {c: "sum" for c in count_cols}
    return df.groupby(group_cols, as_index=False).agg(agg_map)


#Create the sidebar with Controls & Filters

st.sidebar.markdown("### Filters & Controls")

#Positions Filter
position_map = {
    "Defenders": "DEF",
    "Midfielders": "MID",
    "Forwards": "FWD",
}

with st.sidebar.popover("Positions"):
    selected_labels = st.multiselect(
            "Positions",
            options=list(position_map.keys()),
            default=list(position_map.keys())
        )


    selected_positions = [position_map[label] for label in selected_labels]

    filtered_df = merged.copy()
    if selected_positions:  # only filter if user selected something
        filtered_df = filtered_df[filtered_df["position"].isin(selected_positions)]
    else:
        # If nothing selected, you can decide: show empty or fall back to all.
        # I recommend showing empty + a warning.
        st.sidebar.warning("Select at least one position.")
        filtered_df = filtered_df.iloc[0:0  
        ]  # empty dataframe with same columns    


#Matchweek Filter
# GW slider bounds
gw_series = merged["gw"].dropna()
if gw_series.empty:
    st.sidebar.warning("No gameweek data available.")
    filtered_df = filtered_df.iloc[0:0]
else:
    gw_min = int(gw_series.min())
    gw_max = int(gw_series.max())

    gw_start, gw_end = st.sidebar.slider(
        "Gameweek range",
        min_value=gw_min,
        max_value=gw_max,
        value=(gw_min, gw_max),
        step=1
    )

    filtered_df = filtered_df[
        (filtered_df["gw"] >= gw_start) &
        (filtered_df["gw"] <= gw_end)
    ]

#minutes Threshold Filter
min_minutes = st.sidebar.number_input(
    "Minimum Minutes Played",
    min_value=0,
    max_value=4000,
    value=500,
    step=50
)

filtered_df = filtered_df[filtered_df["szn_mins"] >= min_minutes]

#Price range filter
prices_map = _normalize_prices(prices)
latest_gw = _season_latest_gw(prices_map)

if latest_gw is None:
    st.sidebar.warning("Price data loaded, but no GW keys found.")
else:
    # Build a per-player price lookup once
    player_ids = filtered_df["player_id"].astype(str).unique()
    pid_to_price = {pid: _current_price(prices_map, pid, latest_gw) for pid in player_ids}

    # Determine slider bounds from players currently in filtered_df (after mins/GW/pos filters)
    avail_prices = [p for p in pid_to_price.values() if p is not None]

    if not avail_prices:
        st.sidebar.warning("No matching players have price data.")
        # Optional: if you prefer to drop all in this case:
        # filtered_df = filtered_df.iloc[0:0]
    else:
        pmin = round(float(min(avail_prices)), 1)
        pmax = round(float(max(avail_prices)), 1)

        lo, hi = st.sidebar.slider(
            "Price range (current price)",
            min_value=pmin,
            max_value=pmax,
            value=(pmin, pmax),
            step=0.1,
        )

        allowed_pids = {pid for pid, p in pid_to_price.items() if (p is not None and lo <= p <= hi)}
        filtered_df = filtered_df[filtered_df["player_id"].astype(str).isin(allowed_pids)]

#Team Filter
team_map = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton": "BHA",
    "Burnley": "BUR",
    "Chelsea": "CHE",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Liverpool": "LIV",
    "Leeds": "LEE",
    "Manchester City": "MCI",
    "Manchester United": "MUN",
    "Newcastle": "NEW",
    "Nottingham Forest": "NFO",
    "Sunderland": "SUN",
    "Tottenham": "TOT",
    "West Ham": "WHU",
    "Wolves": "WOL",
}

with st.sidebar.popover("Teams"):
    selected_labels = st.multiselect(
            "Teams",
            options=list(team_map.keys()),
            default=list(team_map.keys())
        )


    selected_teams = [team_map[label] for label in selected_labels]
    
    if selected_teams:  # only filter if user selected something
        filtered_df = filtered_df[filtered_df["team"].isin(selected_teams)]
    else:
        st.sidebar.warning("Select at least one team.")
        filtered_df = filtered_df.iloc[0:0] 



player_summary = totals_90s(
    df=filtered_df,          # or merged
    group_cols=meta_data,    # your groupby keys
    count_cols=counts        # summed stats
)

if player_summary.empty:
    st.warning("No matching players found. Adjust filters to see results.")
    st.stop()

player_summary["mins_90"] = (player_summary["min"] / 90).replace(0, np.nan)

PER90_EXCLUDE = {"min", "90s"}

per90_stats = [c for c in counts if c not in PER90_EXCLUDE]

for stat in per90_stats:
    player_summary[f"{stat}_per90"] = (
        player_summary[stat] / player_summary["mins_90"]
    )

# Optional: clean NaNs (players with 0 mins)
player_summary[[f"{c}_per90" for c in per90_stats]] = (
    player_summary[[f"{c}_per90" for c in per90_stats]].fillna(0)
)

st.sidebar.markdown("### Display Mode")

mode = st.sidebar.radio(
    "Show stats as:",
    options=["Totals", "Per 90"],
    index=0,
    horizontal=True
)

use_per90 = (mode == "Per 90")



def stat_col(stat: str) -> str:
    """Return the correct column name based on the UI mode."""
    if use_per90:
        return f"{stat}_per90"
    return stat





#Creating the charts and plots
#The KPI information at the top of the page

def kpi_card(title, value, delta=None, suffix="", positive_good=True):
    color = "#2ecc71" if (delta is not None and delta >= 0) == positive_good else "#e74c3c"

    st.markdown(
        f"""
        <div style="
            padding: 1rem;
            border-radius: 12px;
            background-color: #0e1117;
            border: 1px solid #2a2a2a;
            text-align: center;
            height:100px;
            margin-bottom:15px;
        ">
            <div style="font-size: 0.85rem; color: #9aa0a6;">{title}</div>
            <div style="display: flex; align-items: baseline; justify-content: center; gap: 0.5rem; margin-top: 0.5rem;">            
                <div style="font-size: 1rem; font-weight: 600; justify-content: left;">{value}</div>
                <div style="font-size: 0.8rem; color: #9aa0a6;">{suffix}</div>
            </div>
            {f'<div style="color:{color}; font-size:0.9rem;">{delta:+.2f}</div>' if delta is not None else ""}
        </div>
        """,
        unsafe_allow_html=True
    )


#main container
main = st.container(border=True)

with main:
    kpi = st.container(border=False)
    with kpi:
        kpi1, kpi2, kpi3, kpi4= st.columns(4, vertical_alignment="center")

        #highest goal scorer
        with kpi1:
                highest_goal_scorer = player_summary.loc[player_summary['gls'].idxmax()]
                kpi_card(
                    title="Top Goal Scorer",
                    value=f"{highest_goal_scorer['player']}",
                    suffix=f"{highest_goal_scorer['gls']}"
                )

        #most assists
        with kpi2:
                most_assists = player_summary.loc[player_summary['ast'].idxmax()]
                kpi_card(
                    title="Top Assist Provider",
                    value=f"{most_assists['player']}",
                    suffix=f"{most_assists['ast']}"
                )

        #highest DefCon
        with kpi3:
                col = stat_col("defcon")
                leader = player_summary.loc[player_summary[col].idxmax()]

                suffix = "/90" if use_per90 else ""
                kpi_card("Defcon leader", leader["player"], suffix=f"{leader[col]:.2f}")


        #Xg Overperformance & underperformance
        with kpi4:
                player_summary['xg_diff'] = player_summary['gls'] - player_summary['xg']
                xg_overperformer = player_summary.loc[player_summary['xg_diff'].idxmax()]
                xg_underperformer = player_summary.loc[player_summary['xg_diff'].idxmin()]
                kpi_card(
                    title="XG Overperformer",
                    value=f"{xg_overperformer['player']}",
                    suffix=f"{xg_overperformer['gls']:.2f}",
                    delta=xg_overperformer['xg_diff'],
                    positive_good=True
                )


    #Charting area
    row1 = st.columns(2, vertical_alignment="center", border=True)
    row2 = st.columns(2, vertical_alignment="center", border=True)


    with row1[0]:
        player_summary = player_summary.copy()

        player_summary['xg_diff'] = player_summary['gls'] - player_summary['xg']

        top_overperformers = player_summary.nlargest(5, 'xg_diff')
        top_underperformers = player_summary.nsmallest(5, 'xg_diff')

        top_overperformers = top_overperformers.sort_values("xg_diff", ascending=False)
        top_underperformers = top_underperformers.sort_values("xg_diff", ascending=False)

        order = list(top_overperformers["player"]) + list(top_underperformers["player"])

        fig_ast = go.Figure()
        fig_ast.add_trace(go.Bar(
            y=top_overperformers['player'],
            x=top_overperformers['xg_diff'],
            name='Overperformers',
            marker_color='Green',
            orientation='h',
            text=top_overperformers['gls']
        ))
        fig_ast.add_trace(go.Bar(
            y=top_underperformers['player'],
            x=top_underperformers['xg_diff'],
            name='Underperformers',
            marker_color='red',
            orientation='h',
            text=top_underperformers['gls']
            
        ))
        fig_ast.update_layout(
            height=350,        # 👈 critical
            margin=dict(t=40, b=30, l=20, r=20),
            title = dict (
                text='Top 5 xG Overperformers & Underperformers',
                x=0.3
                ),
            yaxis_title='Player',
            xaxis_title='xG Difference (Goals - xG)',
            barmode='group'
        )
        fig_ast.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_ast, use_container_width=True)


    #Defensive Contributions
    with row2[1]:
        df = player_summary.copy()

        tkl = stat_col("tkl_int")
        recov = stat_col("recoveries")

        px_fig = px.scatter(
            df,
            x=tkl,
            y=recov,
            color="team",
            hover_name="player"
        )

        fig = go.Figure()
        fig.add_traces(px_fig.data)        # ✅ adds the traces, not the figure
        fig.update_layout(
            px_fig.layout,
            height=350,        # 👈 critical
            margin=dict(t=40, b=30, l=20, r=20),
            title = dict (
                text='Top 5 xG Overperformers & Underperformers',
                x=0.3
                ),
            xaxis_title='Tackles & Interceptions',
            yaxis_title='Recoveries',)   # ✅ optional: copy layout too

        st.plotly_chart(fig, use_container_width=True)

