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
fixture_calendar = pd.read_csv("data/processed/registry/fixtures/2025-2026/fixture_calendar.csv")
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
fixture_calendar_df = pd.DataFrame(fixture_calendar)
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

# keeper stats (saves) for FPL scoring
keepers_subset = keepers_df[["player", "player_id", "game_id", "saves"]].drop_duplicates(subset=["player_id", "game_id"])
merged = merged.merge(
    keepers_subset, on=["player", "player_id", "game_id"], how="left", validate="many_to_one"
)

#Add GW info
sched = schedule_df.copy()
sched["gw"] = (
    sched["round"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .astype("Int64")
)

#add goals against + cleansheet info (robust: handle alternative column names or score parsing)
if "ga" in sched.columns:
    sched["ga"] = pd.to_numeric(sched["ga"], errors="coerce")
else:
    alt_cols = ["goals_against", "goals_allowed", "GA", "away_goals", "a"]
    found = next((c for c in alt_cols if c in sched.columns), None)
    if found:
        sched["ga"] = pd.to_numeric(sched[found], errors="coerce")
    else:
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

            sched["ga"] = sched.apply(_opp_goals, axis=1)
        else:
            sched["ga"] = np.nan

sched["cs"] = sched["ga"] == 0

gw_map = (
    sched[["game_id", "gw", "cs", "ga"]]
    .drop_duplicates(subset=["game_id"], keep="first")
)

merged = merged.merge(gw_map, on="game_id", how="left", validate="many_to_one")


merged.fillna(0, inplace=True)
merged['defcon'] = (merged['tkl_int'] + merged['blocks'] + merged['clr']) 

FPL_SCORING = {
    "minutes_1_59": {"GK": 1, "DEF": 1, "MID": 1, "FWD": 1},
    "minutes_60_plus": {"GK": 2, "DEF": 2, "MID": 2, "FWD": 2},
    "goal": {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4},
    "assist": {"GK": 3, "DEF": 3, "MID": 3, "FWD": 3},
    "clean_sheet": {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "goals_conceded_2": {"GK": -1, "DEF": -1, "MID": 0, "FWD": 0},
    "saves_3": {"GK": 1, "DEF": 0, "MID": 0, "FWD": 0},
    "penalty_save": {"GK": 5, "DEF": 0, "MID": 0, "FWD": 0},
    "penalty_miss": {"GK": -2, "DEF": -2, "MID": -2, "FWD": -2},
    "defensive_contribution": {"GK": 0, "DEF": 2, "MID": 2, "FWD": 2},
    "yellow_card": {"GK": -1, "DEF": -1, "MID": -1, "FWD": -1},
    "red_card": {"GK": -3, "DEF": -3, "MID": -3, "FWD": -3},
    "own_goal": {"GK": -2, "DEF": -2, "MID": -2, "FWD": -2},
}

pos = merged["position"].fillna("")
mins = merged["min"].fillna(0)
ga = pd.to_numeric(merged["ga"], errors="coerce").fillna(0)
saves = pd.to_numeric(merged["saves"], errors="coerce").fillna(0)
pkatt = pd.to_numeric(merged["pkatt"], errors="coerce").fillna(0)
pk = pd.to_numeric(merged["pk"], errors="coerce").fillna(0)

appearance_pts = np.select(
    [mins >= 60, mins >= 1],
    [2, 1],
    default=0
)

goal_pts = merged["gls"] * pos.map(FPL_SCORING["goal"]).fillna(0)
assist_pts = merged["ast"] * pos.map(FPL_SCORING["assist"]).fillna(0)

clean_sheet_pts = np.where(
    (mins >= 60) & (merged["cs"] == 1),
    pos.map(FPL_SCORING["clean_sheet"]).fillna(0),
    0
)

gc_twos = np.floor(ga / 2)
goals_conceded_pts = np.where(
    mins > 0,
    gc_twos * pos.map(FPL_SCORING["goals_conceded_2"]).fillna(0),
    0
)

saves_pts = np.floor(saves / 3) * pos.map(FPL_SCORING["saves_3"]).fillna(0)

penalty_miss_pts = (pkatt - pk).clip(lower=0) * FPL_SCORING["penalty_miss"]["MID"]
penalty_save_pts = 0

defcon_total = merged["tkl_int"] + merged["blocks"] + merged["clr"]
mid_fwd_total = defcon_total + merged["recoveries"]
defcon_pts = np.where(
    (pos == "DEF") & (defcon_total >= 10),
    FPL_SCORING["defensive_contribution"]["DEF"],
    np.where(
        pos.isin(["MID", "FWD"]) & (mid_fwd_total >= 12),
        FPL_SCORING["defensive_contribution"]["MID"],
        0
    )
)

discipline_pts = (
    merged["crdy"] * FPL_SCORING["yellow_card"]["MID"] +
    merged["crdr"] * FPL_SCORING["red_card"]["MID"] +
    merged["og"] * FPL_SCORING["own_goal"]["MID"]
)

merged["fpl_points"] = (
    appearance_pts
    + goal_pts
    + assist_pts
    + clean_sheet_pts
    + goals_conceded_pts
    + saves_pts
    + penalty_save_pts
    + penalty_miss_pts
    + defcon_pts
    + discipline_pts
)

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


def _build_fdr_map(fixtures_df: pd.DataFrame) -> dict[str, int]:
    if fixtures_df.empty:
        return {}
    played = fixtures_df[fixtures_df["status"] == "finished"].copy()
    played["xga"] = pd.to_numeric(played["xga"], errors="coerce")
    avg_xga = played.groupby("team")["xga"].mean()
    if avg_xga.empty:
        return {}
    rank = avg_xga.rank(pct=True, ascending=True)
    difficulty = (1 + (1 - rank) * 4).round().astype(int).clip(1, 5)
    return difficulty.to_dict()


def _fdr_color(fdr: float | int | None) -> str:
    if pd.isna(fdr):
        return ""
    colors = {
        1: "#2ecc71",
        2: "#7bd389",
        3: "#f1c40f",
        4: "#e67e22",
        5: "#e74c3c",
    }
    return colors.get(int(fdr), "")

def _render_fixtures_panel(summary_df: pd.DataFrame, fixtures_df: pd.DataFrame) -> None:
    fixtures = fixtures_df.copy()
    if fixtures.empty:
        st.info("No fixture data available.")
        return

    fixtures["date_sched"] = pd.to_datetime(fixtures["date_sched"], errors="coerce")
    fixtures["opponent"] = np.where(
        fixtures["is_home"].astype(int) == 1, fixtures["away"], fixtures["home"]
    )
    fdr_map = _build_fdr_map(fixtures)
    upcoming = fixtures[fixtures["status"] != "finished"].copy()
    upcoming["gw_orig"] = pd.to_numeric(upcoming["gw_orig"], errors="coerce")

    next_gws = (
        upcoming["gw_orig"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .head(3)
        .tolist()
    )
    if not next_gws:
        st.info("No upcoming fixtures found.")
        return

    teams = sorted(summary_df["team"].dropna().unique())
    gw_cols = [f"Gameweek {gw}" for gw in next_gws]
    fdr_cols = [f"__fdr_{i+1}" for i in range(len(next_gws))]
    rows = []
    for team in teams:
        team_upcoming = upcoming[upcoming["team"] == team]
        row = {"Team": team}
        for i, gw in enumerate(next_gws):
            gw_fixtures = team_upcoming[team_upcoming["gw_orig"] == gw]
            if gw_fixtures.empty:
                row[gw_cols[i]] = ""
                row[fdr_cols[i]] = np.nan
                continue
            opp_list = []
            fdr_list = []
            for _, r in gw_fixtures.iterrows():
                opp = r["opponent"]
                ha = "H" if int(r["is_home"]) == 1 else "A"
                opp_list.append(f"{opp} ({ha})")
                fdr_list.append(fdr_map.get(opp, np.nan))
            row[gw_cols[i]] = " / ".join(opp_list)
            row[fdr_cols[i]] = np.nanmean(fdr_list) if fdr_list else np.nan
        rows.append(row)

    fixtures_view = pd.DataFrame(rows)
    fixtures_display = fixtures_view.drop(columns=fdr_cols)
    fdr_lookup = dict(
        zip(
            fixtures_view.index,
            fixtures_view[fdr_cols].to_numpy().tolist()
        )
    )

    def _style_fdr_display(row):
        styles = {col: "" for col in row.index}
        fdr_vals = fdr_lookup.get(row.name, [])
        for col, fdr_val in zip(gw_cols, fdr_vals):
            color = _fdr_color(fdr_val)
            if color:
                styles[col] = f"background-color: {color}; color: #0e1117;"
        return [styles[col] for col in row.index]

    styled = fixtures_display.style.apply(_style_fdr_display, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=320)

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
    "fpl_points",

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

st.sidebar.markdown("#### Positions")
pos_cols = st.sidebar.columns(len(position_map))
pos_defaults = {label: True for label in position_map}
pos_selected = []
for col, label in zip(pos_cols, position_map.keys()):
    if col.checkbox(label, value=pos_defaults[label], key=f"pos_{position_map[label]}"):
        pos_selected.append(position_map[label])

filtered_df = merged.copy()
if pos_selected:  # only filter if user selected something
    filtered_df = filtered_df[filtered_df["position"].isin(pos_selected)]
else:
    st.sidebar.warning("Select at least one position.")
    filtered_df = filtered_df.iloc[0:0]  # empty dataframe with same columns


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

if gw_start == gw_min and gw_end == gw_max:
    gw_range_label = "This season"
elif gw_end == gw_max:
    gw_range_label = f"Since GW {gw_start}"
else:
    gw_range_label = f"From GW {gw_start} to GW {gw_end}"

#minutes Threshold Filter
gw_count = filtered_df["gw"].nunique() if "gw" in filtered_df.columns else 0
min_minutes = int(0.4 * gw_count * 90) if gw_count else 0
st.sidebar.markdown(f"Minimum Minutes Played (auto): {min_minutes}")

minutes_by_player = filtered_df.groupby("player_id")["min"].sum()
eligible_players = minutes_by_player[minutes_by_player >= min_minutes].index
filtered_df = filtered_df[filtered_df["player_id"].isin(eligible_players)]

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
        pmin = np.floor(float(min(avail_prices)) * 2) / 2
        pmax = np.ceil(float(max(avail_prices)) * 2) / 2

        lo, hi = st.sidebar.slider(
            "Price range (current price)",
            min_value=pmin,
            max_value=pmax,
            value=(pmin, pmax),
            step=0.5,
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



st.sidebar.markdown("### Display Mode")

if "display_mode" not in st.session_state:
    st.session_state["display_mode"] = "Per 90"

mode = st.sidebar.radio(
    "Show stats as:",
    options=["Totals", "Per 90"],
    index=1,
    key="display_mode",
    horizontal=True
)

use_per90 = (mode == "Per 90")


def stat_col(stat: str) -> str:
    """Return the correct column name based on the UI mode."""
    if use_per90:
        return f"{stat}_per90"
    return stat


def build_player_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = totals_90s(
        df=df,              # or merged
        group_cols=meta_data,
        count_cols=counts
    )
    if summary.empty:
        return summary

    summary["mins_90"] = (summary["min"] / 90).replace(0, np.nan)

    per90_stats = [c for c in counts if c not in {"min", "90s"}]
    for stat in per90_stats:
        summary[f"{stat}_per90"] = summary[stat] / summary["mins_90"]

    # Optional: clean NaNs (players with 0 mins)
    per90_cols = [f"{c}_per90" for c in per90_stats]
    summary[per90_cols] = (
        summary[per90_cols]
        .fillna(0)
        .infer_objects(copy=False)
    )
    return summary


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

def show_plot(fig, filename: str) -> None:
    config = {
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "scale": 2
        }
    }
    st.plotly_chart(fig, use_container_width=True, config=config)


def render_player_tab(df: pd.DataFrame, position_label: str = "All") -> None:
    summary = build_player_summary(df)
    if summary.empty:
        st.warning("No matching players found. Adjust filters to see results.")
        return

    #main container
    main = st.container(border=True)

    with main:
        kpi = st.container(border=False)
        with kpi:
            def _kpi_leader(title: str, col: str, suffix_fmt: str = "{:.2f}", add_per90: bool = False) -> None:
                leader = summary.loc[summary[col].idxmax()]
                suffix = suffix_fmt.format(leader[col])
                if add_per90:
                    suffix = f"{suffix}/90"
                kpi_card(title, leader["player"], suffix=suffix)

            if position_label in {"FWD", "MID", "DEF"}:
                kpi1, kpi2, kpi3, kpi4 = st.columns(4, vertical_alignment="center")

                if position_label == "FWD":
                    with kpi1:
                        _kpi_leader("Top Scorer", "gls", "{:.2f}")
                    with kpi2:
                        _kpi_leader("Top Assist Provider", "ast", "{:.0f}")
                    with kpi3:
                        _kpi_leader("xG/90 Leader", "xg_per90", "{:.2f}")
                    with kpi4:
                        _kpi_leader(
                            "Points Return",
                            stat_col("fpl_points"),
                            "{:.2f}",
                            add_per90=use_per90
                        )
                elif position_label == "MID":
                    with kpi1:
                        _kpi_leader("Top Scorer", "gls", "{:.2f}")
                    with kpi2:
                        _kpi_leader("Top Assist Provider", "ast", "{:.0f}")
                    with kpi3:
                        _kpi_leader("xA/90 Leader", "xa_per90", "{:.2f}")
                    with kpi4:
                        _kpi_leader(
                            "Points Return",
                            stat_col("fpl_points"),
                            "{:.2f}",
                            add_per90=use_per90
                        )
                else:
                    g_a_col = "g_a_per90" if use_per90 else "_g_a"
                    summary[g_a_col] = summary[stat_col("gls")] + summary[stat_col("ast")]
                    with kpi1:
                        _kpi_leader("Tackles+Ints Leader", stat_col("tkl_int"), "{:.2f}", add_per90=use_per90)
                    with kpi2:
                        _kpi_leader("Defcon Leader", stat_col("defcon"), "{:.2f}", add_per90=use_per90)
                    with kpi3:
                        _kpi_leader("G+A Leader", g_a_col, "{:.2f}", add_per90=use_per90)
                    with kpi4:
                        _kpi_leader(
                            "Points Return",
                            stat_col("fpl_points"),
                            "{:.2f}",
                            add_per90=use_per90
                        )
            else:
                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5, vertical_alignment="center")

                #highest goal scorer
                with kpi1:
                        highest_goal_scorer = summary.loc[summary['gls'].idxmax()]
                        kpi_card(
                            title="Top Goal Scorer",
                            value=f"{highest_goal_scorer['player']}",
                        suffix=f"{int(highest_goal_scorer['gls'])}"
                        )

                #most assists
                with kpi2:
                        most_assists = summary.loc[summary['ast'].idxmax()]
                        kpi_card(
                            title="Top Assist Provider",
                            value=f"{most_assists['player']}",
                        suffix=f"{int(most_assists['ast'])}"
                        )

                #highest DefCon
                with kpi3:
                        col = stat_col("defcon")
                        leader = summary.loc[summary[col].idxmax()]

                        suffix = "/90" if use_per90 else ""
                        kpi_card("Defcon leader", leader["player"], suffix=f"{leader[col]:.2f}{suffix}")


                #Xg Overperformance & underperformance
                with kpi4:
                        kpi_df = summary.copy()
                        kpi_df['xg_diff'] = kpi_df['gls'] - kpi_df['xg']
                        xg_overperformer = kpi_df.loc[kpi_df['xg_diff'].idxmax()]
                        xg_underperformer = kpi_df.loc[kpi_df['xg_diff'].idxmin()]
                        kpi_card(
                            title="XG Overperformer",
                            value=f"{xg_overperformer['player']}",
                            suffix=f"{xg_overperformer['gls']:.2f}",
                            delta=xg_overperformer['xg_diff'],
                            positive_good=True
                        )

                #Points return
                with kpi5:
                        col = stat_col("fpl_points")
                        leader = summary.loc[summary[col].idxmax()]
                        suffix = "/90" if use_per90 else ""
                        kpi_card("Points return", leader["player"], suffix=f"{leader[col]:.2f}{suffix}")


        #Charting area
        row1 = st.columns(2, vertical_alignment="center", border=True)
        row2 = st.columns(3, vertical_alignment="center", border=True)

        with row1[0]:
            points_col = stat_col("fpl_points")
            price_df = summary.copy()

            if latest_gw is None or not prices_map:
                st.info("Price data not available for points vs price chart.")
            else:
                price_df["price_current"] = price_df["player_id"].astype(str).map(
                    lambda pid: _current_price(prices_map, pid, latest_gw)
                )
                price_df = price_df[price_df["price_current"].notna()]

                if price_df.empty:
                    st.info("No matching players have price data for this view.")
                else:
                    gw_count = df["gw"].nunique() if "gw" in df.columns else None
                    if gw_count:
                        min_90s = 0.4 * gw_count
                        price_df = price_df[price_df["mins_90"] >= min_90s]
                        if price_df.empty:
                            st.info("No players meet the minutes threshold for this GW range.")
                            return
                    q1, q2, q3 = price_df[points_col].quantile([0.25, 0.5, 0.75])
                    price_df["points_band"] = pd.cut(
                        price_df[points_col],
                        bins=[-np.inf, q1, q2, q3, np.inf],
                        labels=["Bottom 25%", "25-50%", "50-75%", "Top 25%"]
                    )
                    band_colors = {
                        "Bottom 25%": "#d9534f",
                        "25-50%": "#f0ad4e",
                        "50-75%": "#5bc0de",
                        "Top 25%": "#5cb85c",
                    }
                    point_colors = {
                        "Bottom 25%": "#6a3d9a",
                        "25-50%": "#1f78b4",
                        "50-75%": "#33a02c",
                        "Top 25%": "#ff7f00",
                    }
                    px_fig = px.scatter(
                        price_df,
                        x="price_current",
                        y=points_col,
                        color="points_band",
                        hover_name="player",
                        color_discrete_map=point_colors,
                        size="mins_90",
                        size_max=int(price_df["mins_90"].max()) if "mins_90" in price_df.columns else 18
                    )
                    fig = go.Figure()
                    fig.add_traces(px_fig.data)
                    x_min = price_df["price_current"].min()
                    x_max = price_df["price_current"].max()
                    y_min = price_df[points_col].min()
                    y_max = price_df[points_col].max()
                    x_mean = price_df["price_current"].mean()
                    y_mean = price_df[points_col].mean()
                    fig.add_hrect(y0=y_min, y1=q1, fillcolor=band_colors["Bottom 25%"], opacity=0.10, line_width=0)
                    fig.add_hrect(y0=q1, y1=q2, fillcolor=band_colors["25-50%"], opacity=0.10, line_width=0)
                    fig.add_hrect(y0=q2, y1=q3, fillcolor=band_colors["50-75%"], opacity=0.10, line_width=0)
                    fig.add_hrect(y0=q3, y1=y_max, fillcolor=band_colors["Top 25%"], opacity=0.10, line_width=0)
                    fig.add_shape(
                        type="line",
                        x0=x_mean, x1=x_mean,
                        y0=price_df[points_col].min(), y1=price_df[points_col].max(),
                        line=dict(color="#9aa0a6", width=1, dash="dash")
                    )
                    fig.add_shape(
                        type="line",
                        x0=price_df["price_current"].min(), x1=price_df["price_current"].max(),
                        y0=y_mean, y1=y_mean,
                        line=dict(color="#9aa0a6", width=1, dash="dash")
                    )
                    fig.update_layout(
                        px_fig.layout,
                        height=350,
                        margin=dict(t=40, b=30, l=20, r=20),
                        title=dict(
                            text=f"Points Return vs Price ({position_label}) - {gw_range_label}",
                            x=0.5,
                            xanchor="center"
                        ),
                        xaxis_title="Current Price",
                        yaxis_title="Points Return" + ("/90" if use_per90 else ""),
                        legend_title_text="Points Quartile"
                    )
                    # Quadrant labels
                    fig.add_annotation(
                        x=(x_mean + x_max) / 2, y=(y_mean + y_max) / 2, xref="x", yref="y",
                        text="Premium (High Price/High Return)",
                        showarrow=False, xanchor="center", yanchor="middle",
                        font=dict(size=10, color="#9aa0a6")
                    )
                    fig.add_annotation(
                        x=(x_min + x_mean) / 2, y=(y_mean + y_max) / 2, xref="x", yref="y",
                        text="Cheap & Good (Low Price/High Return)",
                        showarrow=False, xanchor="center", yanchor="middle",
                        font=dict(size=10, color="#9aa0a6")
                    )
                    fig.add_annotation(
                        x=(x_mean + x_max) / 2, y=(y_min + y_mean) / 2, xref="x", yref="y",
                        text="Expensive/Not Worth",
                        showarrow=False, xanchor="center", yanchor="middle",
                        font=dict(size=10, color="#9aa0a6")
                    )
                    fig.add_annotation(
                        x=(x_min + x_mean) / 2, y=(y_min + y_mean) / 2, xref="x", yref="y",
                        text="Cheap/Low Return",
                        showarrow=False, xanchor="center", yanchor="middle",
                        font=dict(size=10, color="#9aa0a6")
                    )
                    # Standout player labels: top return and best value
                    price_df["value_ratio"] = price_df[points_col] / price_df["price_current"]
                    standout = pd.concat(
                        [
                            price_df.nlargest(5, points_col),
                            price_df.nlargest(5, "value_ratio"),
                        ]
                    ).drop_duplicates(subset=["player_id"])
                    for _, row in standout.iterrows():
                        fig.add_annotation(
                            x=row["price_current"],
                            y=row[points_col],
                            text=row["player"],
                            showarrow=False,
                            xanchor="left",
                            yanchor="bottom",
                            font=dict(size=10, color="#e6e6e6"),
                        )
                    show_plot(fig, "points_return_vs_price")


        if position_label == "FWD":
            with row1[1]:
                # Render MID-specific chart when MID is selected
                if position_label == "MID":
                    if "game_id" in df.columns:
                        mins_by_player = df.groupby("player_id")["min"].sum()
                        matches_by_player = df.groupby("player_id")["game_id"].nunique()
                        mins_per_match = (mins_by_player / matches_by_player.replace(0, np.nan))

                        summary_mid = summary.copy()
                        summary_mid["mins_per_match"] = summary_mid["player_id"].map(mins_per_match)
                        summary_mid = summary_mid.dropna(subset=["mins_per_match", "fpl_points_per90"])

                        q1, q2, q3 = summary_mid["fpl_points_per90"].quantile([0.25, 0.5, 0.75])
                        summary_mid["points_band"] = pd.cut(
                            summary_mid["fpl_points_per90"],
                            bins=[-np.inf, q1, q2, q3, np.inf],
                            labels=["Bottom 25%", "25-50%", "50-75%", "Top 25%"]
                        )
                        band_colors = {
                            "Bottom 25%": "#d9534f",
                            "25-50%": "#f0ad4e",
                            "50-75%": "#5bc0de",
                            "Top 25%": "#5cb85c",
                        }
                        point_colors = {
                            "Bottom 25%": "#6a3d9a",
                            "25-50%": "#1f78b4",
                            "50-75%": "#33a02c",
                            "Top 25%": "#ff7f00",
                        }
                        px_fig = px.scatter(
                            summary_mid,
                            x="mins_per_match",
                            y="fpl_points_per90",
                            color="points_band",
                            hover_name="player",
                            color_discrete_map=point_colors,
                            size="min",
                            size_max=18
                        )
                        fig = go.Figure()
                        fig.add_traces(px_fig.data)
                        x_min = summary_mid["mins_per_match"].min()
                        x_max = summary_mid["mins_per_match"].max()
                        y_min = summary_mid["fpl_points_per90"].min()
                        y_max = summary_mid["fpl_points_per90"].max()
                        x_mean = summary_mid["mins_per_match"].mean()
                        y_mean = summary_mid["fpl_points_per90"].mean()
                        fig.add_hrect(y0=y_min, y1=q1, fillcolor=band_colors["Bottom 25%"], opacity=0.10, line_width=0)
                        fig.add_hrect(y0=q1, y1=q2, fillcolor=band_colors["25-50%"], opacity=0.10, line_width=0)
                        fig.add_hrect(y0=q2, y1=q3, fillcolor=band_colors["50-75%"], opacity=0.10, line_width=0)
                        fig.add_hrect(y0=q3, y1=y_max, fillcolor=band_colors["Top 25%"], opacity=0.10, line_width=0)
                        fig.add_shape(
                            type="line",
                            x0=x_mean, x1=x_mean,
                            y0=y_min, y1=y_max,
                            line=dict(color="#9aa0a6", width=1, dash="dash")
                        )
                        fig.add_shape(
                            type="line",
                            x0=x_min, x1=x_max,
                            y0=y_mean, y1=y_mean,
                            line=dict(color="#9aa0a6", width=1, dash="dash")
                        )
                        fig.update_layout(
                            px_fig.layout,
                            height=350,
                            margin=dict(t=40, b=30, l=20, r=20),
                        title=dict(text=f"Points per 90 vs Minutes per Match - {gw_range_label}", x=0.5, xanchor="center"),
                            xaxis_title="Minutes per Match",
                            yaxis_title="Points per 90",
                            legend_title_text="Points Quartile"
                        )
                        show_plot(fig, "points_per90_vs_minutes_per_match")
                    else:
                        st.info("Missing match data for MID chart.")
                # Render FWD-specific chart when FWD is selected
                elif position_label == "FWD":
                    chart_df = summary.copy()
                    gls_col = stat_col("gls")
                    xg_col = stat_col("xg")
                    diff_label = "Goals - xG" + ("/90" if use_per90 else "")

                    chart_df["xg_diff"] = chart_df[gls_col] - chart_df[xg_col]

                    top_overperformers = chart_df.nlargest(5, "xg_diff")
                    top_underperformers = chart_df.nsmallest(5, "xg_diff")

                    top_overperformers = top_overperformers.sort_values("xg_diff", ascending=False)
                    top_underperformers = top_underperformers.sort_values("xg_diff", ascending=False)

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
                        height=350,
                        margin=dict(t=40, b=30, l=20, r=20),
                        title=dict(
                    text=f"Top 5 xG Overperformers & Underperformers - {gw_range_label}",
                    x=0.5,
                    xanchor="center"
                    ),
                        yaxis_title='Player',
                        xaxis_title=f"xG Difference ({diff_label})",
                        barmode='group'
                    )
                    fig_ast.update_yaxes(autorange='reversed')
                    show_plot(fig_ast, "xg_over_under")
                else:
                    st.info("No specific chart for this position.")
                
            with row2[0]:
                if "xg_per90" in summary.columns and "sh_per90" in summary.columns:
                    px_fig = px.scatter(
                        summary,
                        x="sh_per90",
                        y="xg_per90",
                        color="team",
                        hover_name="player"
                    )
                    fig = go.Figure()
                    fig.add_traces(px_fig.data)
                    fig.update_layout(
                        px_fig.layout,
                        height=350,
                        margin=dict(t=40, b=30, l=20, r=20),
                        title=dict(text=f"xG per 90 vs Shots per 90 - {gw_range_label}", x=0.5, xanchor="center"),
                        xaxis_title="Shots per 90",
                        yaxis_title="xG per 90",
                    )
                    show_plot(fig, "xg_per90_vs_shots_per90")
                else:
                    st.info("Missing per-90 columns for xG or shots.")

            with row2[1]:
                if "touch_att_pen_per90" in summary.columns and "kp_per90" in summary.columns:
                    px_fig = px.scatter(
                        summary,
                        x="touch_att_pen_per90",
                        y="kp_per90",
                        color="team",
                        hover_name="player"
                    )
                    fig = go.Figure()
                    fig.add_traces(px_fig.data)
                    fig.update_layout(
                        px_fig.layout,
                        height=350,
                        margin=dict(t=40, b=30, l=20, r=20),
                        title=dict(text=f"Touches in Att Pen per 90 vs Key Passes per 90 - {gw_range_label}", x=0.5, xanchor="center"),
                        xaxis_title="Touches in Att Pen per 90",
                        yaxis_title="Key Passes per 90",
                    )
                    show_plot(fig, "touches_att_pen_vs_key_passes")
                else:
                    st.info("Missing per-90 columns for touches in box or key passes.")

            with row2[2]:
                st.markdown("<h3 style=\"text-align:center\">Upcoming Fixtures (FDR)</h3>", unsafe_allow_html=True)
                _render_fixtures_panel(summary, fixture_calendar_df)
        else:
            with row2[0]:
                if position_label == "MID":
                    st.empty()
                else:
                    st.empty()

            #Defensive Contributions
            with row2[1]:
                df = summary.copy()

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
                fig.add_traces(px_fig.data)        # ?. adds the traces, not the figure
                x_mean = df[tkl].mean()
                y_mean = df[recov].mean()
                fig.add_shape(
                    type="line",
                    x0=x_mean, x1=x_mean, y0=df[recov].min(), y1=df[recov].max(),
                    line=dict(color="#9aa0a6", width=1, dash="dash")
                )
                fig.add_shape(
                    type="line",
                    x0=df[tkl].min(), x1=df[tkl].max(), y0=y_mean, y1=y_mean,
                    line=dict(color="#9aa0a6", width=1, dash="dash")
                )
                fig.update_layout(
                    px_fig.layout,
                    height=350,        # dY`^ critical
                    margin=dict(t=40, b=30, l=20, r=20),
                    title = dict (
                        text=f"Defensive Contributions: Tackles+Interceptions vs Recoveries - {gw_range_label}",
                        x=0.5,
                        xanchor="center"
                        ),
                    xaxis_title='Tackles & Interceptions',
                    yaxis_title='Recoveries',)   # ?. optional: copy layout too

                show_plot(fig, "defensive_contributions")

            with row2[2]:
                st.markdown("<h3 style=\"text-align:center\">Upcoming Fixtures (FDR)</h3>", unsafe_allow_html=True)
                _render_fixtures_panel(summary, fixture_calendar_df)

        row3 = st.container(border=True)
        with row3:
            st.markdown("<h3 style=\"text-align:center\">Player Match Logs</h3>", unsafe_allow_html=True)
            search = st.text_input(
                "Search player",
                key=f"player_search_{position_label}",
                placeholder="Type a player name..."
            )

            stats_df = df.copy()
            if search:
                stats_df = stats_df[stats_df["player"].str.contains(search, case=False, na=False)]

            base_cols = [
                "player", "team", "gw", "game_date", "opponent", "opponent_id", "is_home",
                "min", "gls", "ast", "xg", "xa", "fpl_points"
            ]
            show_cols = [c for c in base_cols if c in stats_df.columns]
            if "game_date" in stats_df.columns:
                stats_df = stats_df.sort_values(["player", "gw", "game_date"])

            st.dataframe(stats_df[show_cols], use_container_width=True, hide_index=True, height=350)


tabs = st.tabs(["All", "Forwards", "Midfielders", "Defenders"])

with tabs[0]:
    render_player_tab(filtered_df, "All")

with tabs[1]:
    render_player_tab(filtered_df[filtered_df["position"] == "FWD"], "FWD")

with tabs[2]:
    render_player_tab(filtered_df[filtered_df["position"] == "MID"], "MID")

with tabs[3]:
    render_player_tab(filtered_df[filtered_df["position"] == "DEF"], "DEF")
