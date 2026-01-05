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
from dash import Dash, dcc, html, Input, Output

import streamlit as st

#initialize and setup the streamlit app
st.set_page_config(
    page_title= "Player Analysis & Deep Dive",
    layout="wide",
    )
st.title("Player Analysis & Deep Dive")

st.sidebar.header("Players Analysis")


#import the data
pd.set_option("display.max_columns", None)

#match data
defending = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_match/defense.csv")
passing = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_match/passing.csv")
misc =  pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_match/misc.csv")
keepers= pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_match/keepers.csv")
pass_type = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_match/passing_types.csv")
possession = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_match/possession.csv")
summary = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_match/summary.csv")
schedule = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/team_match/schedule.csv")

#season data
szn_defending = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/defense.csv")
szn_passing = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/passing.csv")
szn_misc = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/misc.csv")
szn_gca = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/goal_shot_creation.csv")
szn_shooting = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/shooting.csv")
szn_possession = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/possession.csv")
szn_pass_type = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/passing_types.csv")
szn_standards = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/standard.csv")
szn_keepers= pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/keeper.csv")
szn_keeper_adv= pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/keeper_adv.csv")
szn_playing_time= pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/playing_time.csv")

#fpl data
fpl = pd.read_csv("../data/processed/fpl/2025-2026/gws/merged_gws.csv")

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

#merge match data
defending_df = defending_df.merge(misc_df[['player_id', 'game_id', 'recov']], left_on=['player_id', 'game_id'], right_on=['player_id', 'game_id'], how='left')
defending_df = defending_df.merge(summary_df[['player_id', 'game_id', 'position']], left_on=['player_id', 'game_id'], right_on=['player_id', 'game_id'], how='left')


#Creating the charts and plots

kpi = st.columns(4)
charts = st.columns(2)

for top in kpi + charts:
    top.metric(label="Placeholder", value="0", delta="0%")
    
container = st.container()
with container:
    chart1 = px.bar(szn_shooting_df.sort_values(by='gls', ascending=False).head(10), x='gls', y='player', title='Top 10 Players by Goals Scored (Gls) - 2025/26 Season', labels={'Gls': 'Goals Scored', 'player': 'Player'}, orientation='h', height=600)
    st.plotly_chart(chart1, use_container_width=True)
    chart2 = px.bar(szn_shooting_df.sort_values(by='xg', ascending=False).head(10), x='xg', y='player', title='Top 10 Players by Expected Goals (xG) - 2025/26 Season', labels={'xg': 'Expected Goals', 'player': 'Player'}, orientation='h', height=600)
    st.plotly_chart(chart2, use_container_width=True)