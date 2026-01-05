import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

pd.set_option("display.max_columns", None)
defending = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_match/defense.csv")
szn_defending = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/defense.csv")
passing = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/passing.csv")
pass_type = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/passing_types.csv")
szn_misc = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_season/misc.csv")
matchday = pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/team_match/defense.csv")
misc =  pd.read_csv("../data/processed/fbref/ENG-Premier League/2025-2026/player_match/misc.csv")

defending_df = pd.DataFrame(defending)
szn_defending_df = pd.DataFrame(szn_defending)
passing_df = pd.DataFrame(passing)
pass_type_df = pd.DataFrame(pass_type)
matchday_df = pd.DataFrame(matchday)
misc_df = pd.DataFrame(misc)

fpl = pd.read_csv("../data/processed/fpl/2025-2026/gws/merged_gws.csv")

fpl_df = pd.DataFrame(fpl)


#add matchday to defending_df
defending_df = defending_df.merge(misc_df[['player_id', 'game_id', 'recov']], left_on=['player_id', 'game_id'], right_on=['player_id', 'game_id'], how='left')
defending_df = defending_df.merge(matchday_df[['round', 'game_id']], left_on='game_id', right_on='game_id', how='left')

#drop duplicate rows by name and game
defending_df = defending_df.drop_duplicates(subset=['player_id', 'game_id'])

#Create defcomp column
defending_df['defcon'] = defending_df['tkl_int'] + defending_df['blocks'] + defending_df['clr']
defending_df.head()



#filter the top 20 defensive contributions
midfielders = defending_df[defending_df['position'] == 'MID']
fpl_mids = fpl_df[fpl_df['position'] == 'MID']
players = midfielders.sort_values(by='defcon', ascending=False).head(30)

szn_mids = szn_defending_df[szn_defending_df['position'] == 'MID']
szn_mids = szn_mids.merge(szn_misc[['player_id', 'recov']], left_on='player_id', right_on='player_id', how='left')

szn_mids = szn_mids[szn_mids['90s'] > 5]
szn_mids['recov/90'] = szn_mids['recov'] / szn_mids['90s']
szn_mids['tkl_int/90'] = szn_mids['tkl_int'] / szn_mids['90s'] 






mid_defcon = midfielders['defcon'] + midfielders['recov']
midfielders['defcon'] = mid_defcon

rice = midfielders[midfielders['player'] == 'Declan Rice']
caicedo = midfielders[midfielders['player'] == 'Moisés Caicedo']


app = Dash()

chart1 = px.scatter(szn_mids, x='tkl_int/90', y='recov/90', hover_name='player', title="Top 30 Midfielders by Defensive Contributions in a single game")

app.layout = html.Div(children=[
    html.H1(children='Defensive Contributions Dashboard 2025-2026 Season'),
    dcc.Graph(
        id='defensive-contributions-scatter',
        figure=chart1
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)