from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit_shadcn_ui as ui

st.set_page_config(page_title="Fantasy Premier league dashboard", layout="wide")

FPL_ROOT = Path("data/processed/fpl")
FBREF_ROOT = Path("data/processed/fbref")
WHOSCORED_ROOT = Path("data/processed/whoscored")
UNDERSTAT_ROOT = Path("data/processed/understat")
PROCESSED_ROOT = Path("data/processed")
EXCLUDED_DIRS = {"_audit"}

DATA_LIST = ["fpl", "fbref", "whoscored", "understat", "clubelo"]
SEASON_LIST = ["2026-2027","2025-2026", "2024-2025", "2023-2024", "2022-2023", "2021-2022"]

@st.cache_data(show_spinner=False)
def discover_data():
    if not PROCESSED_ROOT.exists():
        return []
    data_providers = [
        d.name
        for d in PROCESSED_ROOT.iterdir()
        if d.is_dir() and d.name in DATA_LIST
    ]
    return data_providers

@st.cache_data(show_spinner=False)
def discover_leagues() -> list[str]:
    if not PROCESSED_ROOT.exists():
        return []
    leagues = [
        d.name
        for d in PROCESSED_ROOT.iterdir()
        if d.is_dir() and d.name not in EXCLUDED_DIRS
    ]
    return sorted(leagues)

@st.cache_data(show_spinner=False)
def discover_seasons(data_provider, league):
    data_dir = PROCESSED_ROOT / data_provider / league
    if not data_dir.exists():
        return []
    seasons = [
        d.name
        for d in data_dir.iterdir()
        if d.is_dir() and d.name in SEASON_LIST
    ]
    return seasons

def build_player_bio(df):
    required_cols = [
        "player_id",
        "name",
        "team",
        "team_id",
        "fpl_pos",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return pd.DataFrame()
    table = df.copy()


def main() :
    st.title("Fantasy Premier League Player Dashboard")
    fruit = ui.select(
        "Fruit",
        ["Apple", "Banana", "Orange"],
        value="Banana",
    )

    enabled = ui.switch("Enable notifications", value=True)

    if ui.button("Save"):
        st.write({"fruit": fruit, "enabled": enabled})

        data_providers = discover_data()
        seasons = discover_seasons("fpl", "ENG-Premier League")
        st.sidebar.radio(
            "Data", options=data_providers)
        st.sidebar.radio(
            "Data", options=seasons)


if __name__ == "__main__":
    main()
