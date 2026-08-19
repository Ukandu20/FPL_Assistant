"""Public entry point for the FPL Assistant analytics application."""

from pathlib import Path
import sys


# Streamlit adds this file's directory (``apps/fpl``) to ``sys.path`` when the
# app is launched by filename. Add the repository root so page modules can
# import the shared ``apps.fpl`` package without requiring an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from apps.fpl.state import shortlist
from apps.fpl.ui import inject_global_styles


APP_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="FPL Assistant",
    page_icon="⚽",
    layout="wide",
)
inject_global_styles()

navigation = st.navigation(
    {
        "Plan": [
            st.Page(
                APP_ROOT / "pages" / "0_home.py",
                title="Gameweek Hub",
                icon="⚽",
                default=True,
                url_path="gameweek",
            ),
            st.Page(
                APP_ROOT / "pages" / "0_main.py",
                title="Players",
                icon="👤",
                url_path="players",
            ),
            st.Page(
                APP_ROOT / "pages" / "3_compare.py",
                title="Compare",
                icon="⚖️",
                url_path="compare",
            ),
        ],
        "Explore": [
            st.Page(
                APP_ROOT / "pages" / "2_teams.py",
                title="Teams",
                icon="🛡️",
                url_path="teams",
            ),
            st.Page(
                APP_ROOT / "pages" / "1_league.py",
                title="League",
                icon="🏆",
                url_path="league",
            ),
        ]
    },
    position="sidebar",
    expanded=True,
)
st.sidebar.caption(f"Shortlist · {len(shortlist())} players")
navigation.run()
