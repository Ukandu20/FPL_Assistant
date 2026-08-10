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


APP_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="FPL Assistant",
    page_icon="⚽",
    layout="wide",
)

navigation = st.navigation(
    {
        "Explore": [
            st.Page(
                APP_ROOT / "pages" / "1_league.py",
                title="League Insights",
                icon="🏆",
                default=True,
                url_path="league",
            ),
            st.Page(
                APP_ROOT / "pages" / "0_main.py",
                title="Player Card",
                icon="👤",
                url_path="players",
            ),
            st.Page(
                APP_ROOT / "pages" / "2_teams.py",
                title="Teams",
                icon="🛡️",
                url_path="teams",
            ),
        ]
    },
    position="sidebar",
    expanded=True,
)
navigation.run()
