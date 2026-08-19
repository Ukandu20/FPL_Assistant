"""Compatibility launcher for the retired legacy player-analysis page.

The canonical player experience now lives in ``0_main.py`` and is exposed as
"Player Card" by ``apps/fpl/app.py``. Keeping this small launcher avoids a hard
failure for old direct commands and bookmarks without maintaining two player
data pipelines.
"""

from pathlib import Path
import runpy
import sys


PLAYER_CARD_PAGE = Path(__file__).with_name("0_main.py")
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# When this compatibility page is launched directly, Streamlit places the
# ``pages`` directory on ``sys.path`` instead of the repository root. Bootstrap
# the root before executing the canonical page so ``apps.fpl`` imports resolve
# in both direct and multipage launches.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    runpy.run_path(str(PLAYER_CARD_PAGE), run_name="__main__")
