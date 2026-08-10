"""Compatibility launcher for the retired legacy player-analysis page.

The canonical player experience now lives in ``0_main.py`` and is exposed as
"Player Card" by ``apps/fpl/app.py``. Keeping this small launcher avoids a hard
failure for old direct commands and bookmarks without maintaining two player
data pipelines.
"""

from pathlib import Path
import runpy


PLAYER_CARD_PAGE = Path(__file__).with_name("0_main.py")

if __name__ == "__main__":
    runpy.run_path(str(PLAYER_CARD_PAGE), run_name="__main__")
