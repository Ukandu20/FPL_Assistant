from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for candidate in (str(SRC), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

SOCCERDATA_DIR = ROOT / "artifacts" / "cache" / "soccerdata_test"
SOCCERDATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SOCCERDATA_DIR", str(SOCCERDATA_DIR.resolve()))

