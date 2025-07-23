#!/usr/bin/env python3
"""
scripts/extract_all_fbref_sheets.py

Reads every sheet from data/raw/2024-25.xlsx and writes it out as its own CSV:

  • Sheets starting with "Player" → data/raw/players/fbref/2024_25/<sheet_slug>.csv  
  • Sheets starting with "Squad"  → data/raw/teams/fbref/2024_25/<sheet_slug>.csv  
  • All others                    → data/raw/other_sheets/fbref/2024_25/<sheet_slug>.csv  
"""

import re
import pandas as pd
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
SEASON     = "2024_25"
EXCEL_FILE = Path("data/raw/2024-25.xlsx")

# ─── Helpers ─────────────────────────────────────────────────────────────────
def slugify(name: str) -> str:
    """Convert sheet name to filesystem-safe lowercase slug."""
    return re.sub(r'[^0-9a-zA-Z]+', '_', name).strip('_').lower()

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    xls = pd.ExcelFile(EXCEL_FILE)
    for sheet in xls.sheet_names:
        # ignore non‐string names
        if not isinstance(sheet, str):
            continue

        # parse with header row 2 (zero‐based index)
        df = xls.parse(sheet, header=1)

        # pick your output dir
        name_lower = sheet.lower()
        if name_lower.startswith("player"):
            out_dir = Path("data/raw/players/fbref") / SEASON
        elif name_lower.startswith("squad"):
            out_dir = Path("data/raw/teams/fbref") / SEASON
        else:
            out_dir = Path("data/raw/other_sheets/fbref") / SEASON

        # ensure directory exists
        out_dir.mkdir(parents=True, exist_ok=True)

        # write the CSV
        slug = slugify(sheet)
        out_file = out_dir / f"{slug}.csv"
        df.to_csv(out_file, index=False)

        print(f"✅ Saved '{sheet}' → {out_file}")

if __name__ == "__main__":
    main()
