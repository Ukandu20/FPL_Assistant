#!/usr/bin/env python3
"""
Fetch squad pages once per season and build name→slug maps.

Usage:
  py fetch_rosters.py --league "ENG-Premier League" --season 2024-2025 \
      --out-dir data/raw/fbref
"""
import argparse, json, re, sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

SQUAD_URL_RE = re.compile(r"/en/squads/([0-9a-f]{8})/([^/]+)/([a-z\-]+)-Stats")
PLAYER_RE    = re.compile(r"/en/players/([0-9a-f]{8})/")

def roster_urls(league, season):
    """Return every squad URL for <league>/<season> (relies on FBref CSV you've already downloaded)."""
    root = Path(f"data/raw/fbref/{league}/{season}/team_season")
    for csv in root.glob("standard.csv"):
        try:
            df = pd.read_csv(csv, nrows=0)                      # UTF-8 first
        except UnicodeDecodeError:
            df = pd.read_csv(csv, nrows=0, encoding="latin1")   # fallback
        url_col = next((c for c in df.columns if "url" in c.lower()), None)
        if url_col:
            full = pd.read_csv(csv, usecols=[url_col])[url_col].dropna().unique()
            for u in full:
                m = SQUAD_URL_RE.search(u)
                if m:
                    yield "https://fbref.com"+m.group(0)

def build_slug_map(urls):
    mapping = {}
    for url in tqdm(urls, desc="rosters"):
        r = requests.get(url, timeout=20)
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.select("table#stats_standard a"):
            h = a.get("href","")
            m = PLAYER_RE.search(h)
            if m:
                slug = m.group(1)
                name = a.text.strip()
                key  = re.sub(r"\s+"," ",name.lower())
                mapping[key] = slug
    return mapping

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--league", required=True)
    p.add_argument("--season", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    urls = list(roster_urls(a.league, a.season))
    if not urls:
        sys.exit("No squad URLs discovered—have you scraped team_season tables?")
    slug_map = build_slug_map(urls)
    out = a.out_dir / a.league / a.season / "slug_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(slug_map, indent=2))
    print(f"✨ wrote {len(slug_map):,} player slugs → {out}")

if __name__ == "__main__":
    main()
