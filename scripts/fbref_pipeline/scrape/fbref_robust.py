# scripts/fbref_pipeline/scrape/fbref_robust.py
from __future__ import annotations
import re
import time
from typing import Literal, Optional, Tuple, List, Dict

import pandas as pd
import requests
from lxml import html

League = Literal[
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (compatible; FPL-Assistant/1.0; +https://example.com)",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}

LEAGUE_MAP: dict[League, dict[str, str | int]] = {
    "ENG-Premier League": {"comp_id": 9,  "slug": "Premier-League"},
    "ESP-La Liga":        {"comp_id": 12, "slug": "La-Liga"},
    "ITA-Serie A":        {"comp_id": 11, "slug": "Serie-A"},
    "GER-Bundesliga":     {"comp_id": 20, "slug": "Bundesliga"},
    "FRA-Ligue 1":        {"comp_id": 13, "slug": "Ligue-1"},
}


# ───────────────────────── Internals ─────────────────────────

def _season_str_ok(season: str) -> str:
    s = str(season)
    if not re.match(r"^\d{4}-\d{4}$", s):
        raise ValueError(f"season must be 'YYYY-YYYY', got {s!r}")
    return s

def _fetch_season_main_page(league: League, season: str, pause: float = 0.6) -> tuple[str, html.HtmlElement]:
    """Fetch main competition season page; many tables are in HTML comments → we decomment."""
    meta = LEAGUE_MAP[league]
    comp = meta["comp_id"]
    slug = meta["slug"]
    season = _season_str_ok(season)
    url = f"https://fbref.com/en/comps/{comp}/{season}/{season}-{slug}-Stats"
    time.sleep(pause)
    r = requests.get(url, headers=HEADERS, timeout=45)
    r.raise_for_status()
    # FBref often comments out the stats tables. Make them visible for XPath.
    text = re.sub(r"<!--|-->", "", r.text)
    doc = html.fromstring(text)
    return url, doc

def _pick_table(nodes: List[html.HtmlElement], want: Optional[str]) -> html.HtmlElement:
    """Pick a table node; prefer id containing 'for' or 'against' per want."""
    if not nodes:
        raise RuntimeError("No candidate tables")
    if want is None:
        return nodes[0]
    for n in nodes:
        nid = n.get("id", "")
        if want in nid:
            return n
    return nodes[0]

def _read_table(node: html.HtmlElement) -> pd.DataFrame:
    """Parse an lxml table element into a clean pandas DataFrame."""
    df = pd.read_html(html.tostring(node, encoding="unicode"))[0]
    # Flatten multiindex headers (FBref loves these)
    if getattr(df, "columns", None) is not None and getattr(df.columns, "nlevels", 1) > 1:
        df.columns = [
            " ".join([str(c) for c in col if c and not str(c).startswith("Unnamed")]).strip()
            for col in df.columns.values
        ]
    df = df.dropna(how="all")
    return df


# ───────────────────────── Robust readers (season level) ─────────────────────────

def read_team_season_stats_robust(
    league: League,
    season: str,
    stat_type: str,
    opponent_stats: bool = False,
) -> pd.DataFrame:
    """
    Robust TEAM season stats from main league page.
    Matches ids like: stats_squads_<stat_type>_{for|against} OR stats_teams_<stat_type>*
    """
    url, doc = _fetch_season_main_page(league, season)
    want = "against" if opponent_stats else "for"
    xpaths = [
        f"//table[contains(@id, 'stats_squads_{stat_type}')]",
        f"//table[contains(@id, 'stats_teams_{stat_type}')]",
        f"//table[contains(@id, '{stat_type}') and contains(@id, 'squads')]",
        f"//table[contains(@id, '{stat_type}') and contains(@id, 'teams')]",
    ]
    nodes: List[html.HtmlElement] = []
    for xp in xpaths:
        nodes = doc.xpath(xp)
        if nodes:
            break
    if not nodes:
        raise RuntimeError(f"No tables for stat_type={stat_type!r} at {url}")
    node = _pick_table(nodes, want)
    df = _read_table(node)

    # Normalize common columns
    rename = {"Squad": "team", "Rk": "rk"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["opponent_stats"] = bool(opponent_stats)
    df["season"] = season
    df["league"] = league
    return df

def read_player_season_stats_robust(
    league: League,
    season: str,
    stat_type: str,
    opponent_stats: bool = False,
) -> pd.DataFrame:
    """
    Robust PLAYER season stats from main league page.
    Matches ids like: stats_players_<stat_type>_{for|against}
    """
    url, doc = _fetch_season_main_page(league, season)
    want = "against" if opponent_stats else "for"
    xpaths = [
        f"//table[contains(@id, 'stats_players_{stat_type}')]",
        f"//table[contains(@id, '{stat_type}') and contains(@id, 'players')]",
    ]
    nodes: List[html.HtmlElement] = []
    for xp in xpaths:
        nodes = doc.xpath(xp)
        if nodes:
            break
    if not nodes:
        raise RuntimeError(f"No PLAYER tables for stat_type={stat_type!r} at {url}")
    node = _pick_table(nodes, want)
    df = _read_table(node)

    rename = {"Player": "player", "Squad": "team", "Rk": "rk"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["opponent_stats"] = bool(opponent_stats)
    df["season"] = season
    df["league"] = league
    return df


# ───────────────────────── Team-match fallback via player-match aggregation ─────────────────────────

_NUMERIC_PERCENT_HINTS = ("%", "per 90", "/90", "Per 90", "Pct", "Rate")
_MINUTES_CANDIDATES = ("Min", "minutes", "Minutes", "Playing Time Min", "Min_Playing Time")

def _is_percent_col(col: str) -> bool:
    col_l = col.lower()
    if any(tok.lower() in col_l for tok in _NUMERIC_PERCENT_HINTS):
        return True
    return col_l.endswith("%") or "pct" in col_l or col_l.endswith("_pct")

def _weighted_avg(series: pd.Series, weights: pd.Series) -> float:
    w = pd.to_numeric(weights, errors="coerce")
    x = pd.to_numeric(series, errors="coerce")
    if w.notna().sum() == 0 or float(w.sum() or 0) == 0.0:
        return float(pd.to_numeric(series, errors="coerce").mean())
    return float((x * w).sum() / w.sum())

def team_match_from_player_match_df(
    player_df: pd.DataFrame,
    league: str,
    season: str,
) -> pd.DataFrame:
    """
    Aggregate a player-match DataFrame into team-match level.
    - Sums for numeric totals.
    - Minutes-weighted averages for percentage-like columns when minutes are present.
    Requires columns: ['match_id', 'team'] plus numeric stats.
    """
    df = player_df.copy()
    # Identify team column
    if "team" not in df.columns and "Squad" in df.columns:
        df = df.rename(columns={"Squad": "team"})
    if "team" not in df.columns:
        raise ValueError("player_match DF lacks 'team' (or 'Squad') column.")

    # Minutes column for weighting
    min_col = None
    for c in _MINUTES_CANDIDATES:
        if c in df.columns:
            min_col = c
            break

    # Ensure season/league present for grouping keys
    if "season" not in df.columns:
        df["season"] = season
    if "league" not in df.columns:
        df["league"] = league

    g = df.groupby(["season", "league", "team"], dropna=False)

    # numeric columns to aggregate (drop obvious indices)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in num_cols if c not in {"rk"}]

    out_rows = []
    for keys, sub in g:
        row = dict(zip(["season", "league", "team"], keys))
        for c in num_cols:
            if _is_percent_col(c):
                if min_col and min_col in sub.columns:
                    row[c] = _weighted_avg(sub[c], sub[min_col])
                else:
                    row[c] = float(pd.to_numeric(sub[c], errors="coerce").mean())
            else:
                row[c] = float(pd.to_numeric(sub[c], errors="coerce").sum())
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out["source"] = "aggregated_from_player_match"
    return out

def team_match_from_soccerdata_fallback(
    fb,  # sd.FBref instance
    stat_type: str,
    league: str,
    season: str,
) -> pd.DataFrame:
    """
    If read_team_match_stats fails, try to pull player_match via soccerdata
    and aggregate to team-match level in-memory.
    """
    pm_df = fb.read_player_match_stats(stat_type=stat_type)
    if not isinstance(pm_df, pd.DataFrame) or pm_df.empty:
        raise RuntimeError(f"player_match for stat={stat_type!r} also unavailable for fallback.")
    return team_match_from_player_match_df(pm_df, league=league, season=season)
