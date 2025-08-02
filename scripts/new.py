#!/usr/bin/env python3
"""
clean_fbref_csvs.py – ALL-IN-ONE (league & global masters)   2025-07-30 rev B

• Cleans FBref CSVs → data/processed/fbref/<LEAGUE>/<SEASON>/<player|team>_*/*.csv
• Stable 8-hex player_id / team_id (lookup JSONs keep them persistent)
• Builds per-season, per-league, and global master JSONs

Changes in this version
───────────────────────
1. primary position = **most-common raw “Pos” across all player_match rows**
2. External maps
      data/config/position_map.json   raw → canonical   (GK/DEF/MID/FWD)
      data/config/team_map.json       name → short code (ARS, MCI, …)
   Override via --pos-map / --team-map.
3. “LW” / “RW” now map to “MID” in the default pos-map.
"""

from __future__ import annotations
import argparse, json, logging, re, unicodedata, threading, secrets
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib  import Path
from typing   import Pattern, Tuple, Optional, Dict, List

import pandas as pd
from tqdm import tqdm

# ─────────────── CONSTANTS ───────────────
RenameRule   = Tuple[Pattern, str]
PREFIXES     = ("player_season_", "player_match_", "team_season_", "team_match_")
ID_LIKE      = {"game_id", "player_id", "team_id", "game", "home", "away"}
TEAM_SLUG_RE = re.compile(r"/squads/([0-9a-f]{8})/", re.I)
GAME_RE      = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<home>.+?)-(?P<away>.+)")

MASTER_PLAYER_JSON = "master_players.json"
MASTER_TEAM_JSON   = "master_teams.json"
LEAGUE_MP_JSON     = "master_players.json"
LEAGUE_MT_JSON     = "master_teams.json"
SEASON_PLAYER_JSON = "season_players.json"
LOOKUP_PLAYER_JSON = "_id_lookup_players.json"
LOOKUP_TEAM_JSON   = "_id_lookup_teams.json"

lock = threading.Lock()

# ───────────── JSON helpers ─────────────
def load_json(p: Path) -> dict:
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            logging.warning("Could not parse %s; starting blank", p)
    return {}

def save_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), "utf-8")

def _load_map(p: Path, kind: str) -> dict:
    if not p.is_file():
        raise FileNotFoundError(f"{kind} map not found: {p}")
    raw = json.loads(p.read_text("utf-8"))
    return {k.strip().upper(): v.strip().upper() for k, v in raw.items()}

# ───────────── ID generation ────────────
def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "")).strip().lower()

def get_player_id(name: str, mp: dict) -> str:
    if not name:
        return ""
    k = _norm_key(name)
    with lock:
        if k not in mp:
            new = secrets.token_hex(4)
            while new in mp.values():
                new = secrets.token_hex(4)
            mp[k] = new
    return mp[k]

def extract_team_slug(val: str | None) -> str | None:
    if not isinstance(val, str):
        return None
    m = TEAM_SLUG_RE.search(val)
    return m.group(1).lower() if m else None

def get_team_id(name: str, mt: dict) -> str:
    if not name:
        return ""
    k = _norm_key(name)
    with lock:
        if k not in mt:
            new = secrets.token_hex(4)
            while new in mt.values():
                new = secrets.token_hex(4)
            mt[k] = new
    return mt[k]

def find_pos_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = c.lower()
        if cl == "pos" or cl.startswith("pos") or cl.startswith("position"):
            return c
    return None



# ─────────── master-merge helpers ──────────
def merge_player(master: dict, season: str, league: str, row: dict):
    pid = row["player_id"]
    career = master.setdefault(pid, {
        "name":   row["name"],
        "nation": row.get("nation"),
        "born":   row.get("born"),
        "career": {}
    })
    career.setdefault("nation", row.get("nation"))
    career.setdefault("born",   row.get("born"))
    career["career"][season] = {
        "team":     row.get("team"),
        "position": row.get("position"),
        "league":   league
    }

def merge_team(master, season, league, team_id, team_name, player_map):
    rec = master.setdefault(team_id, {"name": team_name, "career": {}})
    rec["career"][season] = {
        "league": league,
        "players": [{"id": pid, "name": player_map[pid]} for pid in sorted(player_map)]
    }

# ───────────── misc helpers ─────────────
def normalize(col: str, drop_numeric=True) -> str:
    s = re.sub(r"(?i)^unnamed(?:_\d+)?_", "", str(col)).lower()
    for a, b in (("+", "_plus_"), ("-", "_minus_"), ("/", "_"), ("%", "_pct")):
        s = s.replace(a, b)
    s = re.sub(r"\bsquad\b", "club", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"__+", "_", s).strip("_")
    if drop_numeric:
        s = "_".join(tok for tok in s.split("_") if not tok.isdigit())
    return s

def load_rules(p: Optional[Path]):
    if not p or not p.exists():
        return []
    return [(re.compile(r["pattern"], re.I), r["replacement"])
            for r in json.loads(p.read_text("utf-8"))]

def apply_rules(stem: str, rules):
    for pat, repl in rules:
        if pat.search(stem):
            return repl
    return stem

def strip_prefix(stem: str):
    for p in PREFIXES:
        if stem.startswith(p):
            return stem[len(p):]
    return stem

def choose_subfolder(stem: str):
    if stem.startswith("player_season_"):
        return "player_season"
    if stem.startswith("player_match_"):
        return "player_match"
    if stem.startswith("team_season_"):
        return "team_season"
    if stem.startswith("team_match_"):
        return "team_match"
    return "player_match"


def split_game_column(df: pd.DataFrame, team_map: dict):
    if "game" not in df.columns:
        return
    parts = df["game"].astype(str).str.extract(GAME_RE)
    if parts["date"].isna().all():
        return
    df["game_date"] = parts["date"]
    home_short = parts["home"].apply(lambda s: team_map.get(s.strip().upper(), s.strip()))
    away_short = parts["away"].apply(lambda s: team_map.get(s.strip().upper(), s.strip()))
    df["is_home"] = home_short
    df["is_away"] = away_short


# ──────────── core cleaner ────────────
def clean_csv(
    path: Path,
    season: str,
    league: str,
    root: Path,
    rules,
    drop_numeric: bool,
    mp_lookup,
    mt_lookup,
    mp_league,
    mt_league,
    season_players: dict,
    season_teams: dict,
    position_counts: dict,
    pos_map: dict,
    team_map: dict,
    force=False,
):
    stem = apply_rules(path.stem, rules)
    out = root / "fbref" / league / season / choose_subfolder(stem) / f"{strip_prefix(stem)}.csv"
    if out.exists() and not force:
        return

    sched = path.stem.endswith("schedule")
    df = pd.read_csv(path, header=[0] if sched else [0, 1, 2])

    for col in ("team", "club"):
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda s: team_map.get(s.strip().upper(), s.strip())
            )

    if sub in ("player_match", "team_match"):
        split_game_column(df, team_map)
        
    if sched:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        return

    sub = choose_subfolder(path.stem)

    # flatten header
    keep, new_cols, seen = [], [], set()
    for ix, col in enumerate(df.columns):
        lvl0, lvl1, lvl2 = [str(x).strip() for x in col]
        label = next((x for x in (lvl2, lvl1, lvl0)
                      if x and not re.match(r"(?i)unnamed", x)), "")
        if not label:
            continue
        base = normalize(label, drop_numeric)
        for cand in (base,
                     normalize(f"{lvl1}_{label}", drop_numeric),
                     normalize(f"{lvl0}_{base}", drop_numeric)):
            if cand not in seen:
                flat = cand
                break
        keep.append(ix), new_cols.append(flat), seen.add(flat)
    df = df.iloc[:, keep]
    df.columns = new_cols
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # numeric coercion
    for c in df.columns:
        if c in ID_LIKE or c in ("age", "born") or df[c].dtype != object:
            continue
        num = pd.to_numeric(df[c].astype(str).str.replace(r"[,\u202f%]", "", regex=True),
                            errors="coerce")
        if 0 < num.notna().sum() < len(num):
            df[c] = num

    # ─── player_season ───────────────────────────────────────────────
    if sub == "player_season":
        df["player_id"] = df["player"].apply(lambda n: get_player_id(n, mp_lookup))
        pcol = find_pos_col(df)
        df["pri_position"] = (df[pcol].astype(str).str.split(",", n=1).str[0]
                              .str.strip().str.upper()) if pcol else ""
        df["position"] = df["pri_position"].map(pos_map).fillna("UNK")

        
        for _, r in df.iterrows():
            pid = r["player_id"]
            full_team = (r.get("team") or r.get("club") or "").strip()
            short_team = team_map.get(full_team.upper(), full_team)
            if full_team:
                tid = get_team_id(full_team, mt_lookup)
                mt_league[_norm_key(full_team)] = tid
                season_teams.setdefault(tid, {"name": short_team, "players": {}})["players"][pid] = r["player"]

            born_val = r.get("born")
            born_year = int(born_val.year) if isinstance(born_val, pd.Timestamp) \
                        else int(born_val) if pd.notna(born_val) else None

            season_players[pid] = {
                "player_id": pid,
                "name": r["player"],
                "team": short_team,
                "nation": r.get("nation"),
                "born": born_year,
                "pri_position": r["pri_position"],
                "position": r["position"],
            }
            mp_league.setdefault(_norm_key(r["player"]), pid)

    # ─── player_match ────────────────────────────────────────────────
    if sub == "player_match" and "player" in df:
        if "player_id" not in df:
            df["player_id"] = df["player"].apply(lambda n: get_player_id(n, mp_lookup))

        df["_raw_pos"] = df["pos"].astype(str).str.split(",",n= 1).str[0].str.strip().str.upper()
        for pid, raw in zip(df["player_id"], df["_raw_pos"]):
            if raw:
                position_counts[pid][raw] += 1

        num_col = next((c for c in ("jersey_number", "number") if c in df.columns), None)
        if num_col:
            for _, r in df.iterrows():
                pid = r["player_id"]
                if pid in season_players and not season_players[pid].get("jersey_number"):
                    season_players[pid]["jersey_number"] = r[num_col]

    # ─── ensure team_id column ───────────────────────────────────────
    if "team_id" not in df:
        url_cols = [c for c in df.columns if any(k in c.lower() for k in ("url", "link", "squad"))]

        def team_id_row(r):
            for c in url_cols:
                slug = extract_team_slug(r[c])
                if slug:
                    return slug
            full_team = (r.get("team") or r.get("club") or "").strip()
            return get_team_id(full_team, mt_lookup)

        df["team_id"] = df.apply(team_id_row, axis=1)

    # save cleaned CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

# ─────────────── CLI & orchestrator ───────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league")
    ap.add_argument("--season")
    ap.add_argument("--raw-dir",   type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--rules",     type=Path)
    ap.add_argument("--pos-map",   type=Path, default=Path("data/config/position_map.json"))
    ap.add_argument("--team-map",  type=Path, default=Path("data/config/team_map.json"))
    ap.add_argument("--workers",   type=int, default=4)
    ap.add_argument("--force",     action="store_true")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s %(levelname)s: %(message)s")

    rules           = load_rules(a.rules)
    pos_map         = _load_map(a.pos_map,  "Position")
    team_map        = _load_map(a.team_map, "Team")
    position_counts = defaultdict(Counter)      # pid → Counter

    mp_lookup  = load_json(a.clean_dir / LOOKUP_PLAYER_JSON)
    mt_lookup  = load_json(a.clean_dir / LOOKUP_TEAM_JSON)
    mp_global  = load_json(a.clean_dir / MASTER_PLAYER_JSON)
    mt_global  = load_json(a.clean_dir / MASTER_TEAM_JSON)

    for league_dir in a.raw_dir.iterdir():
        if not league_dir.is_dir():
            continue
        league = league_dir.name
        if a.league and league != a.league:
            continue

        mp_league = load_json(a.clean_dir / "fbref" / league / LEAGUE_MP_JSON)
        mt_league = load_json(a.clean_dir / "fbref" / league / LEAGUE_MT_JSON)

        for season_dir in sorted(league_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            season = season_dir.name
            if a.season and season != a.season:
                continue

            ps_files    = [f for f in season_dir.rglob("*player_season_*.csv")]
            other_files = [f for f in season_dir.rglob("*.csv") if f not in ps_files]

            season_players: dict = {}
            season_teams:   dict = {}

            mapper = ThreadPoolExecutor(a.workers).map if a.workers > 1 else map

            # pass 1 – player_season
            list(tqdm(mapper(lambda f: clean_csv(
                    f, season, league, a.clean_dir, rules, True,
                    mp_lookup, mt_lookup, mp_league, mt_league,
                    season_players, season_teams, position_counts,
                    pos_map, team_map, a.force), ps_files),
                total=len(ps_files),
                desc=f"{league} {season} player-season"))

            # pass 2 – everything else
            list(tqdm(mapper(lambda f: clean_csv(
                    f, season, league, a.clean_dir, rules, True,
                    mp_lookup, mt_lookup, mp_league, mt_league,
                    season_players, season_teams, position_counts,
                    pos_map, team_map, a.force), other_files),
                total=len(other_files),
                desc=f"{league} {season} other"))

            # ── derive final position from match counts ───────────────
            for pid, rec in season_players.items():
                if position_counts[pid]:
                    raw = position_counts[pid].most_common(1)[0][0]
                else:
                    raw = rec["pri_position"]
                rec["pri_position"] = raw
                rec["position"]     = pos_map.get(raw, "UNK")

            # season JSON
            season_out = a.clean_dir / "fbref" / league / season / "player_season"
            season_out.mkdir(parents=True, exist_ok=True)
            save_json(season_out / SEASON_PLAYER_JSON, season_players)

            # merge into masters
            for info in season_players.values():
                merge_player(mp_league, season, league, info)
                merge_player(mp_global, season, league, info)
            for tid, trec in season_teams.items():
                merge_team(mt_league, season, league, tid, trec["name"], trec["players"])
                merge_team(mt_global, season, league, tid, trec["name"], trec["players"])

            save_json(a.clean_dir / "fbref" / league / LEAGUE_MP_JSON, mp_league)
            save_json(a.clean_dir / "fbref" / league / LEAGUE_MT_JSON, mt_league)

    # global writes
    save_json(a.clean_dir / LOOKUP_PLAYER_JSON, mp_lookup)
    save_json(a.clean_dir / LOOKUP_TEAM_JSON,   mt_lookup)
    save_json(a.clean_dir / MASTER_PLAYER_JSON, mp_global)
    save_json(a.clean_dir / MASTER_TEAM_JSON,   mt_global)
    logging.info("🎉 Cleaning complete!")

if __name__ == "__main__":
    main()
