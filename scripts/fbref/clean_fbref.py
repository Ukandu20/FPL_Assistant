#!/usr/bin/env python3
"""
clean_fbref_csvs.py – ALL-IN-ONE cleaner                 2025-07-31  rev K

∆ Added fpl_pos:
   • Inherit from player’s latest prior season (any league), else fallback.
   • Column saved in player_season + player_match CSVs and season_players.json
"""

from __future__ import annotations
import argparse, json, logging, re, unicodedata, threading, secrets, hashlib
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Pattern, Tuple, Optional

import pandas as pd
from tqdm import tqdm

# ───────────── CONSTANTS ─────────────
PREFIXES     = ("player_season_", "player_match_", "team_season_", "team_match_")
TEAMLIKE     = {"team", "club", "squad"}
ID_LIKE      = {"game_id", "player_id", "team_id", "game", "home", "away"}

TEAM_SLUG_RE = re.compile(r"/squads/([0-9a-f]{8})", re.I)
GAME_URL_RE  = re.compile(r"/matches/([0-9a-f]{8})", re.I)
GAME_RE      = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<home>.+?)-(?P<away>.+)")

MASTER_PLAYER_JSON = "master_players.json"
MASTER_TEAM_JSON   = "master_teams.json"
LEAGUE_MP_JSON     = "master_players.json"
LEAGUE_MT_JSON     = "master_teams.json"
SEASON_PLAYER_JSON = "season_players.json"
LOOKUP_PLAYER_JSON = "_id_lookup_players.json"
LOOKUP_TEAM_JSON   = "_id_lookup_teams.json"

lock = threading.Lock()

# ───────────── tiny helpers ─────────────
def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "")).strip().lower()

def load_json(p: Path) -> dict:
    if p.exists():
        try:  return json.loads(p.read_text("utf-8"))
        except Exception: logging.warning("Cannot parse %s – starting blank", p)
    return {}

def save_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), "utf-8")

def _load_map(p: Path) -> dict:
    raw = json.loads(p.read_text("utf-8"))
    return ({k.strip().upper(): v.strip().upper() for k, v in raw.items()} |
            {k.strip().lower(): v.strip().upper() for k, v in raw.items()})

def strip_prefix(name: str) -> str:
    for p in PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name

def get_player_id(name: str, mp: dict) -> str:
    if not name: return ""
    k = _norm_key(name)
    with lock:
        if k not in mp:
            new = secrets.token_hex(4)
            while new in mp.values():
                new = secrets.token_hex(4)
            mp[k] = new
    return mp[k]

def get_team_id(name: str, mt: dict) -> str:
    if not name: return ""
    k = _norm_key(name)
    with lock:
        if k not in mt:
            new = secrets.token_hex(4)
            while new in mt.values():
                new = secrets.token_hex(4)
            mt[k] = new
    return mt[k]

def extract_team_slug(text: str | None) -> str | None:
    if not isinstance(text, str): return None
    m = TEAM_SLUG_RE.search(text)
    return m.group(1).lower() if m else None

def extract_game_slug(text: str | None) -> str | None:
    if not isinstance(text, str): return None
    m = GAME_URL_RE.search(text)
    return m.group(1).lower() if m else None

def make_game_id(date: str, home: str, away: str) -> str:
    key = f"{date}-{home}-{away}".encode()
    return hashlib.blake2b(key, digest_size=4).hexdigest()

def season_key(season_str: str) -> int:
    """Convert '2019-2020' → 2019 for sorting."""
    return int(season_str.split("-")[0])

def last_fpl_pos(pid: str, mp_global: dict, curr_season: str) -> str | None:
    rec = mp_global.get(pid)
    if not rec: return None
    past = [s for s in rec["career"] if season_key(s) < season_key(curr_season)]
    if not past: return None
    latest = max(past, key=season_key)
    return rec["career"][latest]["position"]

# NEW ──────────────────────────────────────────────────────────
def add_opponent_ids(df: pd.DataFrame) -> None:
    """
    Adds an opponent_id column in-place.
    Requires existing 'game_id' and 'team_id' columns.
    Fills '' for rows where a matching opponent cannot be determined.
    """
    if {"game_id", "team_id"} <= set(df.columns):
        pairs = df[["game_id", "team_id"]].drop_duplicates()
        gid_to_teams = pairs.groupby("game_id")["team_id"].apply(list).to_dict()

        def _opp(row):
            teams = gid_to_teams.get(row["game_id"])
            if teams and len(teams) == 2:
                return teams[1] if row["team_id"] == teams[0] else teams[0]
            return ""

        df["opponent_id"] = df.apply(_opp, axis=1)

# ───────────── header helpers ─────────────
def normalize(col: str) -> str:
    s = re.sub(r"(?i)^unnamed(?:_\d+)?_", "", str(col)).lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return re.sub(r"__+", "_", s).strip("_")

def flatten_header(df: pd.DataFrame) -> pd.DataFrame:
    # handle already-flat frames (mostly *_schedule CSVs)
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels == 1:
        df.columns = [normalize(c) for c in df.columns]
        return df
    keep, names, seen = [], [], set()
    for idx, triple in enumerate(df.columns):
        parts = [str(x).strip() for x in triple]
        label = next((x for x in (parts[2], parts[1], parts[0])
                      if x and not re.match(r"(?i)unnamed", x)), "")
        if not label: continue
        base = normalize(label)
        for cand in (base,
                     normalize(f"{parts[1]}_{label}"),
                     normalize(f"{parts[0]}_{base}")):
            if cand not in seen:
                keep.append(idx); names.append(cand); seen.add(cand); break
    flat = df.iloc[:, keep].copy()
    flat.columns = names
    return flat.loc[:, ~flat.columns.duplicated()]

def choose_subfolder(stem: str):
    if stem.startswith("player_season_"): return "player_season"
    if stem.startswith("player_match_" ): return "player_match"
    if stem.startswith("team_season_"  ): return "team_season"
    if stem.startswith("team_match_"   ): return "team_match"
    return "player_match"

def split_game_cols(df: pd.DataFrame, team_map: dict):
    parts = df["game"].astype(str).str.extract(GAME_RE)
    good  = parts["date"].notna()
    if not good.any(): return None
    df.loc[good, "game_date"] = parts.loc[good, "date"]
    df.loc[good, "is_home"]   = parts.loc[good, "home"].str.strip().apply(
        lambda s: team_map.get(s.lower(), team_map.get(s.upper(), s.strip())))
    df.loc[good, "is_away"]   = parts.loc[good, "away"].str.strip().apply(
        lambda s: team_map.get(s.lower(), team_map.get(s.upper(), s.strip())))
    return good

# ───────────── master-merge helpers ─────────────
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
        "team"    : row.get("team"),
        "position": row.get("position"),
        "fpl_position" : row.get("fpl_pos"),
        "league"  : league
    }

def merge_team(master, season, league, team_id, team_name, player_map):
    rec = master.setdefault(team_id, {"name": team_name, "career": {}})
    rec["career"][season] = {
        "league" : league,
        "players": [{"id": pid, "name": player_map[pid]} for pid in sorted(player_map)]
    }

# ───────────── core cleaner for one CSV ─────────────
def clean_csv(
    path: Path,
    season: str,
    league: str,
    root: Path,
    rules,
    mp_lookup,
    mt_lookup,
    mp_league,
    mt_league,
    mp_global,
    season_players: dict,
    season_teams: dict,
    pos_counts: dict[str, Counter],
    pos_map: dict,
    team_map: dict,
    force=False,
):
    stem = path.stem
    for pat, repl in rules:
        if pat.search(stem):
            stem = pat.sub(repl, stem); break
    out = root / "fbref" / league / season / choose_subfolder(stem) / f"{strip_prefix(stem)}.csv"
    if out.exists() and not force:
        return

    sched = path.stem.endswith("schedule")
    # schedule tables have a single header row
    df = pd.read_csv(path, header=[0] if sched else [0, 1, 2])

    df  = flatten_header(df)
    sub = choose_subfolder(path.stem)

    # 1️⃣  team mapping
    for col in df.columns:
        if col.lower() in TEAMLIKE:
            df[col] = (
                df[col].astype(str).apply(
                    lambda s: team_map.get(s.strip().lower(),
                                           team_map.get(s.strip().upper(),
                                                        s.strip().lower()))
                ).str.upper()          # ← ensure final value is MUN, ARS …
            )

    # 2️⃣  split game & game_id
    if sub in {"player_match", "team_match"} and "game" in df.columns:
        good = split_game_cols(df, team_map)
        if good is not None and "game_id" not in df.columns:
            def _gid(row):
                slug = next((extract_game_slug(str(v))
                             for v in row if isinstance(v, str)
                             if extract_game_slug(str(v))), None)
                return slug or make_game_id(row["game_date"], row["is_home"], row["is_away"])
            df.loc[good, "game_id"] = df.loc[good].apply(_gid, axis=1)

    # 3️⃣  numeric coercion (skip id-like + age/born)
    for c in df.columns:
        if c in ID_LIKE or c in ("age", "born") or df[c].dtype != object:
            continue
        num = pd.to_numeric(df[c].astype(str)
                            .str.replace(r"[,\u202f%]", "", regex=True),
                            errors="coerce")
        if 0 < num.notna().sum() < len(num):
            df[c] = num

    # 4️⃣  age/born nullable Int64
    for col in ("age", "born"):
        if col in df.columns:
            df[col] = (df[col].astype(str)
                               .str.split("-", n=1).str[0]
                               .str.extract(r"(\d+)", expand=False)
                               .astype("Int64"))

    # ── player_season extras ───────────────────────────────
    if sub == "player_season":
        df["player_id"] = df["player"].apply(lambda n: get_player_id(n, mp_lookup))

        pcol = next((c for c in df.columns if c.lower().startswith("pos")), None)
        df["pri_position"] = (
            df[pcol].astype(str).str.split(",", n=1).str[0].str.strip().str.upper()
            if pcol else "")
        df["position"] = df["pri_position"].map(pos_map).fillna("UNK")

        # compute fpl_pos and build season_players
        for _, r in df.iterrows():
            pid  = r["player_id"]
            team = r.get("team")

            prev_pos = last_fpl_pos(pid, mp_global, season)
            fpl_pos  = prev_pos or r["position"]

            born_year = int(r["born"]) if pd.notna(r["born"]) else None
            season_players[pid] = {
                "player_id"   : pid,
                "name"        : r["player"],
                "team"        : team,
                "nation"      : r.get("nation"),
                "born"        : born_year,
                "pri_position": r["pri_position"],
                "position"    : r["position"],
                "fpl_pos"     : fpl_pos
            }

            if team:
                tid = get_team_id(team, mt_lookup)
                mt_league[_norm_key(team)] = tid
                season_teams.setdefault(tid, {"name": team, "players": {}})["players"][pid] = r["player"]

            mp_league.setdefault(_norm_key(r["player"]), pid)

        # attach fpl_pos column
        df["fpl_pos"] = df["player_id"].map(lambda pid: season_players[pid]["fpl_pos"])

    # ── player_match extras (position voting only; fpl_pos later) ──
    if sub == "player_match" and "player" in df:
        if "player_id" not in df:
            df["player_id"] = df["player"].apply(lambda n: get_player_id(n, mp_lookup))
        pcol = next((c for c in df.columns if c.lower().startswith("pos")), None)
        if pcol:
            df["_raw_pos"] = df[pcol].astype(str).str.split(",", n=1).str[0].str.strip().str.upper()
            for pid, raw in zip(df["player_id"], df["_raw_pos"]):
                if raw: pos_counts[pid][raw] += 1

    # ensure team_id
    if "team_id" not in df:
        url_cols = [c for c in df.columns if "url" in c.lower() or "link" in c.lower()]
        def row_team_id(r):
            for c in url_cols:
                slug = extract_team_slug(r[c])
                if slug: return slug
            return get_team_id((r.get("team") or r.get("club") or r.get("squad") or "").strip(), mt_lookup)
        df["team_id"] = df.apply(row_team_id, axis=1)


    # NEW: inject opponent_id for team_match & player_match
    if sub in {"team_match", "player_match"} and {"game_id","team_id"} <= set(df.columns):
        add_opponent_ids(df)

        
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, na_rep="")

# ───────────── orchestrator ─────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league"); ap.add_argument("--season")
    ap.add_argument("--raw-dir",   type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--rules",     type=Path)
    ap.add_argument("--pos-map",   type=Path, default=Path("data/config/position_map.json"))
    ap.add_argument("--team-map",  type=Path, default=Path("data/config/team_map.json"))
    ap.add_argument("--workers",   type=int, default=4)
    ap.add_argument("--force",     action="store_true")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level.upper(),
                        format="%(asctime)s %(levelname)s: %(message)s")

    rules      = [(re.compile(r["pattern"], re.I), r["replacement"])
                  for r in json.loads(a.rules.read_text("utf-8"))] if a.rules and a.rules.exists() else []
    pos_map    = _load_map(a.pos_map)
    team_map   = _load_map(a.team_map)
    pos_counts = defaultdict(Counter)

    mp_lookup = load_json(a.clean_dir / LOOKUP_PLAYER_JSON)
    mt_lookup = load_json(a.clean_dir / LOOKUP_TEAM_JSON)
    mp_global = load_json(a.clean_dir / MASTER_PLAYER_JSON)
    mt_global = load_json(a.clean_dir / MASTER_TEAM_JSON)

    for league_dir in a.raw_dir.iterdir():
        if not league_dir.is_dir(): continue
        league = league_dir.name
        if a.league and league != a.league: continue

        mp_league = load_json(a.clean_dir / "fbref" / league / LEAGUE_MP_JSON)
        mt_league = load_json(a.clean_dir / "fbref" / league / LEAGUE_MT_JSON)

        for season_dir in sorted(league_dir.iterdir()):
            if not season_dir.is_dir(): continue
            season = season_dir.name
            if a.season and season != a.season: continue

            ps_files    = [*season_dir.rglob("*player_season_*.csv")]
            other_files = [f for f in season_dir.rglob("*.csv") if f not in ps_files]

            season_players, season_teams = {}, {}
            mapper = ThreadPoolExecutor(a.workers).map if a.workers > 1 else map

            list(tqdm(mapper(lambda f: clean_csv(
                    f, season, league, a.clean_dir, rules,
                    mp_lookup, mt_lookup, mp_league, mt_league, mp_global,
                    season_players, season_teams, pos_counts,
                    pos_map, team_map, a.force), ps_files),
                total=len(ps_files), desc=f"{league} {season} player-season"))

            list(tqdm(mapper(lambda f: clean_csv(
                    f, season, league, a.clean_dir, rules,
                    mp_lookup, mt_lookup, mp_league, mt_league, mp_global,
                    season_players, season_teams, pos_counts,
                    pos_map, team_map, a.force), other_files),
                total=len(other_files), desc=f"{league} {season} other"))

            # majority-vote positions
            for pid, rec in season_players.items():
                raw = pos_counts[pid].most_common(1)[0][0] if pos_counts[pid] else rec["pri_position"]
                rec["pri_position"] = raw
                rec["position"]     = pos_map.get(raw, "UNK")
                # If this is the first season we met the player, fpl_pos may need update now
                if last_fpl_pos(pid, mp_global, season) is None:
                    rec["fpl_pos"] = rec["position"]

            # fpl_pos & other sync into player_match
            pid2pos = {pid: rec["position"] for pid, rec in season_players.items()}
            pid2fpl = {pid: rec["fpl_pos"]  for pid, rec in season_players.items()}

            base_dir = a.clean_dir / "fbref" / league / season

            # round map from team_match
            round_map: dict[str, str] = {}
            for fp in (base_dir / "team_match").glob("*.csv"):
                df_tm = pd.read_csv(fp)
                if {"game_id", "round"} <= set(df_tm.columns):
                    for gid, rnd in zip(df_tm["game_id"], df_tm["round"]):
                        if pd.notna(gid) and pd.notna(rnd):
                            round_map[str(gid)] = str(rnd)

            for fp in (base_dir / "player_match").glob("*.csv"):
                df_sync = pd.read_csv(fp)

                if "player_id" in df_sync.columns:
                    # position sync
                    df_sync["position"] = (
                        df_sync.get("position", pd.Series("UNK", index=df_sync.index))
                        .where(~df_sync["player_id"].map(pid2pos).notna(),
                               df_sync["player_id"].map(pid2pos))
                    )
                    # fpl_pos sync
                    df_sync["fpl_pos"] = df_sync["player_id"].map(pid2fpl)

                if "game_id" in df_sync.columns:
                    df_sync["round"] = df_sync["game_id"].map(round_map)

                # dtype fix for age/born + team mapping
                for col in ("age", "born"):
                    if col in df_sync.columns:
                        df_sync[col] = (
                            df_sync[col].astype(str)
                            .str.extract(r"(\d+)", expand=False)
                            .astype("Int64")
                        )
                for col in df_sync.columns:
                    if col.lower() in TEAMLIKE:
                        df_sync[col] = (
                            df_sync[col].astype(str).apply(
                                lambda s: team_map.get(s.strip().lower(),
                                                       team_map.get(s.strip().upper(),
                                                                    s.strip().lower()))
                            ).str.upper()
                        )
                df_sync.to_csv(fp, index=False, na_rep="")

            # ---------- JSON outputs ----------
            season_out = base_dir / "player_season"
            season_out.mkdir(parents=True, exist_ok=True)
            save_json(season_out / SEASON_PLAYER_JSON, season_players)

            for info in season_players.values():
                merge_player(mp_league, season, league, info)
                merge_player(mp_global, season, league, info)
            for tid, trec in season_teams.items():
                merge_team(mt_league, season, league, tid, trec["name"], trec["players"])
                merge_team(mt_global, season, league, tid, trec["name"], trec["players"])

            save_json(a.clean_dir / "fbref" / league / LEAGUE_MP_JSON, mp_league)
            save_json(a.clean_dir / "fbref" / league / LEAGUE_MT_JSON, mt_league)

    save_json(a.clean_dir / LOOKUP_PLAYER_JSON, mp_lookup)
    save_json(a.clean_dir / LOOKUP_TEAM_JSON,   mt_lookup)
    save_json(a.clean_dir / MASTER_PLAYER_JSON, mp_global)
    save_json(a.clean_dir / MASTER_TEAM_JSON,   mt_global)
    logging.info("🎉 Cleaning complete!")

if __name__ == "__main__":
    main()
