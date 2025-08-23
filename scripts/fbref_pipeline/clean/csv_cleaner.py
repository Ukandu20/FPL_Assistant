#!/usr/bin/env python3
"""
clean_fbref_csvs.py – ALL-IN-ONE cleaner                 2025-08-03  rev N

Fixes
─────
• `is_promoted` now compares the standardised `team` code against the clubs
  present in the *immediately-previous* season, so team_season CSVs are correct.
• Keeps venue renaming, home/away booleans, opponent_id injection, fpl_pos, etc.
"""

from __future__ import annotations
import argparse, json, logging, re, unicodedata, threading, secrets, hashlib
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# ────────── CONSTANTS ──────────
PREFIXES     = ("player_season_", "player_match_", "team_season_", "team_match_")
TEAMLIKE     = {"team", "club", "squad"}
ID_LIKE      = {"game_id", "player_id", "team_id", "game", "home", "away"}

TEAM_SLUG_RE = re.compile(r"/squads/([0-9a-z]{8})", re.I)
GAME_URL_RE  = re.compile(r"/matches/([0-9a-z]{8})", re.I)
GAME_RE      = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<home>.+?)\s*[-–—]\s*(?P<away>.+)")

MASTER_PLAYER_JSON = "master_players.json"
MASTER_TEAM_JSON   = "master_teams.json"
LEAGUE_MP_JSON     = "master_players.json"
LEAGUE_MT_JSON     = "master_teams.json"
SEASON_PLAYER_JSON = "season_players.json"
LOOKUP_PLAYER_JSON = "_id_lookup_players.json"
LOOKUP_TEAM_JSON   = "_id_lookup_teams.json"

lock = threading.Lock()

# ────────── tiny helpers ──────────
def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "")).strip().lower()

def load_json(p: Path) -> dict:
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            logging.warning("Cannot parse %s – starting blank", p)
    return {}

def save_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), "utf-8")

def _load_map(p: Path) -> dict:
    raw = json.loads(p.read_text("utf-8"))

    mapping = {k.strip().upper(): v.strip().upper() for k, v in raw.items()}
    mapping |= {k.strip().lower(): v.strip().upper() for k, v in raw.items()}

    # alias every short code to itself (both cases)
    mapping |= {v.strip().upper(): v.strip().upper() for v in raw.values()}
    mapping |= {v.strip().lower(): v.strip().upper() for v in raw.values()}
    return mapping


def strip_prefix(fname: str) -> str:
    for pre in PREFIXES:
        if fname.startswith(pre):
            return fname[len(pre):]
    return fname

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

def get_team_id(name: str, mt: dict, team_map: dict | None = None) -> str:
    if not name:
        return ""
    s = str(name).strip()
    # Map to canonical short code if we have a map
    if team_map:
        canon = (team_map.get(s.lower()) or team_map.get(s.upper()) or s)
    else:
        canon = s
    k = _norm_key(canon)  # <-- always 'ars', 'bha', etc.
    with lock:
        if k not in mt:
            new = secrets.token_hex(4)
            while new in mt.values():
                new = secrets.token_hex(4)
            mt[k] = new
    return mt[k]


def extract_team_slug(text: str | None) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = TEAM_SLUG_RE.search(text)
    return m.group(1).lower() if m else None

def extract_game_slug(text: str | None) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = GAME_URL_RE.search(text)
    return m.group(1).lower() if m else None

def make_game_id(date: str, home: str, away: str) -> str:
    return hashlib.blake2b(f"{date}-{home}-{away}".encode(), digest_size=4).hexdigest()

def season_key(s: str) -> int:          # '2019-20' → 2019
    return int(s.split("-")[0])

def last_fpl_pos(pid: str, mp_global: dict, curr_season: str) -> Optional[str]:
    rec = mp_global.get(pid)
    if not rec:
        return None
    past = [s for s in rec["career"] if season_key(s) < season_key(curr_season)]
    if not past:
        return None
    latest = max(past, key=season_key)
    return rec["career"][latest]["position"]

# ────────── NEW helpers ──────────
def add_opponent_ids(df: pd.DataFrame):
    if {"game_id", "team_id"} <= set(df.columns):
        pairs = df[["game_id", "team_id"]].drop_duplicates()
        gid_map = pairs.groupby("game_id")["team_id"].apply(list).to_dict()

        def _opp(r):
            teams = gid_map.get(r["game_id"], [])
            if len(teams) == 2:
                return teams[1] if r["team_id"] == teams[0] else teams[0]
            return ""
        df["opponent_id"] = df.apply(_opp, axis=1)

def inject_promoted(df: pd.DataFrame, prev_team_codes: set[str] | None):
    if "team" in df.columns and prev_team_codes is not None:
        df["is_promoted"] = df["team"].apply(
            lambda t: pd.NA if t == "" else int(t not in prev_team_codes)
        ).astype("Int8")

# ────────── relegation helper ──────────
def _write_relegations(clean_dir: Path,
                       league: str,
                       season: str,
                       relegated: set[str],
                       fill_na_zero: bool = False):
    """
    Inject `is_relegated` into team_season, team_match, player_match,
    and player_season for `season`.
    """
    for folder in ("team_season", "team_match",
                   "player_match", "player_season"):
        base = clean_dir / "fbref" / league / season / folder
        if not base.exists():
            continue
        for fp in base.glob("*.csv"):
            df = pd.read_csv(fp)
            if "team" not in df.columns:
                continue
            if "is_relegated" not in df.columns:
                df["is_relegated"] = pd.NA
            df["is_relegated"] = df["team"].apply(
                lambda t: int(t in relegated) if t else pd.NA
            )
            if fill_na_zero:
                df["is_relegated"] = df["is_relegated"].fillna(0)
            df["is_relegated"] = df["is_relegated"].astype("Int8")
            df.to_csv(fp, index=False, na_rep="")


# ────────── header helpers ──────────
def normalize(col: str) -> str:
    col = re.sub(r"(?i)^unnamed(?:_\d+)?_", "", str(col)).lower()
    col = re.sub(r"[^a-z0-9_]+", "_", col)
    return re.sub(r"__+", "_", col).strip("_")

def flatten_header(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels == 1:
        df.columns = [normalize(c) for c in df.columns]
        return df

    keep, names, used = [], [], set()
    for idx, triple in enumerate(df.columns):
        p = [str(x).strip() for x in triple]
        label = next((x for x in (p[2], p[1], p[0]) if x and not re.match(r"(?i)unnamed", x)), "")
        if not label:
            continue
        base = normalize(label)
        for cand in (base,
                     normalize(f"{p[1]}_{label}"),
                     normalize(f"{p[0]}_{base}")):
            if cand not in used:
                keep.append(idx)
                names.append(cand)
                used.add(cand)
                break
    flat = df.iloc[:, keep].copy()
    flat.columns = names
    return flat.loc[:, ~flat.columns.duplicated()]

def folder_for(stem: str) -> str:
    if stem.startswith("player_season_"):
        return "player_season"
    if stem.startswith("player_match_"):
        return "player_match"
    if stem.startswith("team_season_"):
        return "team_season"
    if stem.startswith("team_match_"):
        return "team_match"
    return "player_match"

def split_game_cols(df: pd.DataFrame, team_map: dict):
    parts = df["game"].astype(str).str.extract(GAME_RE.pattern)
    good = parts["date"].notna()
    if not good.any():
        return None
    df.loc[good, "game_date"] = parts.loc[good, "date"]
    df.loc[good, "home"] = parts.loc[good, "home"].str.strip().map(
        lambda s: team_map.get(s.lower(), team_map.get(s.upper(), s.strip())))
    df.loc[good, "away"] = parts.loc[good, "away"].str.strip().map(
        lambda s: team_map.get(s.lower(), team_map.get(s.upper(), s.strip())))
    return good

# ────────── master merge helpers ──────────
def merge_player(master, season, league, row):
    pid = row["player_id"]
    rec = master.setdefault(
        pid,
        {"name": row["name"], "nation": row.get("nation"), "born": row.get("born"), "career": {}},
    )
    rec.setdefault("nation", row.get("nation"))
    rec.setdefault("born", row.get("born"))
    rec["career"][season] = {
        "team": row.get("team"),
        "position": row.get("position"),
        "fpl_position": row.get("fpl_pos"),
        "league": league,
    }

def merge_team(master, season, league, tid, tname, pmap):
    rec = master.setdefault(tid, {"name": tname, "career": {}})
    rec["career"][season] = {
        "league": league,
        "players": [{"id": pid, "name": pmap[pid]} for pid in sorted(pmap)],
    }

# ────────── core cleaner ──────────
def clean_csv(
    path: Path,
    season: str,
    league: str,
    clean_root: Path,
    rules,
    mp_lookup,
    mt_lookup,
    mp_league,
    mt_league,
    mp_global,
    season_players,
    season_teams,
    pos_counts,
    pos_map,
    team_map,
    prev_team_codes: set[str] | None,
    force=False,
):
    stem = path.stem
    for pat, repl in rules:
        if pat.search(stem):
            stem = pat.sub(repl, stem)
            break
    out = clean_root / "fbref" / league / season / folder_for(stem) / f"{strip_prefix(stem)}.csv"
    if out.exists() and not force:
        return

    sched = path.stem.endswith("schedule")
    df = pd.read_csv(path, header=[0] if sched else [0, 1, 2])
    df = flatten_header(df)
    sub = folder_for(path.stem)

    # 1️⃣ map teams
    for col in df.columns:
        if col.lower() in TEAMLIKE:
            df[col] = df[col].astype(str).apply(
                lambda s: team_map.get(s.strip().lower(), team_map.get(s.strip().upper(), s.strip()))
            ).str.upper()

    # 2️⃣ split game + game_id
    if sub in {"team_match", "player_match"} and "game" in df.columns:
        good = split_game_cols(df, team_map)
        if good is not None and "game_id" not in df.columns:

            def gid(row):
                slug = next(
                    (
                        extract_game_slug(str(v))
                        for v in row
                        if isinstance(v, str) and extract_game_slug(str(v))
                    ),
                    None,
                )
                return slug or make_game_id(row["game_date"], row["home"], row["away"])

            df.loc[good, "game_id"] = df.loc[good].apply(gid, axis=1)

    # 3️⃣ numeric coercion
    for col in df.columns:
        if col in ID_LIKE or col in {"age", "born"} or df[col].dtype != object:
            continue
        num = pd.to_numeric(df[col].astype(str).str.replace(r"[,\u202f%]", "", regex=True), errors="coerce")
        if 0 < num.notna().sum() < len(num):
            df[col] = num

    # 4️⃣ age/born
    for col in ("age", "born"):
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.split("-", n=1)
                .str[0]
                .str.extract(r"(\d+)", expand=False)
                .astype("Int64")
            )

    # player_season extras
    if sub == "player_season":
        df["player_id"] = df["player"].apply(lambda n: get_player_id(n, mp_lookup))
        pcol = next((c for c in df.columns if c.lower().startswith("pos")), None)
        df["pri_position"] = (
            df[pcol].astype(str).str.split(",", n=1).str[0].str.strip().str.upper() if pcol else ""
        )
        df["position"] = df["pri_position"].map(pos_map).fillna("UNK")

        for _, r in df.iterrows():
            pid, team = r["player_id"], r.get("team")
            fpl_pos = last_fpl_pos(pid, mp_global, season) or r["position"]
            season_players[pid] = {
                "player_id": pid,
                "name": r["player"],
                "team": team,
                "nation": r.get("nation"),
                "born": int(r["born"]) if pd.notna(r["born"]) else None,
                "pri_position": r["pri_position"],
                "position": r["position"],
                "fpl_pos": fpl_pos,
            }
            if team:
                tid = get_team_id(team, mt_lookup, team_map)
                mt_league[_norm_key(team)] = tid
                season_teams.setdefault(tid, {"name": team, "players": {}})["players"][pid] = r["player"]
            mp_league.setdefault(_norm_key(r["player"]), pid)

        df["fpl_pos"] = df["player_id"].map(lambda pid: season_players[pid]["fpl_pos"])

    # player_match vote tally
    if sub == "player_match" and "player" in df:
        if "player_id" not in df:
            df["player_id"] = df["player"].apply(lambda n: get_player_id(n, mp_lookup))
        pcol = next((c for c in df.columns if c.lower().startswith("pos")), None)
        if pcol:
            raw = df[pcol].astype(str).str.split(",", n=1).str[0].str.strip().str.upper()
            for pid, rpos in zip(df["player_id"], raw):
                if rpos:
                    pos_counts[pid][rpos] += 1

    # ensure team_id
    if "team_id" not in df:
        url_cols = [c for c in df.columns if "url" in c.lower() or "link" in c.lower()]

        def row_tid(r):
            for c in url_cols:
                slug = extract_team_slug(r[c])
                if slug:
                    return slug
            return get_team_id(
                (r.get("team") or r.get("club") or r.get("squad") or "").strip(), mt_lookup, team_map
            )

        df["team_id"] = df.apply(row_tid, axis=1)

    # promoted flag
    inject_promoted(df, prev_team_codes)

    # KEEP/ADD this:
    if "team" in df.columns and {"home", "away"} <= set(df.columns):
        df["is_home"] = (df["team"] == df["home"]).astype("Int8")
        df["is_away"] = (df["team"] == df["away"]).astype("Int8")

    # opponent IDs
    if sub in {"team_match", "player_match"} and {"game_id", "team_id"} <= set(df.columns):
        add_opponent_ids(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, na_rep="")

# ────────── orchestrator ──────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league")
    ap.add_argument("--season")
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--rules", type=Path)
    ap.add_argument("--pos-map", type=Path, default=Path("data/config/positions.json"))
    ap.add_argument("--team-map", type=Path, default=Path("data/config/teams.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s: %(message)s"
    )

    rules = (
        [
            (re.compile(r["pattern"], re.I), r["replacement"])
            for r in json.loads(args.rules.read_text("utf-8"))
        ]
        if args.rules and args.rules.exists()
        else []
    )
    pos_map = _load_map(args.pos_map)
    team_map = _load_map(args.team_map)
    pos_counts = defaultdict(Counter)

    mp_lookup = load_json(args.clean_dir / "registry" / LOOKUP_PLAYER_JSON)
    mt_lookup = load_json(args.clean_dir / "registry" / LOOKUP_TEAM_JSON)
    mp_global = load_json(args.clean_dir / "registry" / MASTER_PLAYER_JSON)
    mt_global = load_json(args.clean_dir / "registry" / MASTER_TEAM_JSON)

    for league_dir in args.raw_dir.iterdir():
        if not league_dir.is_dir():
            continue
        league = league_dir.name
        if args.league and league != args.league:
            continue

        mp_league = load_json(args.clean_dir / "fbref" / league / "registry" / LEAGUE_MP_JSON)
        mt_league = load_json(args.clean_dir / "fbref" / league / "registry" / LEAGUE_MT_JSON)

        prev_team_codes: set[str] | None = None  # updated after each season
        prev_season_name: str | None = None

        for season_dir in sorted(league_dir.iterdir()):  # earliest → latest
            
            if not season_dir.is_dir():
                continue
            season = season_dir.name
            if args.season and season != args.season:
                continue

            snapshot = set(prev_team_codes) if prev_team_codes else set()

            ps_files = [*season_dir.rglob("*player_season_*.csv")]
            other_files = [p for p in season_dir.rglob("*.csv") if p not in ps_files]

            season_players, season_teams = {}, {}
            mapper = ThreadPoolExecutor(args.workers).map if args.workers > 1 else map

            list(
                tqdm(
                    mapper(
                        lambda f: clean_csv(
                            f,
                            season,
                            league,
                            args.clean_dir,
                            rules,
                            mp_lookup,
                            mt_lookup,
                            mp_league,
                            mt_league,
                            mp_global,
                            season_players,
                            season_teams,
                            pos_counts,
                            pos_map,
                            team_map,
                            snapshot,
                            args.force,
                        ),
                        ps_files,
                    ),
                    total=len(ps_files),
                    desc=f"{league} {season} player-season",
                )
            )
            list(
                tqdm(
                    mapper(
                        lambda f: clean_csv(
                            f,
                            season,
                            league,
                            args.clean_dir,
                            rules,
                            mp_lookup,
                            mt_lookup,
                            mp_league,
                            mt_league,
                            mp_global,
                            season_players,
                            season_teams,
                            pos_counts,
                            pos_map,
                            team_map,
                            snapshot,
                            args.force,
                        ),
                        other_files,
                    ),
                    total=len(other_files),
                    desc=f"{league} {season} other",
                )
            )

            # majority vote for positions
            for pid, rec in season_players.items():
                raw = (
                    pos_counts[pid].most_common(1)[0][0]
                    if pos_counts[pid]
                    else rec["pri_position"]
                )
                rec["pri_position"] = raw
                rec["position"] = pos_map.get(raw, "UNK")
                if last_fpl_pos(pid, mp_global, season) is None:
                    rec["fpl_pos"] = rec["position"]

            # sync positions into player_match
            pid2pos = {pid: r["position"] for pid, r in season_players.items()}
            pid2fpl = {pid: r["fpl_pos"] for pid, r in season_players.items()}
            base_dir = args.clean_dir / "fbref" / league / season
            for fp in (base_dir / "player_match").glob("*.csv"):
                df_sync = pd.read_csv(fp)
                if "player_id" in df_sync.columns:
                    df_sync["position"] = df_sync["player_id"].map(pid2pos).fillna(
                        df_sync.get("position", "UNK")
                    )
                    df_sync["fpl_pos"] = df_sync["player_id"].map(pid2fpl)
                if {"team", "home", "away"} <= set(df_sync.columns):
                    df_sync["is_home"] = (df_sync["team"] == df_sync["home"]).astype("Int8")
                    df_sync["is_away"] = (df_sync["team"] == df_sync["away"]).astype("Int8")
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

            save_json(args.clean_dir / "fbref" / league / "registry" / LEAGUE_MP_JSON, mp_league)
            save_json(args.clean_dir / "fbref" / league / "registry" / LEAGUE_MT_JSON, mt_league)

            curr_team_codes = {trec["name"] for trec in season_teams.values()}

            # <<< NEW: once we know the current season,
            # compute relegations for the PREVIOUS season and write them.
            if prev_season_name is not None and prev_team_codes is not None:
                relegated_prev = prev_team_codes - curr_team_codes
                logging.info(
                    "Marking relegations for %s %s: %s",
                    league, prev_season_name,
                    ", ".join(sorted(relegated_prev)) if relegated_prev else "(none)"
                )
                _write_relegations(args.clean_dir, league, prev_season_name, relegated_prev, fill_na_zero=False)

            


            # next season's snapshot: standardised team codes
            prev_team_codes = curr_team_codes
            prev_season_name = season

        # Ensure the latest processed season has is_relegated=0 everywhere
        if prev_season_name is not None:
            logging.info("Zero-filling is_relegated for current season %s %s", league, prev_season_name)
            _write_relegations(args.clean_dir, league, prev_season_name, set(), fill_na_zero=True)

    # save global lookups
    save_json(args.clean_dir / "registry" / LOOKUP_PLAYER_JSON, mp_lookup)
    save_json(args.clean_dir / "registry" / LOOKUP_TEAM_JSON, mt_lookup)
    save_json(args.clean_dir / "registry" / MASTER_PLAYER_JSON, mp_global)
    save_json(args.clean_dir / "registry" / MASTER_TEAM_JSON, mt_global)
    logging.info("🎉 FBref cleaning complete!")

if __name__ == "__main__":
    main()