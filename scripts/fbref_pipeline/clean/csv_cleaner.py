#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_fbref_csvs.py – ALL-IN-ONE cleaner (transfer-aware)     2025-09-18  rev N+transfer

Adds (non-breaking):
• Per-season team windows from player_match → registry JSON sets legacy `team`/`team_id` to the
  *latest* team; emits `teams` array only if a player changed teams (A→B→…).
• As-of filler for per-row CSVs: ONLY fills missing `team`/`team_id` using the windows; never overwrites.
• Audit to catch any (player_id, date) rows that would claim >1 team_id.

Everything else is preserved from your rev N.
"""

from __future__ import annotations
import argparse, json, logging, re, unicodedata, threading, secrets, hashlib
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import date

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
    canon = (team_map.get(s.lower()) or team_map.get(s.upper()) or s) if team_map else s
    k = _norm_key(canon)
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

def make_game_id(date_s: str, home: str, away: str) -> str:
    return hashlib.blake2b(f"{date_s}-{home}-{away}".encode(), digest_size=4).hexdigest()

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

# ────────── NEW: relegation helper ──────────
def _write_relegations(clean_dir: Path,
                       league: str,
                       season: str,
                       relegated: set[str],
                       fill_na_zero: bool = False):
    """
    Inject `is_relegated` into team_season, team_match, player_match,
    and player_season for `season`.
    """
    for folder in ("team_season", "team_match", "player_match", "player_season"):
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
        for cand in (base, normalize(f"{p[1]}_{label}"), normalize(f"{p[0]}_{base}")):
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

    season_entry = {
        "team": row.get("team"),
        "team_id": row.get("team_id"),
        "position": row.get("position"),
        "fpl_position": row.get("fpl_pos"),
        "league": league,
    }
    # Only attach windows if a transfer occurred
    if isinstance(row.get("teams"), list) and len(row["teams"]) > 1:
        season_entry["teams"] = row["teams"]

    rec["career"][season] = season_entry

def merge_team(master, season, league, tid, tname, pmap):
    rec = master.setdefault(tid, {"name": tname, "career": {}})
    rec["career"][season] = {
        "league": league,
        "players": [{"id": pid, "name": pmap[pid]} for pid in sorted(pmap)],
    }

# ────────── NEW: transfer windows + as-of fill + audit ──────────
def _build_team_windows_from_cleaned_player_match(base_dir: Path) -> pd.DataFrame:
    """
    Reads cleaned player_match CSVs in <base_dir>/player_match/*.csv and returns spans:
    [player_id, team, team_id, first_game(date), last_game(date)]
    """
    pm_dir = base_dir / "player_match"
    frames = []
    for fp in pm_dir.glob("*.csv"):
        df = pd.read_csv(fp, parse_dates=["game_date"], low_memory=False)
        need = {"player_id", "team", "team_id", "game_date"}
        if need.issubset(df.columns) and not df.empty:
            frames.append(df[list(need)].copy())
    if not frames:
        return pd.DataFrame(columns=["player_id", "team", "team_id", "first_game", "last_game"])

    pm_all = pd.concat(frames, ignore_index=True)
    pm_all = pm_all.dropna(subset=["player_id", "team_id", "game_date"]).copy()
    pm_all["player_id"] = pm_all["player_id"].astype(str)
    pm_all["team_id"]   = pm_all["team_id"].astype(str)
    pm_all["game_date"] = pd.to_datetime(pm_all["game_date"])

    spans = (
        pm_all.groupby(["player_id", "team", "team_id"], dropna=False)
              .agg(first_game=("game_date", "min"),
                   last_game=("game_date", "max"))
              .reset_index()
    )
    spans["first_game"] = spans["first_game"].dt.date
    spans["last_game"]  = spans["last_game"].dt.date
    return spans

def _choose_latest_team_row(spans_for_player: pd.DataFrame) -> pd.Series:
    g = spans_for_player.sort_values(["last_game", "first_game"], ascending=[False, False])
    return g.iloc[0]

def _windows_list(spans_for_player: pd.DataFrame) -> List[Dict[str, Optional[str]]]:
    out = []
    g = spans_for_player.sort_values("first_game")
    for _, r in g.iterrows():
        out.append({
            "team":   ("" if pd.isna(r["team"]) else str(r["team"])),
            "team_id":("" if pd.isna(r["team_id"]) else str(r["team_id"])),
            "from":   (None if pd.isna(r["first_game"]) else r["first_game"].isoformat()),
            "to":     (None if pd.isna(r["last_game"])  else r["last_game"].isoformat()),
        })
    return out

def _build_asof_map(spans: pd.DataFrame) -> Dict[str, List[Tuple[Optional[date], Optional[date], str, str]]]:
    m: Dict[str, List[Tuple[Optional[date], Optional[date], str, str]]] = {}
    for pid, grp in spans.groupby("player_id"):
        rows = []
        for _, r in grp.iterrows():
            rows.append((r["first_game"], r["last_game"], str(r["team_id"]), str(r["team"])))
        m[str(pid)] = sorted(rows, key=lambda t: (t[0] or date.min))
    return m

def _fill_team_asof_missing(fp: Path, asof_map: Dict[str, List[Tuple[Optional[date], Optional[date], str, str]]]):
    df = pd.read_csv(fp, low_memory=False)
    if df.empty or "player_id" not in df.columns or "game_date" not in df.columns:
        return
    # normalize date
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    has_tid = "team_id" in df.columns
    has_team = "team" in df.columns
    if not (has_tid or has_team):
        return

    def is_missing(x) -> bool:
        return pd.isna(x) or str(x) == ""

    idx = df.index[(~df["player_id"].isna()) & (
        (has_tid and df["team_id"].apply(is_missing)) |
        (has_team and df["team"].apply(is_missing))
    )]

    for i in idx:
        pid = str(df.at[i, "player_id"])
        dt  = df.at[i, "game_date"]
        wins = asof_map.get(pid, [])
        fill_tid, fill_team = None, None
        for lo, hi, tid, tm in wins:
            if (lo is None or dt >= lo) and (hi is None or dt <= hi):
                fill_tid, fill_team = tid, tm
        if fill_tid or fill_team:
            if has_tid and is_missing(df.at[i, "team_id"]):
                df.at[i, "team_id"] = fill_tid if fill_tid is not None else df.at[i, "team_id"]
            if has_team and is_missing(df.at[i, "team"]):
                df.at[i, "team"] = fill_team if fill_team is not None else df.at[i, "team"]

    # audit: each (player_id, date) must have exactly one team_id
    if has_tid:
        g = (df.dropna(subset=["player_id", "team_id", "game_date"])
               .groupby(["player_id", "game_date"])["team_id"].nunique())
        bad = g[g > 1]
        if len(bad):
            examples = bad.head(5).to_dict()
            raise AssertionError(f"[player_match audit] Multiple team_ids for same (player_id,date) in {fp.name}: {examples}")

    df.to_csv(fp, index=False, na_rep="")

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
                "team_id": "",  # will be set from windows if available later
                "nation": r.get("nation"),
                "born": int(r["born"]) if pd.notna(r["born"]) else None,
                "pri_position": r["pri_position"],
                "position": r["position"],
                "fpl_pos": fpl_pos,
            }
            if team:
                tid = get_team_id(team, mt_lookup, team_map)
                # remember a team_id candidate; final latest will come from windows
                season_players[pid]["team_id"] = tid
                # build team roster map for this season
                mt_league[_norm_key(team)] = tid
                season_teams.setdefault(tid, {"name": team, "players": {}})["players"][pid] = r["player"]
            mp_league.setdefault(_norm_key(r["player"]), pid)

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
            return get_team_id((r.get("team") or r.get("club") or r.get("squad") or "").strip(),
                               mt_lookup, team_map)
        df["team_id"] = df.apply(row_tid, axis=1)

    # promoted flag
    if "team" in df.columns:
        # prev_team_codes is set of standardised team codes from previous season
        if prev_team_codes is not None:
            df["is_promoted"] = df["team"].apply(
                lambda t: pd.NA if t == "" else int(t not in prev_team_codes)
            ).astype("Int8")

    # venue flags
    if "team" in df.columns and {"home", "away"} <= set(df.columns):
        df["is_home"] = (df["team"] == df["home"]).astype("Int8")
        df["is_away"] = (df["team"] == df["away"]).astype("Int8")

    # opponent IDs
    if sub in {"team_match", "player_match"} and {"game_id", "team_id"} <= set(df.columns):
        pairs = df[["game_id", "team_id"]].drop_duplicates()
        gid_map = pairs.groupby("game_id")["team_id"].apply(list).to_dict()
        def _opp(r):
            teams = gid_map.get(r["game_id"], [])
            if len(teams) == 2:
                return teams[1] if r["team_id"] == teams[0] else teams[0]
            return ""
        df["opponent_id"] = df.apply(_opp, axis=1)

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

    # -------- NEW: resolve raw root so both layouts work --------
    raw_root = args.raw_dir
    if (args.raw_dir / "fbref").exists():
        raw_root = args.raw_dir / "fbref"   # <-- your layout

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

    # iterate leagues under resolved raw_root
    for league_dir in raw_root.iterdir():
        if not league_dir.is_dir():
            continue
        league = league_dir.name
        if args.league and league != args.league:
            continue

        logging.info("Scanning league: %s (root: %s)", league, league_dir)

        mp_league = load_json(args.clean_dir / "fbref" / league / "registry" / LEAGUE_MP_JSON)
        mt_league = load_json(args.clean_dir / "fbref" / league / "registry" / LEAGUE_MT_JSON)

        prev_team_codes: set[str] | None = None
        prev_season_name: str | None = None

        # seasons are subfolders under league_dir
        for season_dir in sorted(league_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            season = season_dir.name
            if args.season and season != args.season:
                continue

            # discover CSVs
            ps_files = [*season_dir.rglob("*player_season_*.csv")]
            other_files = [p for p in season_dir.rglob("*.csv") if p not in ps_files]

            logging.info(
                "[%s %s] found %d player-season CSVs, %d other CSVs",
                league, season, len(ps_files), len(other_files)
            )

            if not ps_files and not other_files:
                logging.warning("[%s %s] no CSVs found under %s", league, season, season_dir)
                continue

            # ---------- unchanged cleaning / writing below ----------
            season_players, season_teams = {}, {}
            mapper = ThreadPoolExecutor(args.workers).map if args.workers > 1 else map

            # 1) player-season
            list(mapper(
                lambda f: clean_csv(
                    f, season, league, args.clean_dir, rules,
                    mp_lookup, mt_lookup, mp_league, mt_league, mp_global,
                    season_players, season_teams, pos_counts,
                    pos_map, team_map, set(prev_team_codes) if prev_team_codes else set(),
                    args.force,
                ),
                ps_files,
            ))
            # 2) others (team_match, player_match, team_season, schedules, etc.)
            list(mapper(
                lambda f: clean_csv(
                    f, season, league, args.clean_dir, rules,
                    mp_lookup, mt_lookup, mp_league, mt_league, mp_global,
                    season_players, season_teams, pos_counts,
                    pos_map, team_map, set(prev_team_codes) if prev_team_codes else set(),
                    args.force,
                ),
                other_files,
            ))

            # 3) majority vote positions (unchanged)
            for pid, rec in season_players.items():
                raw = pos_counts[pid].most_common(1)[0][0] if pos_counts[pid] else rec["pri_position"]
                rec["pri_position"] = raw
                rec["position"] = pos_map.get(raw, "UNK")
                if last_fpl_pos(pid, mp_global, season) is None:
                    rec["fpl_pos"] = rec["position"]

            # 4) sync positions + as-of fill + audit (unchanged from previous message)
            base_dir = args.clean_dir / "fbref" / league / season
            spans = _build_team_windows_from_cleaned_player_match(base_dir)
            asof_map = _build_asof_map(spans) if not spans.empty else {}

            pid2pos = {pid: r["position"] for pid, r in season_players.items()}
            pid2fpl = {pid: r["fpl_pos"] for pid, r in season_players.items()}

            for fp in (base_dir / "player_match").glob("*.csv"):
                df_sync = pd.read_csv(fp, low_memory=False)
                if "player_id" in df_sync.columns:
                    df_sync["position"] = df_sync["player_id"].map(pid2pos).fillna(df_sync.get("position", "UNK"))
                    df_sync["fpl_pos"]  = df_sync["player_id"].map(pid2fpl)
                if {"team", "home", "away"} <= set(df_sync.columns):
                    df_sync["is_home"] = (df_sync["team"] == df_sync["home"]).astype("Int8")
                    df_sync["is_away"] = (df_sync["team"] == df_sync["away"]).astype("Int8")
                df_sync.to_csv(fp, index=False, na_rep="")
                if asof_map:
                    _fill_team_asof_missing(fp, asof_map)

            # 5) set latest team + windows in season_players, write JSONs (unchanged)
            if not spans.empty:
                spans_map = {pid: g for pid, g in spans.groupby("player_id")}
                for pid, rec in season_players.items():
                    g = spans_map.get(pid)
                    if g is None or g.empty:
                        continue
                    latest = _choose_latest_team_row(g)
                    rec["team"] = "" if pd.isna(latest["team"]) else str(latest["team"])
                    rec["team_id"] = "" if pd.isna(latest["team_id"]) else str(latest["team_id"])
                    if g["team_id"].nunique() > 1:
                        rec["teams"] = _windows_list(g)
                    else:
                        rec.pop("teams", None)

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
            if prev_season_name is not None and prev_team_codes is not None:
                relegated_prev = prev_team_codes - curr_team_codes
                logging.info("Marking relegations for %s %s: %s",
                             league, prev_season_name,
                             ", ".join(sorted(relegated_prev)) if relegated_prev else "(none)")
                _write_relegations(args.clean_dir, league, prev_season_name, relegated_prev, fill_na_zero=False)

            prev_team_codes = curr_team_codes
            prev_season_name = season

        if prev_season_name is not None:
            logging.info("Zero-filling is_relegated for current season %s %s", league, prev_season_name)
            _write_relegations(args.clean_dir, league, prev_season_name, set(), fill_na_zero=True)

    save_json(args.clean_dir / "registry" / LOOKUP_PLAYER_JSON, mp_lookup)
    save_json(args.clean_dir / "registry" / LOOKUP_TEAM_JSON, mt_lookup)
    save_json(args.clean_dir / "registry" / MASTER_PLAYER_JSON, mp_global)
    save_json(args.clean_dir / "registry" / MASTER_TEAM_JSON, mt_global)
    logging.info("🎉 FBref cleaning complete (transfer-aware registries, historical rows preserved).")


if __name__ == "__main__":
    main()
