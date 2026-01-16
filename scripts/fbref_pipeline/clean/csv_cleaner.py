#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_cleaner.py – ALL-IN-ONE cleaner (transfer-aware)

Applied changes requested:
- Player ID override system now applies to ALL occurrences (not just a single game):
  ✅ Override key uses (league, season, team_id, player_name, pos)  <-- NO game_id
  ✅ Overrides apply to every row matching that signature across the season.

Overrides file:
  data/processed/registry/overrides/player_id_overrides.json

Expected JSON list rows like:
  {
    "league": "ESP-La Liga",
    "season": "2019-2020",
    "team_id": "009187bd",
    "player": "Raúl García",
    "pos": "AM",
    "new_player_id": "deadbeef"
  }

Suggested overrides (auto-appended when conflicts detected):
  data/processed/registry/overrides/player_id_overrides_suggested.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
import threading
import secrets
import hashlib
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import date

import pandas as pd


# ────────── CONSTANTS ──────────
PREFIXES     = ("player_season_", "player_match_", "team_season_", "team_match_")
TEAMLIKE     = {"team", "club", "squad"}
OPP_LIKE     = {"opponent"}
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

KNOWN_FOLDERS = ("player_season", "player_match", "team_season", "team_match")

# overrides/audits
PLAYER_ID_OVERRIDES_REL = Path("registry") / "overrides" / "player_id_overrides.json"
PLAYER_ID_OVERRIDES_SUG_REL = Path("registry") / "overrides" / "player_id_overrides_suggested.json"
TEAM_CONFLICT_AUDIT_REL = Path("registry") / "audits" / "player_team_conflict"

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

def _json_dump_safe(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)

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

def canonical_out_stem(path: Path) -> str:
    s = path.stem
    s = strip_prefix(s)
    if s.startswith("season_"):
        return s[len("season_"):]
    if s.startswith("match_"):
        return s[len("match_"):]
    return s

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

def season_key(s: str) -> int:
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

def ensure_front_columns(df: pd.DataFrame, front: tuple[str, ...] = ("league", "season")) -> pd.DataFrame:
    cols = list(df.columns)
    front_cols = [c for c in front if c in cols]
    rest = [c for c in cols if c not in front_cols]
    return df[front_cols + rest]


# ────────── overrides (UPDATED: uses pos, not game_id) ──────────
# Key: (league, season, team_id, norm_player_name, pos_upper)
OverrideKey = Tuple[str, str, str, str, str]

def _find_pos_col(df: pd.DataFrame) -> Optional[str]:
    # FBref match tables usually include "pos" or "position" variants; you’re already using startswith("pos")
    return next((c for c in df.columns if c.lower() == "pos"), None) or \
           next((c for c in df.columns if c.lower().startswith("pos")), None)

def load_player_id_overrides(clean_dir: Path) -> Dict[OverrideKey, str]:
    fp = clean_dir / PLAYER_ID_OVERRIDES_REL
    if not fp.exists():
        return {}

    try:
        rows = json.loads(fp.read_text("utf-8"))
    except Exception as e:
        logging.warning("Failed reading overrides %s (%s) — ignoring", fp, e)
        return {}

    out: Dict[OverrideKey, str] = {}
    if not isinstance(rows, list):
        logging.warning("Overrides file %s must be a JSON list — ignoring", fp)
        return {}

    for r in rows:
        try:
            league = str(r.get("league", "")).strip()
            season = str(r.get("season", "")).strip()
            team_id = str(r.get("team_id", "")).strip()
            player = str(r.get("player", "")).strip()
            pos = str(r.get("pos", "")).strip().upper()
            new_pid = str(r.get("new_player_id", "")).strip()

            if not (league and season and team_id and player and pos and new_pid):
                continue

            key: OverrideKey = (league, season, team_id, _norm_key(player), pos)
            out[key] = new_pid
        except Exception:
            continue

    return out

def apply_player_id_overrides_df(
    df: pd.DataFrame,
    overrides: Dict[OverrideKey, str],
    *,
    league: str,
    season: str,
) -> pd.DataFrame:
    """
    Apply overrides onto df['player_id'] for ALL matching rows by (league, season, team_id, player, pos).
    Requires columns: team_id, player, pos, player_id
    """
    if not overrides:
        return df

    need = {"team_id", "player", "player_id"}
    if not need.issubset(df.columns):
        return df

    pos_col = _find_pos_col(df)
    if not pos_col:
        return df

    def _lookup(row):
        team_id = str(row["team_id"])
        player = _norm_key(str(row["player"]))
        pos = str(row[pos_col]).strip().upper()
        if not (team_id and player and pos):
            return ""
        key: OverrideKey = (league, season, team_id, player, pos)
        return overrides.get(key, "")

    new_ids = df.apply(_lookup, axis=1)
    mask = new_ids.astype(str).str.len() > 0
    if mask.any():
        before_unique = df.loc[mask, "player_id"].nunique(dropna=False)
        df.loc[mask, "player_id"] = new_ids.loc[mask]
        after_unique = df.loc[mask, "player_id"].nunique(dropna=False)
        logging.info(
            "[overrides] applied %d player_id override(s) (%d→%d unique) for %s %s",
            int(mask.sum()), int(before_unique), int(after_unique), league, season
        )
    return df


# ────────── position detail helper ──────────
def classify_match_position_detail(pos_value: str) -> str:
    if pos_value is None or str(pos_value).strip() == "":
        return "UNK"

    s = str(pos_value).upper()
    tokens = [t.strip() for t in re.split(r"[,/ ]+", s) if t.strip()]

    if any(t in {"CB", "RCB", "LCB"} for t in tokens):
        return "CB"
    if any(t in {"RWB", "LWB", "WB"} for t in tokens):
        return "WB"
    if any(t in {"RB", "LB", "FB"} for t in tokens):
        return "FB"

    if any(t in {"DM", "CDM"} for t in tokens):
        return "DM"
    if any(t in {"AM", "CAM"} for t in tokens):
        return "AM"
    if any(t in {"CM", "RCM", "LCM"} for t in tokens):
        return "CM"
    if any(t in {"LM", "RM", "WM"} for t in tokens):
        return "WM"

    if any(t in {"LW", "RW", "WF", "W"} for t in tokens):
        return "W"
    if any(t in {"ST", "CF", "SS", "FW"} for t in tokens):
        return "ST"

    if "DF" in tokens:
        return "CB"
    if "MF" in tokens:
        return "CM"

    return "UNK"


def _season_2digit_to_4digit(start_yy: int, pivot: int = 60) -> int:
    return (1900 + start_yy) if start_yy >= pivot else (2000 + start_yy)

def normalize_season_value(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()

    if re.fullmatch(r"\d+(\.0)?", s):
        s = s.split(".")[0]

    if re.fullmatch(r"\d{4}-\d{4}", s):
        return s

    m = re.fullmatch(r"(?P<y1>\d{4})[-/](?P<y2>\d{2})", s)
    if m:
        y1 = int(m.group("y1"))
        y2 = int(m.group("y2"))
        century = (y1 // 100) * 100
        end = century + y2
        if end < y1:
            end += 100
        return f"{y1:04d}-{end:04d}"

    m = re.fullmatch(r"(?P<y1>\d{4})[-/](?P<y2>\d{4})", s)
    if m:
        y1 = int(m.group("y1"))
        y2 = int(m.group("y2"))
        return f"{y1:04d}-{y2:04d}"

    m = re.fullmatch(r"(?P<y1>\d{2})(?P<y2>\d{2})", s)
    if m:
        y1 = int(m.group("y1"))
        y2 = int(m.group("y2"))
        start = _season_2digit_to_4digit(y1, pivot=60)
        end = _season_2digit_to_4digit(y2, pivot=60)
        if end < start:
            end += 100
        return f"{start:04d}-{end:04d}"

    if re.fullmatch(r"\d{4}", s):
        y = int(s)
        return f"{y:04d}-{(y+1):04d}"

    return s


# ────────── relegation helper ──────────
def _write_relegations(clean_dir: Path,
                       league: str,
                       season: str,
                       relegated: set[str],
                       fill_na_zero: bool = False):
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
    col = str(col)
    col = re.sub(r"(?i)^unnamed:\s*", "unnamed_", col)
    col = col.lower()
    col = re.sub(r"[^a-z0-9_]+", "_", col)
    col = re.sub(r"__+", "_", col).strip("_")
    return col

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

def _pick_meta_col(cols: list[str], kind: str) -> Optional[str]:
    kind = kind.lower().strip()

    rx1 = re.compile(rf"^unnamed_level_\d+_{kind}$", re.I)
    for c in cols:
        if rx1.match(c):
            return c

    c2 = f"level_1_{kind}"
    for c in cols:
        if c.lower() == c2:
            return c

    for c in cols:
        cl = c.lower()
        if cl.startswith("unnamed") and (kind in cl):
            return c

    return None

def enforce_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)

    meta_season = _pick_meta_col(cols, "season")
    meta_league = _pick_meta_col(cols, "league")

    has_meta = (meta_season in df.columns) or (meta_league in df.columns)
    if not has_meta:
        return df

    drop_cols = []
    if "season" in df.columns and meta_season and meta_season in df.columns:
        drop_cols.append("season")
    if "league" in df.columns and meta_league and meta_league in df.columns:
        drop_cols.append("league")
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    rename_map = {}
    if meta_season and meta_season in df.columns and meta_season != "season":
        rename_map[meta_season] = "season"
    if meta_league and meta_league in df.columns and meta_league != "league":
        rename_map[meta_league] = "league"
    if rename_map:
        df = df.rename(columns=rename_map)

    leftovers = []
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("season", "league"):
            continue
        if cl == "level_1_season" or cl == "level_1_league":
            leftovers.append(c)
        if cl.startswith("unnamed") and (("season" in cl) or ("league" in cl)):
            leftovers.append(c)

    if leftovers:
        seen = set()
        leftovers = [c for c in leftovers if not (c in seen or seen.add(c))]
        df = df.drop(columns=leftovers, errors="ignore")

    return df

def enforce_season_format(df: pd.DataFrame) -> pd.DataFrame:
    if "season" in df.columns:
        df["season"] = df["season"].apply(normalize_season_value)
    return df


def folder_for(stem: str) -> str:
    st = stem.lower()
    if st.startswith("player_season_"):
        return "player_season"
    if st.startswith("player_match_"):
        return "player_match"
    if st.startswith("team_season_"):
        return "team_season"
    if st.startswith("team_match_"):
        return "team_match"
    if st.startswith("season_"):
        return "player_season"
    if st.startswith("match_"):
        return "player_match"
    return "player_match"

def infer_folder(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    parts_set = set(parts)

    for f in KNOWN_FOLDERS:
        if f in parts_set:
            return f

    if "player" in parts_set:
        st = path.stem.lower()
        if st.startswith("season_"):
            return "player_season"
        if st.startswith("match_"):
            return "player_match"

    if "team" in parts_set:
        st = path.stem.lower()
        if st.startswith("season_"):
            return "team_season"
        if st.startswith("match_"):
            return "team_match"

    return folder_for(path.stem)

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


def _safe_read_fbref_csv(path: Path, is_schedule: bool) -> pd.DataFrame:
    try:
        if is_schedule:
            return pd.read_csv(path, header=0)
        try:
            return pd.read_csv(path, header=[0, 1, 2])
        except pd.errors.ParserError:
            return pd.read_csv(path, header=0)
    except pd.errors.EmptyDataError:
        logging.warning("Empty CSV (skipping): %s", path)
        return pd.DataFrame()
    except Exception as e:
        logging.warning("Failed reading CSV (skipping): %s (%s)", path, e)
        return pd.DataFrame()


# ────────── audits ──────────
def write_same_name_audit(
    clean_dir: Path,
    league: str,
    season: str,
    season_players: dict,
    *,
    out_name: str = "same_name_audit.json",
) -> None:
    by_name: Dict[str, List[str]] = defaultdict(list)
    for pid, rec in season_players.items():
        nm = rec.get("name", "") or ""
        nk = _norm_key(nm)
        if nk:
            by_name[nk].append(pid)

    dup_names = {nk: pids for nk, pids in by_name.items() if len(set(pids)) > 1}
    if not dup_names:
        return

    rows: List[dict] = []
    for nk, pids in sorted(dup_names.items(), key=lambda x: x[0]):
        for pid in sorted(set(pids)):
            rec = season_players.get(pid, {})
            rows.append({
                "league": league,
                "season": season,
                "name": rec.get("name", ""),
                "team": rec.get("team", ""),
                "pos": rec.get("pri_position", rec.get("position", "")),
                "position_detail": rec.get("position_detail", "UNK"),
                "born": rec.get("born"),
                "nation": rec.get("nation"),
                "player_id": "",
                "current_player_id": pid,
            })

    out_dir = clean_dir / "registry" / "audits" / "same_name" / league / season
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / out_name).write_text(_json_dump_safe(rows), "utf-8")
    logging.warning("[AUDIT] %d same-name rows written to %s", len(rows), out_dir / out_name)

def write_player_team_conflict_audit(
    clean_dir: Path,
    *,
    file_path: Path,
    label: str,
    key_cols: List[str],
    conflict_keys: pd.DataFrame,
    rows_df: pd.DataFrame,
    max_rows: int = 200,
) -> Path:
    sample_keys = conflict_keys.head(10)[key_cols].to_dict(orient="records")
    payload = {
        "file": str(file_path),
        "label": label,
        "key_cols": key_cols,
        "num_conflicting_keys": int(len(conflict_keys)),
        "sample_conflicting_keys": sample_keys,
        "rows": rows_df.head(max_rows).to_dict(orient="records"),
    }

    out_dir = clean_dir / TEAM_CONFLICT_AUDIT_REL / file_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_fp = out_dir / f"{file_path.stem}_conflict.json"
    audit_fp.write_text(_json_dump_safe(payload), "utf-8")
    logging.warning("[AUDIT] wrote team conflict audit: %s", audit_fp)
    return audit_fp

def append_suggested_overrides(
    clean_dir: Path,
    *,
    league: str,
    season: str,
    file_path: Path,
    conflict_rows: pd.DataFrame,
) -> Path:
    """
    UPDATED suggestion grain:
      (league, season, team_id, player, pos)   <-- NO game_id
    """
    sug_fp = clean_dir / PLAYER_ID_OVERRIDES_SUG_REL
    sug_fp.parent.mkdir(parents=True, exist_ok=True)

    existing: List[dict] = []
    if sug_fp.exists():
        try:
            raw = json.loads(sug_fp.read_text("utf-8"))
            if isinstance(raw, list):
                existing = raw
        except Exception:
            existing = []

    existing_keys = set()
    for r in existing:
        k = (
            str(r.get("league", "")),
            str(r.get("season", "")),
            str(r.get("team_id", "")),
            _norm_key(str(r.get("player", ""))),
            str(r.get("pos", "")).upper(),
        )
        existing_keys.add(k)

    pos_col = _find_pos_col(conflict_rows)
    need_cols = {"team_id", "player_id", "player"}
    if not need_cols.issubset(conflict_rows.columns) or not pos_col:
        return sug_fp

    new_rows: List[dict] = []
    for _, r in conflict_rows.iterrows():
        k = (league, season, str(r["team_id"]), _norm_key(str(r["player"])), str(r[pos_col]).strip().upper())
        if k in existing_keys:
            continue
        existing_keys.add(k)

        new_rows.append({
            "league": league,
            "season": season,
            "team_id": str(r["team_id"]),
            "player": str(r["player"]),
            "pos": str(r[pos_col]).strip().upper(),
            "old_player_id": str(r["player_id"]),
            "new_player_id": "",   # YOU FILL THIS
            "source_file": str(file_path),
        })

    if new_rows:
        merged = existing + new_rows
        sug_fp.write_text(_json_dump_safe(merged), "utf-8")
        logging.warning("[SUGGEST] appended %d suggested override row(s) to %s", len(new_rows), sug_fp)

    return sug_fp


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
        "position_detail": row.get("position_detail"),
        "league": league,
    }
    if isinstance(row.get("teams"), list) and len(row.get("teams", [])) > 1:
        season_entry["teams"] = row["teams"]

    rec["career"][season] = season_entry

def merge_team(master, season, league, tid, tname, pmap):
    rec = master.setdefault(tid, {"name": tname, "career": {}})
    rec["career"][season] = {
        "league": league,
        "players": [{"id": pid, "name": pmap[pid]} for pid in sorted(pmap)],
    }


# ────────── transfer windows + as-of fill + audit ──────────
def _build_team_windows_from_cleaned_player_match(base_dir: Path) -> pd.DataFrame:
    pm_dir = base_dir / "player_match"
    frames: list[pd.DataFrame] = []

    for fp in pm_dir.glob("*.csv"):
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception as e:
            logging.warning("Window scan: failed reading %s (%s) — skipping", fp.name, e)
            continue

        if df.empty:
            continue

        need = {"player_id", "team", "team_id"}
        if not need.issubset(df.columns):
            continue

        if "game_date" in df.columns:
            dt = pd.to_datetime(df["game_date"], errors="coerce")
        elif "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
        elif "game" in df.columns:
            extracted = df["game"].astype(str).str.extract(r"(?P<d>\d{4}-\d{2}-\d{2})")["d"]
            dt = pd.to_datetime(extracted, errors="coerce")
        else:
            continue

        tmp = df[["player_id", "team", "team_id"]].copy()
        tmp["game_date"] = dt
        tmp = tmp.dropna(subset=["player_id", "team_id", "game_date"]).copy()
        if tmp.empty:
            continue

        tmp["player_id"] = tmp["player_id"].astype(str)
        tmp["team_id"] = tmp["team_id"].astype(str)
        tmp["game_date"] = pd.to_datetime(tmp["game_date"])

        frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["player_id", "team", "team_id", "first_game", "last_game"])

    pm_all = pd.concat(frames, ignore_index=True)

    spans = (
        pm_all.groupby(["player_id", "team", "team_id"], dropna=False)
              .agg(first_game=("game_date", "min"),
                   last_game=("game_date", "max"))
              .reset_index()
    )
    spans["first_game"] = spans["first_game"].dt.date
    spans["last_game"] = spans["last_game"].dt.date
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

def _fill_team_asof_missing(
    fp: Path,
    asof_map: Dict[str, List[Tuple[Optional[date], Optional[date], str, str]]],
    *,
    clean_dir: Path,
    league: str,
    season: str,
):
    df = pd.read_csv(fp, low_memory=False)
    if df.empty or "player_id" not in df.columns or "game_date" not in df.columns:
        return

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
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

    if has_tid:
        if "game_id" in df.columns and df["game_id"].notna().any():
            key_cols = ["player_id", "game_id"]
            drop_cols = ["player_id", "team_id", "game_id"]
            label = "(player_id,game_id)"
        else:
            key_cols = ["player_id", "game_date"]
            drop_cols = ["player_id", "team_id", "game_date"]
            label = "(player_id,date)"

        g = (df.dropna(subset=drop_cols)
               .groupby(key_cols)["team_id"].nunique())
        bad = g[g > 1]

        if len(bad):
            bad_keys_df = bad.reset_index()[key_cols]
            rows_df = df.merge(bad_keys_df, on=key_cols, how="inner")

            write_player_team_conflict_audit(
                clean_dir,
                file_path=fp,
                label=label,
                key_cols=key_cols,
                conflict_keys=bad.reset_index(),
                rows_df=rows_df,
            )
            append_suggested_overrides(
                clean_dir,
                league=league,
                season=season,
                file_path=fp,
                conflict_rows=rows_df,
            )

            logging.warning(
                "[CONTINUE] Detected %d conflicting %s keys in %s. "
                "Wrote audits + suggestions; continuing without crash.",
                int(len(bad)), label, fp.name
            )

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
    pos_detail_counts,
    pos_map,
    team_map,
    prev_team_codes: set[str] | None,
    player_id_overrides: Dict[OverrideKey, str],
    force: bool = False,
    forced_sub: str | None = None,
):
    stem = path.stem
    for pat, repl in rules:
        if pat.search(stem):
            stem = pat.sub(repl, stem)
            break

    sub = forced_sub or infer_folder(path)

    out_name = canonical_out_stem(path) + ".csv"
    out = clean_root / "fbref" / league / season / sub / out_name

    if out.exists() and not force:
        return

    sched = path.stem.endswith("schedule")
    df = _safe_read_fbref_csv(path, sched)
    if df.empty:
        return

    df = flatten_header(df)
    df = enforce_meta_columns(df)
    df = ensure_front_columns(df, ("league", "season"))
    df = enforce_season_format(df)
    df = ensure_front_columns(df, ("league", "season"))

    # round filter
    if "round" in df.columns:
        before = len(df)
        mw = df["round"].astype(str).str.contains(r"(?i)\bmatchweek\b", na=False)
        df = df.loc[mw].copy()
        dropped = before - len(df)
        if dropped:
            logging.info("Dropped %d row(s) where round != Matchweek in %s", dropped, path.name)
        if df.empty:
            return

    # team mapping (team-like + opponent)
    for col in df.columns:
        col_l = col.lower()
        if (col_l in TEAMLIKE) or (col_l in OPP_LIKE):
            df[col] = df[col].astype(str).apply(
                lambda s: team_map.get(s.strip().lower(), team_map.get(s.strip().upper(), s.strip()))
            ).str.upper()

    # split game + game_id
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

    # numeric coercion
    for col in df.columns:
        if col in ID_LIKE or col in {"age", "born"} or df[col].dtype != object:
            continue
        num = pd.to_numeric(df[col].astype(str).str.replace(r"[,\u202f%]", "", regex=True), errors="coerce")
        if 0 < num.notna().sum() < len(num):
            df[col] = num

    # age/born
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
        if "player" in df.columns:
            df["player_id"] = df["player"].apply(lambda n: get_player_id(n, mp_lookup))
        else:
            df["player_id"] = ""

        pcol = next((c for c in df.columns if c.lower().startswith("pos")), None)
        df["pri_position"] = (
            df[pcol].astype(str).str.split(",", n=1).str[0].str.strip().str.upper() if pcol else ""
        )
        df["position"] = df["pri_position"].map(pos_map).fillna("UNK")

        if "position_detail" not in df.columns:
            df["position_detail"] = "UNK"

        for _, r in df.iterrows():
            pid, team = r.get("player_id", ""), r.get("team")
            fpl_pos = last_fpl_pos(pid, mp_global, season) or r.get("position", "UNK")

            season_players[pid] = {
                "player_id": pid,
                "name": r.get("player", ""),
                "team": team,
                "team_id": "",
                "nation": r.get("nation"),
                "born": int(r["born"]) if ("born" in r and pd.notna(r["born"])) else None,
                "pri_position": r.get("pri_position", ""),
                "position": r.get("position", "UNK"),
                "fpl_pos": fpl_pos,
                "position_detail": "UNK",
            }
            if team:
                tid = get_team_id(team, mt_lookup, team_map)
                season_players[pid]["team_id"] = tid
                mt_league[_norm_key(team)] = tid
                season_teams.setdefault(tid, {"name": team, "players": {}})["players"][pid] = r.get("player", "")
            if r.get("player"):
                mp_league.setdefault(_norm_key(r["player"]), pid)

    # player_match vote tally + match-level detail
    if sub == "player_match" and "player" in df.columns:
        if "player_id" not in df.columns:
            df["player_id"] = df["player"].apply(lambda n: get_player_id(n, mp_lookup))

        pcol = _find_pos_col(df)
        if pcol:
            raw = df[pcol].astype(str).str.split(",", n=1).str[0].str.strip().str.upper()
            for pid, rpos in zip(df["player_id"], raw):
                if rpos:
                    pos_counts[pid][rpos] += 1

            df["position_detail_match"] = df[pcol].apply(classify_match_position_detail)
            for pid, det in zip(df["player_id"], df["position_detail_match"]):
                if det and det != "UNK":
                    pos_detail_counts[pid][det] += 1
        else:
            if "position_detail_match" not in df.columns:
                df["position_detail_match"] = "UNK"

    # ensure team_id
    if "team_id" not in df.columns:
        url_cols = [c for c in df.columns if "url" in c.lower() or "link" in c.lower()]

        def row_tid(r):
            for c in url_cols:
                slug = extract_team_slug(r.get(c))
                if slug:
                    return slug
            return get_team_id(
                (r.get("team") or r.get("club") or r.get("squad") or "").strip(),
                mt_lookup,
                team_map
            )

        df["team_id"] = df.apply(row_tid, axis=1)

    # ✅ APPLY PLAYER ID OVERRIDES (league, season, team_id, player, pos) across ALL matches
    if sub == "player_match":
        df = apply_player_id_overrides_df(df, player_id_overrides, league=league, season=season)

    # promoted flag
    if "team" in df.columns and prev_team_codes is not None:
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


# ────────── consolidated CSVs across leagues (per-season) ──────────
def _consolidate_all_leagues(
    clean_dir: Path,
    leagues: list[str],
    seasons: list[str],
    *,
    out_league_name: str = "consolidate",
    folders: tuple[str, ...] = ("player_match", "team_match", "player_season", "team_season"),
    add_source_paths: bool = False,
) -> None:
    if not leagues or not seasons:
        logging.warning("No leagues/seasons to consolidate.")
        return

    for season in seasons:
        for folder in folders:
            all_files: set[str] = set()
            for lg in leagues:
                base = clean_dir / "fbref" / lg / season / folder
                if base.exists():
                    all_files |= {p.name for p in base.glob("*.csv")}

            if not all_files:
                continue

            out_base = clean_dir / "fbref" / out_league_name / season / folder
            out_base.mkdir(parents=True, exist_ok=True)

            for fname in sorted(all_files):
                frames: list[pd.DataFrame] = []

                for lg in leagues:
                    fp = clean_dir / "fbref" / lg / season / folder / fname
                    if not fp.exists():
                        continue

                    try:
                        df = pd.read_csv(fp, low_memory=False)
                    except Exception as e:
                        logging.warning("Failed reading %s: %s", fp, e)
                        continue

                    if df.empty:
                        continue

                    df = enforce_meta_columns(df)
                    df = ensure_front_columns(df, ("league", "season"))

                    df = enforce_season_format(df)
                    df = ensure_front_columns(df, ("league", "season"))

                    if "league" not in df.columns:
                        df.insert(0, "league", lg)
                    if "season" not in df.columns:
                        insert_at = 1 if "league" in df.columns else 0
                        df.insert(insert_at, "season", season)

                    if add_source_paths and "source_file" not in df.columns:
                        insert_at = 2 if {"league", "season"} <= set(df.columns) else 0
                        df.insert(insert_at, "source_file", str(fp))

                    frames.append(df)

                if not frames:
                    continue

                out_df = pd.concat(frames, ignore_index=True, sort=False)
                out_fp = out_base / fname
                out_df.to_csv(out_fp, index=False, na_rep="")


# ────────── consolidated CSVs per league (across seasons) ──────────
def _consolidate_one_league_across_seasons(
    clean_dir: Path,
    league: str,
    seasons: list[str],
    *,
    out_folder_name: str = "consolidated",
    folders: tuple[str, ...] = ("player_match", "team_match", "player_season", "team_season"),
    add_source_paths: bool = False,
) -> None:
    if not seasons:
        logging.warning("[%s] No seasons to consolidate.", league)
        return

    for folder in folders:
        all_files: set[str] = set()
        for season in seasons:
            base = clean_dir / "fbref" / league / season / folder
            if base.exists():
                all_files |= {p.name for p in base.glob("*.csv")}

        if not all_files:
            continue

        out_base = clean_dir / "fbref" / league / out_folder_name / folder
        out_base.mkdir(parents=True, exist_ok=True)

        for fname in sorted(all_files):
            frames: list[pd.DataFrame] = []

            for season in seasons:
                fp = clean_dir / "fbref" / league / season / folder / fname
                if not fp.exists():
                    continue

                try:
                    df = pd.read_csv(fp, low_memory=False)
                except Exception as e:
                    logging.warning("[%s] Failed reading %s: %s", league, fp, e)
                    continue

                if df.empty:
                    continue

                df = enforce_meta_columns(df)
                df = ensure_front_columns(df, ("league", "season"))

                df = enforce_season_format(df)
                df = ensure_front_columns(df, ("league", "season"))

                if "league" not in df.columns:
                    df.insert(0, "league", league)
                if "season" not in df.columns:
                    insert_at = 1 if "league" in df.columns else 0
                    df.insert(insert_at, "season", season)

                if add_source_paths and "source_file" not in df.columns:
                    insert_at = 2 if {"league", "season"} <= set(df.columns) else 0
                    df.insert(insert_at, "source_file", str(fp))

                frames.append(df)

            if not frames:
                continue

            out_df = pd.concat(frames, ignore_index=True, sort=False)

            preferred = [c for c in ["season", "game_date", "date", "round", "team_id", "player_id", "game_id"]
                         if c in out_df.columns]
            if preferred:
                out_df = out_df.sort_values(preferred).reset_index(drop=True)

            out_df.to_csv(out_base / fname, index=False, na_rep="")


# ────────── orchestrator ──────────
def main():
    ap = argparse.ArgumentParser()

    mx = ap.add_mutually_exclusive_group(required=False)
    mx.add_argument("--league", help="Clean one league only (folder name under raw fbref root).")
    mx.add_argument("--all-known-leagues", action="store_true",
                    help="Clean all leagues found under raw fbref root (explicit).")

    ap.add_argument("--season")
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--rules", type=Path)
    ap.add_argument("--pos-map", type=Path, default=Path("data/config/positions.json"))
    ap.add_argument("--team-map", type=Path, default=Path("data/config/teams.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")

    ap.add_argument("--write-consolidated", action="store_true",
                    help="Write consolidated CSVs across leagues into fbref/consolidate/<season>/...")

    ap.add_argument("--write-league-consolidated", action="store_true",
                    help="Write per-league consolidated CSVs stacking seasons into fbref/<league>/consolidated/...")

    args = ap.parse_args()

    if not args.league and not args.all_known_leagues:
        ap.error("You must pass either --league <name> or --all-known-leagues")

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s: %(message)s"
    )

    raw_root = args.raw_dir
    if (args.raw_dir / "fbref").exists():
        raw_root = args.raw_dir / "fbref"

    rules = []
    if args.rules and args.rules.exists():
        raw_rules = json.loads(args.rules.read_text("utf-8"))
        for rule in raw_rules:
            rules.append((re.compile(rule["pattern"], re.I), rule["replacement"]))

    pos_map = _load_map(args.pos_map)
    team_map = _load_map(args.team_map)
    pos_counts = defaultdict(Counter)
    pos_detail_counts = defaultdict(Counter)

    mp_lookup = load_json(args.clean_dir / "registry" / LOOKUP_PLAYER_JSON)
    mt_lookup = load_json(args.clean_dir / "registry" / LOOKUP_TEAM_JSON)
    mp_global = load_json(args.clean_dir / "registry" / MASTER_PLAYER_JSON)
    mt_global = load_json(args.clean_dir / "registry" / MASTER_TEAM_JSON)

    # Load overrides once
    player_id_overrides = load_player_id_overrides(args.clean_dir)
    logging.info("[overrides] loaded %d player_id override(s)", len(player_id_overrides))

    processed_leagues: list[str] = []
    processed_seasons: set[str] = set()
    processed_seasons_by_league: dict[str, set[str]] = defaultdict(set)

    for league_dir in raw_root.iterdir():
        if not league_dir.is_dir():
            continue
        league = league_dir.name

        if args.league and league != args.league:
            continue

        processed_leagues.append(league)
        logging.info("Scanning league: %s (root: %s)", league, league_dir)

        mp_league = load_json(args.clean_dir / "fbref" / league / "registry" / LEAGUE_MP_JSON)
        mt_league = load_json(args.clean_dir / "fbref" / league / "registry" / LEAGUE_MT_JSON)

        prev_team_codes: set[str] | None = None
        prev_season_name: str | None = None

        for season_dir in sorted(league_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            season = season_dir.name
            if args.season and season != args.season:
                continue

            processed_seasons.add(season)
            processed_seasons_by_league[league].add(season)

            ps_files = [*season_dir.rglob("*player_season_*.csv")]

            player_dir = season_dir / "player"
            raw_player_season = []
            if player_dir.exists() and player_dir.is_dir():
                raw_player_season = [*player_dir.glob("season_*.csv")]

            chosen_ps = sorted(set(ps_files) | set(raw_player_season))
            other_files = [p for p in season_dir.rglob("*.csv") if p not in set(chosen_ps)]

            logging.info(
                "[%s %s] found %d player-season CSVs (+%d raw player/season_*), %d other CSVs",
                league, season, len(ps_files), len(raw_player_season), len(other_files)
            )

            if not chosen_ps and not other_files:
                logging.warning("[%s %s] no CSVs found under %s", league, season, season_dir)
                continue

            season_players, season_teams = {}, {}

            mapper = ThreadPoolExecutor(args.workers).map if args.workers > 1 else map
            prev_codes_for_clean = prev_team_codes if prev_team_codes is not None else None

            list(mapper(
                lambda f: clean_csv(
                    f, season, league, args.clean_dir, rules,
                    mp_lookup, mt_lookup, mp_league, mt_league, mp_global,
                    season_players, season_teams, pos_counts, pos_detail_counts,
                    pos_map, team_map, prev_codes_for_clean,
                    player_id_overrides,
                    args.force,
                    forced_sub="player_season",
                ),
                chosen_ps,
            ))

            list(mapper(
                lambda f: clean_csv(
                    f, season, league, args.clean_dir, rules,
                    mp_lookup, mt_lookup, mp_league, mt_league, mp_global,
                    season_players, season_teams, pos_counts, pos_detail_counts,
                    pos_map, team_map, prev_codes_for_clean,
                    player_id_overrides,
                    args.force,
                    forced_sub=None,
                ),
                other_files,
            ))

            # finalize positions + season-level position_detail (mode by appearances)
            for pid, rec in season_players.items():
                raw = pos_counts[pid].most_common(1)[0][0] if pos_counts[pid] else rec["pri_position"]
                rec["pri_position"] = raw
                rec["position"] = pos_map.get(raw, "UNK")
                if last_fpl_pos(pid, mp_global, season) is None:
                    rec["fpl_pos"] = rec["position"]

                rec["position_detail"] = (
                    pos_detail_counts[pid].most_common(1)[0][0]
                    if pos_detail_counts[pid] else "UNK"
                )

            write_same_name_audit(args.clean_dir, league, season, season_players)

            base_dir = args.clean_dir / "fbref" / league / season
            spans = _build_team_windows_from_cleaned_player_match(base_dir)
            asof_map = _build_asof_map(spans) if not spans.empty else {}

            pid2pos = {pid: r["position"] for pid, r in season_players.items()}
            pid2fpl = {pid: r["fpl_pos"] for pid, r in season_players.items()}
            pid2detail = {pid: r.get("position_detail", "UNK") for pid, r in season_players.items()}

            # sync player_match: position / fpl_pos / season position_detail
            for fp in (base_dir / "player_match").glob("*.csv"):
                df_sync = pd.read_csv(fp, low_memory=False)
                if "player_id" in df_sync.columns:
                    df_sync["position"] = df_sync["player_id"].map(pid2pos).fillna(df_sync.get("position", "UNK"))
                    df_sync["fpl_pos"]  = df_sync["player_id"].map(pid2fpl)
                    df_sync["position_detail"] = df_sync["player_id"].map(pid2detail).fillna("UNK")

                if {"team", "home", "away"} <= set(df_sync.columns):
                    df_sync["is_home"] = (df_sync["team"] == df_sync["home"]).astype("Int8")
                    df_sync["is_away"] = (df_sync["team"] == df_sync["away"]).astype("Int8")

                df_sync.to_csv(fp, index=False, na_rep="")
                if asof_map:
                    _fill_team_asof_missing(
                        fp, asof_map,
                        clean_dir=args.clean_dir,
                        league=league,
                        season=season
                    )

            # sync player_season CSVs so they also get season-level position_detail
            for fp in (base_dir / "player_season").glob("*.csv"):
                df_ps = pd.read_csv(fp, low_memory=False)
                if "player_id" in df_ps.columns:
                    df_ps["position_detail"] = df_ps["player_id"].map(pid2detail).fillna("UNK")
                df_ps.to_csv(fp, index=False, na_rep="")

            # update season_players with transfer windows
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
                logging.info(
                    "Marking relegations for %s %s: %s",
                    league, prev_season_name,
                    ", ".join(sorted(relegated_prev)) if relegated_prev else "(none)"
                )
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

    if args.write_consolidated:
        seasons_list = sorted(processed_seasons)
        leagues_list = processed_leagues
        logging.info(
            "Writing consolidated CSVs for %d league(s) across %d season(s) into fbref/consolidate/<season>/...",
            len(leagues_list), len(seasons_list)
        )
        _consolidate_all_leagues(args.clean_dir, leagues_list, seasons_list)

    if args.write_league_consolidated:
        for lg in processed_leagues:
            seasons_list = sorted(processed_seasons_by_league.get(lg, set()))
            logging.info(
                "[%s] Writing league consolidated CSVs across %d season(s) into fbref/%s/consolidated/...",
                lg, len(seasons_list), lg
            )
            _consolidate_one_league_across_seasons(
                args.clean_dir,
                league=lg,
                seasons=seasons_list,
            )

    logging.info("🎉 FBref cleaning complete (transfer-aware registries, historical rows preserved).")


if __name__ == "__main__":
    main()
