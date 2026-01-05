#!/usr/bin/env python3
"""
League-wide unavailability registry manager (auto metadata from master; add by player_id OR name).

What’s new (name + batch support)
---------------------------------
- On `add`, you may specify **either** --player-id or --name (mutually exclusive).
- When using --name, you must also pass --master, --teams-map, and --season so we can resolve:
    -> player_id, name, pos, team (short code), team_id (canonical)
- Optional --club (team short code, e.g. MCI) to disambiguate if multiple players match the name.
- New `add-batch` subcommand: accepts comma-separated lists for --name or --player-id (and per-row fields),
  aligned by index and padded when lists are shorter.

Other behavior (unchanged)
--------------------------
- DOUBTFUL ⇒ treated as available (not banned).
- TRANSFERRED ⇒ always unavailable until cleared.
- You can bound with `until_gw` and/or `matches_remaining` (fixture counter). Both must be active to keep OUT.
- `tick` decrements matches_remaining by 1 (drive DGWs with your external batch).
- `export-banset` writes active player_ids for the optimizer.

Assumptions on master
---------------------
- master[player_id] has a "career" dict keyed by "YYYY-YY" season.
- For that season entry: fpl_pos/position and team (short code).
- name or (first_name, second_name) present for display.

CLI examples
------------
# Add by NAME (AFCON)
python scripts/registry/league_unavailable.py add data/registry/league_unavailable.json \
  --name "Mohamed Salah" --club LIV \
  --status OUT --reason AFCON --gw-now 21 --matches 6 \
  --master data/processed/registry/master_fpl.json \
  --teams-map data/processed/registry/_id_lookup_teams.json \
  --season "2025-2026"

# Add by ID (transfer)
python scripts/registry/league_unavailable.py add data/registry/league_unavailable.json \
  --player-id 9f9f9f9f \
  --status OUT --reason TRANSFERRED --gw-now 21 \
  --master data/processed/registry/master_fpl.json \
  --teams-map data/processed/registry/_id_lookup_teams.json \
  --season "2025-2026"

# Batch add (comma-separated, aligned lists)
python scripts/registry/league_unavailable.py add-batch data/registry/league_unavailable.json \
  --name "Tyrone Mings, Youri Tielemans, Emi Buendía, Andrés García" \
  --club "AVL, AVL, AVL, AVL" \
  --status "OUT, DOUBTFUL, DOUBTFUL, DOUBTFUL" \
  --reason INJURY \
  --detail "Foot, Lower Leg, Head, UNKNOWN" \
  --source "Club statement" \
  --gw-now 8 \
  --season "2025-2026" \
  --master data/processed/registry/master_fpl.json \
  --teams-map data/processed/registry/_id_lookup_teams.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# ----- enums / validation -----
VALID_STATUS = {"OUT", "SUSPENDED", "DOUBTFUL", "RESTED", "UNKNOWN"}
VALID_REASON = {
    "INJURY", "SUSPENSION", "NATIONAL_TEAM", "AFCON", "COVID", "INTERNATIONAL_BREAK",
    "PERSONAL", "REST", "TRANSFERRING", "TRANSFERRED", "UNKNOWN"
}
VALID_POS = {"GK", "DEF", "MID", "FWD"}
CODE_RE = re.compile(r"^[A-Za-z]{2,4}$")
ALNUM_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SEASON_SHORT_RE = re.compile(r"^\d{4}-\d{2}$")

def _is_code_like(s: Optional[str]) -> bool:
    return bool(s) and CODE_RE.fullmatch(s.strip()) is not None

def _is_canonical_id(s: Optional[str]) -> bool:
    s = (s or "").strip()
    return bool(s) and ALNUM_ID_RE.fullmatch(s) is not None and not _is_code_like(s)

def _to_short_season(long_or_short: str) -> str:
    s = long_or_short.strip()
    if SEASON_SHORT_RE.fullmatch(s):
        return s
    if "/" in s:
        a, b = s.split("/", 1)
        return f"{a}-{b}"
    if len(s) == 9 and s[4] == "-":
        a = s[:4]; b = s[-2:]
        return f"{a}-{b}"
    return s

_POS_MAP = {
    "GKP": "GK", "GK": "GK", "G": "GK",
    "DEF": "DEF", "DF": "DEF", "D": "DEF",
    "MID": "MID", "MF": "MID", "M": "MID",
    "FWD": "FWD", "FW": "FWD", "F": "FWD", "ST": "FWD",
}

def _norm_pos(pos_raw: Optional[str]) -> Optional[str]:
    if pos_raw is None: return None
    pr = str(pos_raw).strip().upper()
    return _POS_MAP.get(pr, pr if pr in VALID_POS else None)

def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _norm_name(s: str) -> str:
    # accent-insensitive + casefold, collapse spaces
    return _strip_accents(" ".join(str(s).split())).casefold()

# ----- master helpers -----
def _load_json(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _load_master(path: Optional[str | Path]) -> Optional[Dict[str, dict]]:
    if not path: return None
    return _load_json(path)

def _load_teams_map(path: Optional[str | Path]) -> Optional[Dict[str, str]]:
    if not path: return None
    raw = _load_json(path)
    # Expect { "MCI": "canonical_team_id", ... }
    return {str(k).strip().upper(): str(v).strip() for k, v in raw.items()}

def _pick_career_season(entry: dict, target_short: str) -> Tuple[Optional[str], Optional[dict]]:
    career = entry.get("career", {}) or {}
    if target_short in career and isinstance(career[target_short], dict):
        return target_short, career[target_short]
    candidates = []
    for k, v in career.items():
        if isinstance(v, dict) and SEASON_SHORT_RE.fullmatch(k):
            try:
                year = int(k[:4]); candidates.append((year, k, v))
            except Exception:
                continue
    if candidates:
        candidates.sort()
        _, k, v = candidates[-1]
        return k, v
    return None, None

def _master_name(entry: dict, pid: str) -> str:
    if entry.get("name"):
        return str(entry["name"])
    fn = str(entry.get("first_name") or "").strip()
    sn = str(entry.get("second_name") or "").strip()
    full = (fn + " " + sn).strip()
    return full if full else pid

def _meta_from_master(
    master: Dict[str, dict],
    teams_map: Dict[str, str] | None,
    *,
    player_id: str,
    season_long_or_short: str,
) -> Tuple[str, str, str, str]:
    """
    Returns (name, pos, team_code, team_id)
    """
    if player_id not in master:
        raise ValueError(f"player_id {player_id} not found in master")
    entry = master[player_id]
    target_short = _to_short_season(season_long_or_short)
    season_used, info = _pick_career_season(entry, target_short)
    if not info:
        raise ValueError(
            f"Player {player_id} has no usable career season for '{target_short}'. "
            f"Available: {list((entry.get('career') or {}).keys())}"
        )
    pos = _norm_pos(info.get("fpl_pos") or info.get("position"))
    if not pos:
        raise ValueError(
            f"Unrecognized position for player {player_id} in '{season_used}'. "
            f"Found {info.get('fpl_pos') or info.get('position')!r}"
        )
    team_code = str(info.get("team") or "").upper()
    if not _is_code_like(team_code):
        raise ValueError(f"Missing/invalid team code for player {player_id} in '{season_used}'.")
    if not teams_map or team_code not in teams_map:
        raise ValueError(
            f"teams_map missing entry for team code '{team_code}' needed for player {player_id}."
        )
    team_id = teams_map[team_code]
    if not _is_canonical_id(team_id):
        raise ValueError(
            f"Resolved team_id '{team_id}' is not a canonical id (looks like a code). Fix teams_map."
        )
    name = _master_name(entry, player_id)
    return name, pos, team_code, team_id

def _resolve_player_id_by_name(
    master: Dict[str, dict],
    teams_map: Dict[str, str],
    *,
    name: str,
    season_long_or_short: str,
    club_filter: Optional[str] = None,
) -> Tuple[str, Dict[str, str]]:
    """
    Resolve a player_id from a display name (accent-insensitive, case-insensitive).
    - Prefers exact name matches; falls back to substring matches.
    - If multiple candidates, optional `club_filter` (team short code) narrows.
    Returns (player_id, metadata_dict_for_echo)
    """
    target_short = _to_short_season(season_long_or_short)
    needle = _norm_name(name)
    club_filter = club_filter.upper().strip() if club_filter else None

    exact: List[Tuple[str, str]] = []     # (pid, team_code)
    fuzzy: List[Tuple[str, str]] = []

    for pid, entry in master.items():
        disp = _master_name(entry, pid)
        dn = _norm_name(disp)
        _, info = _pick_career_season(entry, target_short)
        if not info:
            continue
        team_code = str(info.get("team") or "").upper()
        if club_filter and team_code != club_filter:
            continue
        if dn == needle:
            exact.append((pid, team_code))
        elif needle in dn:
            fuzzy.append((pid, team_code))

    cand = exact if exact else fuzzy
    if not cand:
        raise ValueError(f"No player found for name '{name}'"
                         + (f" at club '{club_filter}'" if club_filter else "")
                         + f" in season '{target_short}'.")

    if len(cand) > 1:
        options = ", ".join([f"{pid} ({tc})" for pid, tc in cand[:10]])
        raise ValueError(
            f"Ambiguous name '{name}' — {len(cand)} matches. "
            f"Hint: add --club TEAMCODE. Candidates: {options}" + ("…" if len(cand) > 10 else "")
        )

    pid, tc = cand[0]
    return pid, {"team": tc}

# ----- model -----
@dataclass
class PlayerUnavail:
    # Identity
    player_id: str
    name: Optional[str] = None
    pos: Optional[str] = None            # "GK" | "DEF" | "MID" | "FWD"
    team_id: Optional[str] = None        # canonical id
    team: Optional[str] = None           # short code

    # Status
    status: str = "UNKNOWN"              # OUT|SUSPENDED|DOUBTFUL|RESTED|UNKNOWN
    reason: str = "UNKNOWN"              # INJURY|SUSPENSION|AFCON|...|TRANSFERRED|UNKNOWN
    detail: Optional[str] = None
    source: Optional[str] = None

    # Windows/counters
    added_gw: Optional[int] = None
    last_accounted_gw: Optional[int] = None
    until_gw: Optional[int] = None
    matches_remaining: Optional[int] = None
    active: bool = True                  # True => ban is active (unavailable)

    @staticmethod
    def from_dict(d: dict) -> "PlayerUnavail":
        return PlayerUnavail(
            player_id=str(d.get("player_id")),
            name=d.get("name"),
            pos=(str(d["pos"]).upper() if d.get("pos") else None),
            team_id=(str(d["team_id"]) if d.get("team_id") is not None else None),
            team=(str(d["team"]).upper() if d.get("team") else None),
            status=str(d.get("status", "UNKNOWN")).upper(),
            reason=str(d.get("reason", "UNKNOWN")).upper(),
            detail=d.get("detail"),
            source=d.get("source"),
            added_gw=int(d["added_gw"]) if d.get("added_gw") is not None else None,
            last_accounted_gw=int(d["last_accounted_gw"]) if d.get("last_accounted_gw") is not None else None,
            until_gw=int(d["until_gw"]) if d.get("until_gw") is not None else None,
            matches_remaining=int(d["matches_remaining"]) if d.get("matches_remaining") is not None else None,
            active=bool(d.get("active", True)),
        )

    def validate_metadata(self) -> None:
        if self.pos is not None and self.pos not in VALID_POS:
            raise ValueError(f"Invalid pos '{self.pos}'. Valid: {sorted(VALID_POS)}")
        if self.team is not None and not _is_code_like(self.team):
            raise ValueError(f"team short code must be 2–4 letters (e.g., 'MCI'), got '{self.team}'")
        if self.team_id is not None and not _is_canonical_id(self.team_id):
            raise ValueError(
                f"team_id must be canonical alphanumeric (not a short code). Got '{self.team_id}'."
            )

    def to_dict(self) -> dict:
        return asdict(self)

# ----- IO -----
def _load_registry(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"meta": {"season": "", "last_updated_gw": 0, "notes": ""}, "players": []}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _save_registry(obj: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _idx(players: List[dict], pid: str) -> int:
    for i, d in enumerate(players):
        if str(d.get("player_id")) == pid:
            return i
    return -1

# ----- logic -----
def _normalize_status_reason(status: Optional[str], reason: Optional[str]) -> Tuple[str, str]:
    s = (status or "UNKNOWN").upper()
    r = (reason or "UNKNOWN").upper()
    if s not in VALID_STATUS:
        raise ValueError(f"Invalid status '{s}'. Valid: {sorted(VALID_STATUS)}")
    if r not in VALID_REASON:
        raise ValueError(f"Invalid reason '{r}'. Valid: {sorted(VALID_REASON)}")
    return s, r

def _is_active(entry: PlayerUnavail, gw_now: int) -> bool:
    # True => ban is active (player is unavailable for selection)
    if entry.reason == "TRANSFERRED":
        return True
    if entry.status == "DOUBTFUL":
        return False
    active = True
    if entry.until_gw is not None and gw_now > int(entry.until_gw):
        active = False
    if entry.matches_remaining is not None and int(entry.matches_remaining) <= 0:
        active = False
    return active

def _tick_entry(entry: PlayerUnavail, gw_now: int) -> PlayerUnavail:
    if entry.matches_remaining is not None and entry.matches_remaining > 0:
        entry.matches_remaining -= 1
    entry.last_accounted_gw = gw_now
    entry.active = _is_active(entry, gw_now)
    return entry

def _recompute_active(entry: PlayerUnavail, gw_now: int) -> PlayerUnavail:
    entry.active = _is_active(entry, gw_now)
    return entry

def _active_banset(players: List[dict], gw_now: int) -> List[str]:
    ids = []
    for d in players:
        e = PlayerUnavail.from_dict(d)
        if _is_active(e, gw_now):
            ids.append(e.player_id)
    return sorted(set(ids))

# ----- commands -----
def cmd_add(args) -> None:
    js = _load_registry(args.registry)
    players: List[dict] = js.get("players", [])
    gw_now = int(args.gw_now)
    status, reason = _normalize_status_reason(args.status, args.reason)

    # Validate param pairing
    using_pid = args.player_id is not None
    using_name = args.name is not None
    if using_pid and using_name:
        raise SystemExit("Provide either --player-id or --name (not both).")
    if not using_pid and not using_name:
        raise SystemExit("Provide --player-id OR --name.")

    # Load master & teams_map if any metadata work is needed (always for --name; optional for --player-id)
    master = None
    teams_map = None
    if using_name or args.master or args.teams_map or args.season:
        if not (args.master and args.teams_map and args.season):
            raise SystemExit("When using name resolution or metadata autofill, you must pass --master, --teams-map and --season together.")
        master = _load_master(args.master)
        teams_map = _load_teams_map(args.teams_map)
        if master is None or teams_map is None:
            raise SystemExit("Failed to load --master or --teams-map.")

    # Resolve player_id if adding by name
    player_id = args.player_id
    if using_name:
        player_id, _ = _resolve_player_id_by_name(
            master, teams_map,
            name=args.name, season_long_or_short=args.season, club_filter=args.club
        )

    # Auto metadata if master provided
    auto_name = auto_pos = auto_team = auto_team_id = None
    if master is not None:
        auto_name, auto_pos, auto_team, auto_team_id = _meta_from_master(
            master, teams_map, player_id=player_id, season_long_or_short=args.season
        )

    i = _idx(players, player_id)
    if i < 0:
        e = PlayerUnavail(
            player_id=player_id,
            name=(auto_name if auto_name is not None else None),
            pos=(auto_pos if auto_pos is not None else None),
            team_id=(auto_team_id if auto_team_id is not None else None),
            team=(auto_team if auto_team is not None else None),
            status=status,
            reason=reason,
            detail=args.detail,
            source=args.source,
            added_gw=gw_now,
            last_accounted_gw=gw_now,
            until_gw=args.until_gw,
            matches_remaining=args.matches,
            active=True,
        )
        e.validate_metadata()
        e = _recompute_active(e, gw_now)
        players.append(e.to_dict())
        msg = "added"
    else:
        cur = PlayerUnavail.from_dict(players[i])
        # refresh metadata if master provided
        if master is not None:
            cur.name = auto_name
            cur.pos = auto_pos
            cur.team = auto_team
            cur.team_id = auto_team_id
        cur.validate_metadata()
        # update status/window
        cur.status = status
        cur.reason = reason
        cur.detail = args.detail if args.detail is not None else cur.detail
        cur.source = args.source if args.source is not None else cur.source
        cur.until_gw = args.until_gw if args.until_gw is not None else cur.until_gw
        cur.matches_remaining = args.matches if args.matches is not None else cur.matches_remaining
        cur.last_accounted_gw = gw_now
        cur = _recompute_active(cur, gw_now)
        players[i] = cur.to_dict()
        msg = "updated"

    js["players"] = players
    js.setdefault("meta", {})
    if args.set_meta_season:
        js["meta"]["season"] = args.set_meta_season
    js["meta"]["last_updated_gw"] = gw_now
    _save_registry(js, args.registry)
    print(f"OK ({msg})")

def cmd_set_until(args) -> None:
    js = _load_registry(args.registry)
    players: List[dict] = js.get("players", [])
    gw_now = int(args.gw_now)

    i = _idx(players, args.player_id)
    if i < 0:
        raise SystemExit(f"player_id {args.player_id} not found in registry")
    cur = PlayerUnavail.from_dict(players[i])
    if args.until_gw is not None:
        cur.until_gw = int(args.until_gw)
    if args.matches is not None:
        cur.matches_remaining = int(args.matches)
    cur.last_accounted_gw = gw_now
    cur = _recompute_active(cur, gw_now)
    players[i] = cur.to_dict()

    js["players"] = players
    js.setdefault("meta", {})
    js["meta"]["last_updated_gw"] = gw_now
    _save_registry(js, args.registry)
    print("OK (window set)")

def cmd_clear(args) -> None:
    js = _load_registry(args.registry)
    players: List[dict] = js.get("players", [])
    i = _idx(players, args.player_id)
    if i < 0:
        print("OK (nothing to clear)")
        return
    del players[i]
    js["players"] = players
    _save_registry(js, args.registry)
    print("OK (cleared)")

def cmd_list(args) -> None:
    js = _load_registry(args.registry)
    players: List[dict] = js.get("players", [])
    gw_now = int(args.gw_now) if args.gw_now is not None else (js.get("meta", {}).get("last_updated_gw") or 0)

    rows = []
    for d in players:
        e = PlayerUnavail.from_dict(d)
        active = _is_active(e, gw_now)
        rows.append({
            "player_id": e.player_id,
            "name": e.name,
            "pos": e.pos,
            "team_id": e.team_id,
            "team": e.team,
            "status": e.status,
            "reason": e.reason,
            "detail": e.detail,
            "until_gw": e.until_gw,
            "matches_remaining": e.matches_remaining,
            "added_gw": e.added_gw,
            "last_accounted_gw": e.last_accounted_gw,
            "active_now": active
        })
    print(json.dumps({"gw_now": gw_now, "players": rows}, indent=2))

def cmd_tick(args) -> None:
    js = _load_registry(args.registry)
    players: List[dict] = js.get("players", [])
    gw_now = int(args.gw_now)

    new_players = []
    for d in players:
        e = PlayerUnavail.from_dict(d)
        e = _tick_entry(e, gw_now)
        new_players.append(e.to_dict())

    js["players"] = new_players
    js.setdefault("meta", {})
    js["meta"]["last_updated_gw"] = gw_now
    _save_registry(js, args.registry)
    print("OK (ticked)")

def cmd_export_banset(args) -> None:
    js = _load_registry(args.registry)
    gw_now = int(args.gw_now)
    ids = _active_banset(js.get("players", []), gw_now)
    out = {"gw_now": gw_now, "count": len(ids), "player_ids": ids}
    if args.out:
        _save_registry(out, args.out)
        print(f"OK (wrote {args.out}, {len(ids)} ids)")
    else:
        print(json.dumps(out, indent=2))

# ----- batch helpers & command -----
def _split_list_csv(s: Optional[str]) -> list[str]:
    if not s:
        return []
    # Split by comma, strip surrounding spaces and quotes
    out = []
    for part in s.split(","):
        v = part.strip().strip('"').strip("'")
        if v:
            out.append(v)
    return out

def _pad_to_len(lst: list, n: int) -> list:
    if not lst:
        return []
    if len(lst) >= n:
        return lst[:n]
    return lst + [lst[-1]] * (n - len(lst))

def cmd_add_batch(args) -> None:
    """
    Add/update multiple entries in one go. Values are comma-separated lists aligned by index.
    At least one of --name or --player-id must be provided as a list.
    """
    names = _split_list_csv(args.name)
    pids  = _split_list_csv(args.player_id)
    clubs = _split_list_csv(args.club)

    if not names and not pids:
        raise SystemExit("Provide at least one of --name or --player-id (as comma-separated lists).")
    if names and pids and len(names) != len(pids):
        raise SystemExit("If both --name and --player-id are provided, list lengths must match.")

    N = len(pids) if pids else len(names)

    statuses = _pad_to_len(_split_list_csv(args.status), N) or ["OUT"] * N
    reasons  = _pad_to_len(_split_list_csv(args.reason), N) or ["INJURY"] * N
    details  = _pad_to_len(_split_list_csv(args.detail), N)
    sources  = _pad_to_len(_split_list_csv(args.source), N)
    untils   = _pad_to_len(_split_list_csv(args.until_gw), N)
    matches  = _pad_to_len(_split_list_csv(args.matches), N)
    gw_nows  = _pad_to_len(_split_list_csv(args.gw_now), N)
    clubs    = _pad_to_len(clubs, N)

    # Load registry & (optionally) master stack once
    js = _load_registry(args.registry)
    players: List[dict] = js.get("players", [])

    master = None
    teams_map = None
    if names or args.master or args.teams_map or args.season:
        if not (args.master and args.teams_map and args.season):
            raise SystemExit("add-batch: When using names or metadata autofill, provide --master, --teams-map, and --season.")
        master = _load_master(args.master)
        teams_map = _load_teams_map(args.teams_map)
        if master is None or teams_map is None:
            raise SystemExit("Failed to load --master or --teams-map.")

    ok, fail = 0, 0
    errors: list[str] = []

    for i in range(N):
        try:
            pid = pids[i] if pids else None
            nm  = names[i] if names else None
            club = clubs[i] if clubs and clubs[i] else None

            if nm:
                if not (master and teams_map and args.season):
                    raise ValueError("Row requires name resolution but metadata inputs are missing.")
                pid, _ = _resolve_player_id_by_name(
                    master, teams_map,
                    name=nm, season_long_or_short=args.season, club_filter=club
                )
            assert pid, "Internal: pid resolution failed"

            # Auto metadata if master provided
            auto_name = auto_pos = auto_team = auto_team_id = None
            if master is not None:
                auto_name, auto_pos, auto_team, auto_team_id = _meta_from_master(
                    master, teams_map, player_id=pid, season_long_or_short=args.season
                )

            status = statuses[i].upper()
            reason = reasons[i].upper()
            if status not in VALID_STATUS:
                raise ValueError(f"Invalid status '{status}' at row {i+1}")
            if reason not in VALID_REASON:
                raise ValueError(f"Invalid reason '{reason}' at row {i+1}")

            detail = details[i] if i < len(details) else None
            source = sources[i] if i < len(sources) else None
            until_gw_val = int(untils[i]) if (i < len(untils) and untils[i]) else None
            matches_val  = int(matches[i]) if (i < len(matches) and matches[i]) else None
            gw_now_val   = int(gw_nows[i]) if (i < len(gw_nows) and gw_nows[i]) else None
            if gw_now_val is None:
                raise ValueError("gw_now missing for a row (provide via --gw-now list or scalar).")

            idx = _idx(players, pid)
            if idx < 0:
                e = PlayerUnavail(
                    player_id=pid,
                    name=(auto_name if auto_name is not None else None),
                    pos=(auto_pos if auto_pos is not None else None),
                    team_id=(auto_team_id if auto_team_id is not None else None),
                    team=(auto_team if auto_team is not None else None),
                    status=status,
                    reason=reason,
                    detail=detail,
                    source=source,
                    added_gw=gw_now_val,
                    last_accounted_gw=gw_now_val,
                    until_gw=until_gw_val,
                    matches_remaining=matches_val,
                    active=True,
                )
                e.validate_metadata()
                e = _recompute_active(e, gw_now_val)
                players.append(e.to_dict())
            else:
                cur = PlayerUnavail.from_dict(players[idx])
                if master is not None:
                    cur.name = auto_name
                    cur.pos = auto_pos
                    cur.team = auto_team
                    cur.team_id = auto_team_id
                cur.validate_metadata()
                cur.status = status
                cur.reason = reason
                if detail is not None:
                    cur.detail = detail
                if source is not None:
                    cur.source = source
                if until_gw_val is not None:
                    cur.until_gw = until_gw_val
                if matches_val is not None:
                    cur.matches_remaining = matches_val
                cur.last_accounted_gw = gw_now_val
                cur = _recompute_active(cur, gw_now_val)
                players[idx] = cur.to_dict()

            ok += 1
        except Exception as e:
            fail += 1
            who = (names[i] if names and i < len(names) else (pids[i] if pids and i < len(pids) else f"row {i+1}"))
            errors.append(f"[row {i+1} {who}] {e}")

    js["players"] = players
    js.setdefault("meta", {})
    if args.set_meta_season:
        js["meta"]["season"] = args.set_meta_season
    # last_updated_gw = max of provided gw_now values
    existing = js.get("meta", {}).get("last_updated_gw", 0)
    gw_vals = [int(x) for x in gw_nows if x] if gw_nows else []
    js["meta"]["last_updated_gw"] = max([existing] + gw_vals)
    _save_registry(js, args.registry)

    print(f"OK (batch): {ok} success, {fail} failed")
    if errors:
        for msg in errors[:20]:
            print(" -", msg)
        if len(errors) > 20:
            print(f" ... and {len(errors)-20} more")
        if fail:
            sys.exit(2)

# ----- parser -----
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="League-wide unavailability registry (add by id or name; auto metadata from master)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # add
    p = sub.add_parser("add", help="Add/update an unavailability")
    p.add_argument("registry", help="path to league_unavailable.json")
    g_ident = p.add_mutually_exclusive_group(required=True)
    g_ident.add_argument("--player-id", help="canonical player id")
    g_ident.add_argument("--name", help="display name (accent-insensitive, case-insensitive; supports substring fallback)")
    p.add_argument("--club", help="team short code to disambiguate name (e.g., MCI)")

    # master-powered autofill (required for --name; optional for --player-id if you want to refresh metadata)
    p.add_argument("--master", help="path to master_fpl.json for metadata autofill / name resolution")
    p.add_argument("--teams-map", help="JSON mapping {TEAM_CODE -> canonical team_id}")
    p.add_argument("--season", help="season for metadata lookup (e.g., 2025-2026)")

    # status
    p.add_argument("--status", required=True, choices=sorted(VALID_STATUS))
    p.add_argument("--reason", required=True, choices=sorted(VALID_REASON))
    p.add_argument("--detail", help="free text")
    p.add_argument("--source", help="where did this come from")

    # timing
    p.add_argument("--gw-now", type=int, required=True, help="current gw to evaluate activity at")
    p.add_argument("--until-gw", type=int, help="absolute end gw (optional)")
    p.add_argument("--matches", type=int, help="remaining fixtures counter (optional)")

    # optional: set registry.meta.season
    p.add_argument("--set-meta-season", help="override registry.meta.season")
    p.set_defaults(func=cmd_add)

    # add-batch
    p = sub.add_parser("add-batch", help="Add/update multiple entries with comma-separated lists")
    p.add_argument("registry", help="path to league_unavailable.json")
    p.add_argument("--player-id", help="comma-separated player ids")
    p.add_argument("--name", help='comma-separated names (quote the whole argument). Example: --name "A, B, C"')
    p.add_argument("--club", help="comma-separated team short codes aligned to names (optional)")
    p.add_argument("--master", help="path to master_fpl.json")
    p.add_argument("--teams-map", help="JSON mapping {TEAM_CODE -> canonical team_id}")
    p.add_argument("--season", help="season for metadata lookup, e.g., 2025-2026")
    p.add_argument("--status", help='comma-separated statuses (OUT,SUSPENDED,DOUBTFUL,RESTED,UNKNOWN)')
    p.add_argument("--reason", help='comma-separated reasons (INJURY,SUSPENSION,AFCON,NATIONS_CUP,PERSONAL,REST,TRANSFERRING,TRANSFERRED,UNKNOWN)')
    p.add_argument("--detail", help="comma-separated details")
    p.add_argument("--source", help="comma-separated sources")
    p.add_argument("--until-gw", help="comma-separated ints")
    p.add_argument("--matches", help="comma-separated ints")
    p.add_argument("--gw-now", required=True, help="comma-separated ints, or a single scalar applied to all")
    p.add_argument("--set-meta-season", help="override registry.meta.season")
    p.set_defaults(func=cmd_add_batch)

    # set-until
    p = sub.add_parser("set-until", help="Set/adjust end window for an entry")
    p.add_argument("registry")
    p.add_argument("--player-id", required=True)
    p.add_argument("--gw-now", type=int, required=True)
    p.add_argument("--until-gw", type=int)
    p.add_argument("--matches", type=int)
    p.set_defaults(func=cmd_set_until)

    # clear
    p = sub.add_parser("clear", help="Remove a player from registry")
    p.add_argument("registry")
    p.add_argument("--player-id", required=True)
    p.set_defaults(func=cmd_clear)

    # list
    p = sub.add_parser("list", help="List entries (with active_now computed)")
    p.add_argument("registry")
    p.add_argument("--gw-now", type=int, help="override gw for activity computation")
    p.set_defaults(func=cmd_list)

    # tick
    p = sub.add_parser("tick", help="Advance one GW: decrement counters, recompute active")
    p.add_argument("registry")
    p.add_argument("--gw-now", type=int, required=True)
    p.set_defaults(func=cmd_tick)

    # export-banset
    p = sub.add_parser("export-banset", help="Write active ids for optimizers to exclude")
    p.add_argument("registry")
    p.add_argument("--gw-now", type=int, required=True)
    p.add_argument("--out", help="output json (default: stdout)")
    p.set_defaults(func=cmd_export_banset)

    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()
    try:
        # Guardrails for metadata/name flows
        if args.cmd == "add":
            if args.name and not (args.master and args.teams_map and args.season):
                raise SystemExit("Adding by --name requires --master, --teams-map, and --season.")
            if args.player_id and any([args.master, args.teams_map, args.season]) and not (args.master and args.teams_map and args.season):
                raise SystemExit("When refreshing metadata for --player-id, provide --master, --teams-map, and --season together.")
        if args.cmd == "add-batch":
            # If any of name/metadata provided, ensure full stack present
            if (args.name or args.master or args.teams_map or args.season) and not (args.master and args.teams_map and args.season):
                raise SystemExit("add-batch with names/metadata requires --master, --teams-map, --season.")
        args.func(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
