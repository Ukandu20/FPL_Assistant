#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
import copy
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator

# =========================
# ID formats & helpers
# =========================

ALNUM_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
CODE_RE = re.compile(r"^[A-Za-z]{2,4}$")  # short code like 'MCI', 'ARS'

def is_code_like(s: str | None) -> bool:
    return bool(s) and CODE_RE.fullmatch(str(s).strip()) is not None

def is_alnum_id(s: str | None) -> bool:
    s = (s or "").strip()
    return bool(s) and ALNUM_ID_RE.fullmatch(s) is not None and not is_code_like(s)

# =========================
# Core schema (Team State)
# =========================

class ChipState(BaseModel):
    TC: bool
    BB: bool
    FH: bool
    WC1: bool
    WC2: bool

class ChipUsage(BaseModel):
    gw: int = Field(ge=1)
    chip: Literal["TC", "BB", "FH", "WC1", "WC2"]

class TransferLogEntry(BaseModel):
    gw: int = Field(ge=1)
    outs: List[str] = []
    ins: List[str] = []
    bank_before: float = Field(ge=0)
    bank_after: float = Field(ge=0)
    points_hit: int = Field(ge=0, description="Points cost recorded only; DOES NOT affect bank")

class SquadEntry(BaseModel):
    player_id: str
    name: Optional[str] = None
    pos: Literal["GK", "DEF", "MID", "FWD"]
    # IMPORTANT: canonical alphanumeric id (NOT the short code)
    team_id: str
    # Short code for legacy/merge (e.g., 'MCI', 'ARS')
    team: Optional[str] = None
    sell_price: float = Field(ge=0)
    buy_price: float = Field(ge=0)
    purchase_gw: int = Field(ge=1)

    @field_validator("team_id")
    @classmethod
    def _team_id_alnum(cls, v: str) -> str:
        if not is_alnum_id(v):
            raise ValueError(
                f"team_id must be a canonical alphanumeric id (got '{v}'). "
                "Do not put the 3-letter code in team_id; use the 'team' field for that."
            )
        return v

    @field_validator("team")
    @classmethod
    def _team_code_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not is_code_like(v):
            raise ValueError(f"team (short code) must be 2–4 letters (e.g., 'MCI'), got '{v}'.")
        return v

# -------- Unavailability --------
UNAVAIL_REASONS = Literal[
    "INJURY", "SUSPENSION", "AFCON", "NATIONS_CUP", "PERSONAL", "REST", "TRANSFERRING", "TRANSFERRED", "UNKNOWN"
]
UNAVAIL_STATUS = Literal[
    "OUT", "SUSPENDED", "DOUBTFUL", "RESTED", "UNKNOWN"
]

class UnavailableEntry(BaseModel):
    """
    A record indicating the player is (or was) unavailable.
    - DOUBTFUL status is treated as *available* unless otherwise changed.
    - TRANSFERRED reason is treated as unavailable until you clear it.
    - matches_remaining is a simple GW-based counter (decrements when GW increases).
    """
    player_id: str
    reason: UNAVAIL_REASONS = "UNKNOWN"
    status: UNAVAIL_STATUS = "OUT"
    detail: Optional[str] = None
    source: Optional[str] = None

    added_gw: int = Field(ge=1)
    last_accounted_gw: int = Field(ge=1)
    until_gw: Optional[int] = Field(default=None, ge=1)
    matches_remaining: Optional[int] = Field(default=None, ge=0)

    active: bool = True  # remains in list even when False (audit)

    @field_validator("player_id")
    @classmethod
    def _pid_ok(cls, v: str) -> str:
        if not is_alnum_id(v):
            raise ValueError("player_id must be canonical alphanumeric (not a team code).")
        return v

def _is_entry_active(entry: UnavailableEntry, curr_gw: int) -> bool:
    """Active means: within window, matches_remaining not exhausted, and NOT 'DOUBTFUL'.
       TRANSFERRED is always treated as unavailable (active) until you clear it."""
    if entry.status == "DOUBTFUL":
        return False
    if entry.reason == "TRANSFERRED":
        return True
    if entry.until_gw is not None and curr_gw > entry.until_gw:
        return False
    if entry.matches_remaining is not None and entry.matches_remaining <= 0:
        return False
    return True

def tick_unavailability(state: "TeamState", *, to_gw: Optional[int] = None) -> None:
    """Advance suspension counters to current GW; recompute 'active'."""
    curr = int(to_gw if to_gw is not None else state.gw)
    changed = False
    for i, e in enumerate(list(state.unavailable or [])):
        delta = max(0, curr - e.last_accounted_gw)
        if delta and e.matches_remaining is not None and e.matches_remaining > 0:
            new_left = max(0, e.matches_remaining - delta)  # GW proxy for "matches"
            if new_left != e.matches_remaining:
                state.unavailable[i].matches_remaining = new_left
                changed = True
        if e.last_accounted_gw != curr:
            state.unavailable[i].last_accounted_gw = curr
            changed = True
        new_active = _is_entry_active(state.unavailable[i], curr)
        if state.unavailable[i].active != new_active:
            state.unavailable[i].active = new_active
            changed = True
    if changed:
        pass  # no-op; state mutated

def active_unavailable_ids(state: "TeamState") -> List[str]:
    tick_unavailability(state)
    return [e.player_id for e in (state.unavailable or []) if e.active]

class TeamState(BaseModel):
    # Season/GW pointers
    season: str
    gw: int = Field(ge=1)   # snapshot_gw: last concluded GW used for price snapshots

    # Finance
    initial_budget: float = Field(ge=0, default=100.0)
    bank: float = Field(ge=0)

    # Persisted values
    value_liquidation: float = Field(ge=0, default=0.0) # bank + sum(sell_price)
    value_market: float = Field(ge=0, default=0.0)      # bank + sum(current market price @ state.gw)

    # Transfers & chips
    free_transfers: int = Field(ge=0, default=1)
    chips: ChipState
    chips_log: List[ChipUsage] = []
    transfer_log: List[TransferLogEntry] = []

    # Free Hit backup (bank + squad)
    fh_active: bool = False
    fh_backup_bank: Optional[float] = None
    fh_backup_squad: Optional[List[SquadEntry]] = None

    # Squad
    squad: List[SquadEntry]
    unavailable: List[UnavailableEntry] = []

    @field_validator("squad")
    @classmethod
    def _size_ok(cls, v: List[SquadEntry]) -> List[SquadEntry]:
        if not (0 <= len(v) <= 15):
            raise ValueError("squad size must be between 0 and 15")
        return v

# =========================
# IO helpers
# =========================

def load_json(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(obj: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_team_state_strict(path: str | Path) -> TeamState:
    """Strict load: validate immediately (no migration), and tick unavailability."""
    js = load_json(path)
    state = TeamState.model_validate(js)
    tick_unavailability(state)
    return state

def save_team_state(state: TeamState, path: str | Path) -> None:
    js = json.loads(state.model_dump_json())
    dump_json(js, path)

def load_master(master_path: str | Path) -> Dict[str, dict]:
    return load_json(master_path)

def load_teams_map(teams_map_path: Optional[str | Path]) -> Dict[str, str] | None:
    if not teams_map_path:
        return None
    raw = load_json(teams_map_path)
    # Expect {TEAM_CODE -> TEAM_ID} (e.g., {"MCI": "b11c2f05"})
    return {str(k).strip().upper(): str(v).strip() for k, v in raw.items()}

# =========================
# Migration (legacy team_id -> canonical id)
# =========================

def migrate_codes_inplace(js: dict, code2id: Dict[str, str], *, dry_run: bool = False) -> int:
    """
    In-place: if a squad entry has team_id like 'MCI', move it into 'team' and set team_id=code2id['MCI'].
    Returns number of entries changed.
    """
    changed = 0
    squad = js.get("squad") or []
    for p in squad:
        tid = str(p.get("team_id", "")).strip()
        if is_code_like(tid):  # legacy
            code = tid.upper()
            if code not in code2id:
                raise ValueError(f"Cannot map team code '{code}' — add it to your teams_map file.")
            if not dry_run:
                p["team"] = code
                p["team_id"] = code2id[code]
            changed += 1
    return changed

def load_team_state_relaxed(path: str | Path,
                            teams_map_path: Optional[str | Path] = None,
                            auto_migrate: bool = False,
                            save_if_migrated: bool = True) -> TeamState:
    """
    Relaxed load: if auto_migrate and teams_map provided, rewrite legacy 'team_id' codes before validation.
    Falls back to strict validation if nothing to migrate. Also ticks unavailability.
    """
    js = load_json(path)
    code2id = load_teams_map(teams_map_path) if teams_map_path else None

    if auto_migrate and code2id:
        changed = migrate_codes_inplace(js, code2id, dry_run=False)
        if changed and save_if_migrated:
            dump_json(js, path)
            print(f"[team_state] Auto-migrated {changed} squad entr{'y' if changed==1 else 'ies'} (team_id codes -> canonical ids).")

    # Validate after (possible) migration
    state = TeamState.model_validate(js)
    tick_unavailability(state)
    return state

# =========================
# Season / Position helpers
# =========================

_SEASON_SHORT_RE = re.compile(r"^\d{4}-\d{2}$")

def to_short_season(long_or_short: str) -> str:
    s = long_or_short.strip()
    if _SEASON_SHORT_RE.fullmatch(s):
        return s
    if "/" in s:
        a, b = s.split("/", 1)
        return f"{a}-{b}"
    if len(s) == 9 and s[4] == "-":
        a = s[:4]
        b = s[-2:]
        return f"{a}-{b}"
    return s

_POS_MAP = {
    "GKP": "GK", "GK": "GK", "G": "GK",
    "DEF": "DEF", "DF": "DEF", "D": "DEF",
    "MID": "MID", "MF": "MID", "M": "MID",
    "FWD": "FWD", "FW": "FWD", "F": "FWD", "ST": "FWD",
}

def normalize_pos(pos_raw: Optional[str]) -> Optional[str]:
    if pos_raw is None:
        return None
    pr = str(pos_raw).strip().upper()
    return _POS_MAP.get(pr, pr if pr in {"GK","DEF","MID","FWD"} else None)

def list_career_seasons(entry: dict) -> List[str]:
    career = entry.get("career", {}) or {}
    return sorted(career.keys())

def pick_career_season(entry: dict, target_short: str) -> Tuple[Optional[str], Optional[dict]]:
    career = entry.get("career", {}) or {}
    if target_short in career and isinstance(career[target_short], dict):
        return target_short, career[target_short]
    candidates = []
    for k, v in career.items():
        if isinstance(v, dict) and _SEASON_SHORT_RE.fullmatch(k):
            try:
                year = int(k[:4]); candidates.append((year, k, v))
            except Exception:
                continue
    if candidates:
        candidates.sort()
        _, k, v = candidates[-1]
        return k, v
    return None, None

# =========================
# Master-FPL utilities
# =========================

def get_master_name(entry: dict, pid: str) -> str:
    if entry.get("name"):
        return str(entry["name"])
    fn = str(entry.get("first_name") or "").strip()
    sn = str(entry.get("second_name") or "").strip()
    full = (fn + " " + sn).strip()
    return full if full else pid

def price_at_or_before(master_entry: dict, season_short: str, gw: int) -> Optional[float]:
    prices = master_entry.get("prices", {}).get(season_short, {})
    if not prices:
        return None
    gws = sorted(int(k) for k in prices.keys() if str(k).isdigit())
    candidates = [g for g in gws if g <= gw]
    if not candidates:
        return None
    return float(prices[str(max(candidates))])

def current_price(master_entry: dict, season_short: str, current_gw: int) -> Optional[float]:
    return price_at_or_before(master_entry, season_short, current_gw)

def fpl_selling_price(_purchase_price: float, current_price_val: float) -> float:
    cp = Decimal(str(current_price_val))
    return float(cp.quantize(Decimal("0.1"), rounding=ROUND_DOWN))

def get_pos_team_name(master_entry: dict, target_short: str, *, pid: str) -> Tuple[str, str, str, str]:
    season_used, season_info = pick_career_season(master_entry, target_short)
    if not season_info:
        raise ValueError(
            f"Player {pid} has no usable career season for '{target_short}'. "
            f"Available: {list_career_seasons(master_entry)}"
        )
    pos = normalize_pos(season_info.get("fpl_pos") or season_info.get("position"))
    team_code = season_info.get("team")
    name = get_master_name(master_entry, pid)
    if not pos:
        raise ValueError(
            f"Unrecognized position for player {pid} ({name}) in season '{season_used}'. "
            f"Found: {season_info.get('fpl_pos') or season_info.get('position')!r}."
        )
    if not team_code:
        raise ValueError(f"Missing team code for player {pid} ({name}) in season '{season_used}'.")
    return pos, str(team_code).upper(), season_used, name

def resolve_team_id_required(team_code: str, teams_map: Dict[str, str] | None) -> str:
    if not teams_map:
        raise ValueError("teams_map is required to resolve canonical team_id (code -> id).")
    if team_code not in teams_map:
        raise ValueError(f"teams_map missing entry for team code '{team_code}'.")
    tid = str(teams_map[team_code]).strip()
    if not is_alnum_id(tid):
        raise ValueError(
            f"Resolved team_id '{tid}' is not a canonical alphanumeric id. "
            "Fix teams_map so team_id is not just a 3-letter code."
        )
    return tid

def max_known_gw_for_season(master: Dict[str, dict], season_short: str) -> int:
    max_gw = 0
    for entry in master.values():
        prices = entry.get("prices", {}).get(season_short, {})
        for k in prices.keys():
            if str(k).isdigit():
                max_gw = max(max_gw, int(k))
    return max_gw

# =========================
# Composition helpers (2-5-5-3)
# =========================

EXACT_COMPOSITION = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

def count_positions(players: List[SquadEntry]) -> Dict[str, int]:
    c = {"GK":0, "DEF":0, "MID":0, "FWD":0}
    for p in players:
        c[p.pos] += 1
    return c

def validate_exact_composition(state: TeamState) -> None:
    if len(state.squad) != 15:
        raise ValueError(f"squad must have exactly 15 players, has {len(state.squad)}")
    c = count_positions(state.squad)
    for k, need in EXACT_COMPOSITION.items():
        if c[k] != need:
            raise ValueError(f"invalid composition for {k}: have {c[k]}, need {need}")

# =========================
# Domain operations
# =========================

def _build_entry_from_master(
    state: TeamState,
    master: Dict[str, dict],
    *,
    player_id: str,
    purchase_gw: int,
    teams_map: Dict[str, str] | None,
    season_long: str,
    current_gw: int,
    override_buy: Optional[float] = None,
    override_sell: Optional[float] = None,
) -> SquadEntry:
    entry = master[player_id]
    target_short = to_short_season(season_long)

    pos, team_code, _season_used, name = get_pos_team_name(entry, target_short, pid=player_id)
    team_id = resolve_team_id_required(team_code, teams_map)

    if override_buy is not None:
        buy_price = float(override_buy)
    else:
        buy_price = price_at_or_before(entry, target_short, purchase_gw)
        if buy_price is None:
            raise ValueError(
                f"No price data for player {player_id} ({name}) at/<= GW {purchase_gw} in {target_short}."
            )

    if override_sell is not None:
        sell_price = float(override_sell)
    else:
        cp = current_price(entry, target_short, current_gw) or buy_price
        sell_price = fpl_selling_price(buy_price, cp)

    return SquadEntry(
        player_id=player_id,
        name=name,
        pos=pos,
        team_id=team_id,    # canonical alphanumeric id
        team=team_code,     # short code for legacy/merge
        buy_price=buy_price,
        sell_price=sell_price,
        purchase_gw=purchase_gw,
    )

def add_player_from_master(
    state: TeamState,
    master: Dict[str, dict],
    *,
    player_id: str,
    purchase_gw: Optional[int],
    teams_map: Dict[str, str] | None,
    override_buy: Optional[float],
    override_sell: Optional[float],
    season_long: Optional[str] = None,
    current_gw: Optional[int] = None,
) -> None:
    if player_id not in master:
        raise ValueError(f"player_id {player_id} not in master_fpl.json")

    season_long = season_long or state.season
    current_gw = current_gw or state.gw
    purchase_gw = int(purchase_gw if purchase_gw is not None else state.gw)

    if any(p.player_id == player_id for p in state.squad):
        raise ValueError(f"player {player_id} already in squad")
    if len(state.squad) >= 15:
        raise ValueError("Cannot add more than 15 players to the squad")

    state.squad.append(
        _build_entry_from_master(
            state, master,
            player_id=player_id,
            purchase_gw=purchase_gw,
            teams_map=teams_map,
            season_long=season_long,
            current_gw=current_gw,
            override_buy=override_buy,
            override_sell=override_sell,
        )
    )

def update_sell_from_master(
    state: TeamState,
    master: Dict[str, dict],
    *,
    season_long: Optional[str] = None,
    current_gw: Optional[int] = None,
) -> None:
    season_long = season_long or state.season
    target_short = to_short_season(season_long)
    current_gw = current_gw or state.gw

    for i, entry in enumerate(list(state.squad)):
        m = master.get(entry.player_id)
        if not m:
            continue
        cp = current_price(m, target_short, current_gw) or entry.buy_price
        state.squad[i].sell_price = fpl_selling_price(entry.buy_price, cp)

def prices_snapshot_at_gw(
    state: TeamState,
    master: Dict[str, dict],
    *,
    season_long: Optional[str] = None,
    snapshot_gw: Optional[int] = None,
) -> None:
    season_long = season_long or state.season
    target_short = to_short_season(season_long)
    max_known = max_known_gw_for_season(master, target_short)
    if max_known == 0:
        raise ValueError(f"No prices found for season {target_short} in master")

    if snapshot_gw is None:
        snapshot_gw = max_known
    snapshot_gw = int(snapshot_gw)

    if snapshot_gw > max_known:
        raise ValueError(f"snapshot_gw {snapshot_gw} exceeds last concluded GW {max_known}")

    for i, p in enumerate(state.squad):
        m = master.get(p.player_id)
        if not m:
            continue
        cp = current_price(m, target_short, snapshot_gw) or p.buy_price
        state.squad[i].sell_price = fpl_selling_price(p.buy_price, cp)

    state.gw = snapshot_gw
    tick_unavailability(state, to_gw=state.gw)  # keep unavailability consistent

# =========================
# Value helpers
# =========================

def _round1(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_DOWN))

def calc_liquidation_and_market_value(
    state: TeamState,
    master: Dict[str, dict],
    season_long: Optional[str] = None,
    current_gw: Optional[int] = None,
) -> tuple[float, float]:
    season_long = season_long or state.season
    target_short = to_short_season(season_long)
    current_gw = current_gw or state.gw

    liquidation_value = state.bank + sum(p.sell_price for p in state.squad)

    cur_sum = 0.0
    for p in state.squad:
        entry = master.get(p.player_id)
        if entry:
            cp = current_price(entry, target_short, current_gw) or p.buy_price
        else:
            cp = p.buy_price
        cur_sum += cp
    market_value = state.bank + cur_sum

    return _round1(liquidation_value), _round1(market_value)

def recompute_values(
    state: TeamState,
    master: Dict[str, dict],
    *,
    season_long: Optional[str] = None,
    current_gw: Optional[int] = None,
) -> None:
    lv, mv = calc_liquidation_and_market_value(state, master, season_long, current_gw)
    state.value_liquidation = lv
    state.value_market = mv

def sum_buy_prices(state: TeamState) -> float:
    return sum(p.buy_price for p in state.squad)

def _idx_in_squad(state: TeamState, pid: str) -> int:
    for i, p in enumerate(state.squad):
        if p.player_id == pid:
            return i
    return -1

# =========================
# CSV seeding (idempotent)
# =========================

def seed_squad_from_master(
    state: TeamState,
    master: Dict[str, dict],
    *,
    csv_list: str,
    teams_map: Dict[str, str] | None,
    season_long: Optional[str] = None,
    current_gw: Optional[int] = None,
    seed_at_gw: Optional[int] = None,
    on_duplicate: Literal["error", "skip", "replace", "update-sell"] = "error",
    prune_to_csv: bool = False,
    reset_bank: bool = False,
) -> None:
    """
    CSV columns: player_id[,purchase_gw]
    """
    season_long = season_long or state.season
    current_gw = int(current_gw if current_gw is not None else state.gw)
    seed_at_gw = int(seed_at_gw) if seed_at_gw is not None else None

    wanted_order: List[Tuple[str, Optional[int]]] = []
    seen_csv: set[str] = set()
    with open(csv_list, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "player_id" not in reader.fieldnames:
            raise ValueError("CSV must include 'player_id' column")
        for row in reader:
            pid = (row["player_id"] or "").strip()
            if not pid:
                continue
            if pid in seen_csv:
                print(f"[warn] duplicate player_id in CSV ignored: {pid}")
                continue
            seen_csv.add(pid)
            pgw = int(row["purchase_gw"]) if row.get("purchase_gw") else None
            wanted_order.append((pid, pgw))

    idx = {p.player_id: i for i, p in enumerate(state.squad)}

    for pid, pgw in wanted_order:
        if pid not in master:
            raise ValueError(f"player_id {pid} not in master_fpl.json")

        effective_pgw: int
        override_buy: Optional[float] = None

        if pgw is None:
            if seed_at_gw is None:
                raise ValueError(f"CSV missing purchase_gw for {pid} and --seed-at-gw not provided")
            effective_pgw = seed_at_gw
            cp_seed = price_at_or_before(master[pid], to_short_season(season_long), seed_at_gw)
            if cp_seed is None:
                raise ValueError(f"No price at/<= GW {seed_at_gw} for player {pid} during bootstrap")
            override_buy = float(cp_seed)
        else:
            effective_pgw = int(pgw)

        if pid not in idx:
            if len(state.squad) >= 15:
                raise ValueError("Cannot exceed 15 players; consider --prune-to-csv or clear-squad first")
            state.squad.append(
                _build_entry_from_master(
                    state, master,
                    player_id=pid,
                    purchase_gw=effective_pgw,
                    teams_map=teams_map,
                    season_long=season_long,
                    current_gw=current_gw,
                    override_buy=override_buy,
                )
            )
            idx[pid] = len(state.squad) - 1
            continue

        if on_duplicate == "error":
            name = state.squad[idx[pid]].name or pid
            raise ValueError(f"player {pid} ({name}) already in squad")
        elif on_duplicate == "skip":
            continue
        elif on_duplicate == "replace":
            state.squad[idx[pid]] = _build_entry_from_master(
                state, master,
                player_id=pid,
                purchase_gw=effective_pgw,
                teams_map=teams_map,
                season_long=season_long,
                current_gw=current_gw,
                override_buy=override_buy,
            )
        elif on_duplicate == "update-sell":
            m = master[pid]
            target_short = to_short_season(season_long)
            cp = current_price(m, target_short, current_gw) or state.squad[idx[pid]].buy_price
            state.squad[idx[pid]].sell_price = fpl_selling_price(state.squad[idx[pid]].buy_price, cp)
        else:
            raise ValueError(f"Unknown on_duplicate policy: {on_duplicate}")

    if prune_to_csv:
        keep = set(pid for pid, _ in wanted_order)
        state.squad = [p for p in state.squad if p.player_id in keep]

    if reset_bank:
        total_buy = sum_buy_prices(state)
        state.bank = max(0.0, state.initial_budget - total_buy)

# =========================
# Transfers (finance-only; points hits are logged only)
# =========================

def sell_player_from_master(
    state: TeamState,
    master: Dict[str, dict],
    *,
    player_id: str,
    season_long: Optional[str] = None,
    asof_gw: Optional[int] = None,
    points_hit: int = 0,
    log: bool = True,
) -> float:
    season_long = season_long or state.season
    asof_gw = int(asof_gw if asof_gw is not None else state.gw)
    target_short = to_short_season(season_long)

    i = _idx_in_squad(state, player_id)
    if i < 0:
        raise ValueError(f"player {player_id} not found in squad")

    p = state.squad[i]
    m = master.get(p.player_id)
    cp = current_price(m, target_short, asof_gw) if m else p.buy_price
    proceeds = fpl_selling_price(p.buy_price, cp)

    bank_before = state.bank
    state.bank += proceeds
    del state.squad[i]

    if log:
        state.transfer_log.append(TransferLogEntry(
            gw=asof_gw, outs=[player_id], ins=[], bank_before=bank_before, bank_after=state.bank, points_hit=max(0, points_hit)
        ))
    return proceeds

def buy_player_from_master(
    state: TeamState,
    master: Dict[str, dict],
    *,
    player_id: str,
    teams_map: Dict[str, str] | None = None,
    season_long: Optional[str] = None,
    asof_gw: Optional[int] = None,
    points_hit: int = 0,
    log: bool = True,
) -> float:
    if any(p.player_id == player_id for p in state.squad):
        raise ValueError(f"player {player_id} already in squad")
    if len(state.squad) >= 15:
        raise ValueError("Cannot add more than 15 players to the squad")
    if player_id not in master:
        raise ValueError(f"player_id {player_id} not in master_fpl.json")

    season_long = season_long or state.season
    asof_gw = int(asof_gw if asof_gw is not None else state.gw)
    target_short = to_short_season(season_long)
    m = master[player_id]

    cp = current_price(m, target_short, asof_gw)
    if cp is None:
        raise ValueError(f"No price at/<= GW {asof_gw} for player {player_id}")

    pos, team_code, _season_used, name = get_pos_team_name(m, target_short, pid=player_id)
    team_id = resolve_team_id_required(team_code, teams_map)  # MUST map to canonical id
    buy_price = float(cp)
    sell_price = fpl_selling_price(buy_price, buy_price)

    bank_before = state.bank
    if state.bank + 1e-9 < buy_price:
        raise ValueError(f"Insufficient bank: need {buy_price:.1f}, have {state.bank:.1f}")
    state.bank -= buy_price
    state.squad.append(SquadEntry(
        player_id=player_id, name=name, pos=pos, team_id=team_id, team=team_code,
        buy_price=buy_price, sell_price=sell_price, purchase_gw=asof_gw+1
    ))

    if log:
        state.transfer_log.append(TransferLogEntry(
            gw=asof_gw, outs=[], ins=[player_id], bank_before=bank_before, bank_after=state.bank, points_hit=max(0, points_hit)
        ))
    return buy_price

def swap_transfer_from_master(
    state: TeamState,
    master: Dict[str, dict],
    *,
    out_id: str,
    in_id: str,
    teams_map: Dict[str, str] | None = None,
    season_long: Optional[str] = None,
    asof_gw: Optional[int] = None,
    use_free_transfers: int = 1,
    points_hit: int = 0,
) -> Tuple[float, float, float]:
    if out_id == in_id:
        raise ValueError("out_id and in_id cannot be the same")

    season_long = season_long or state.season
    asof_gw = int(asof_gw if asof_gw is not None else state.gw)
    target_short = to_short_season(season_long)

    if len(state.squad) == 15:
        validate_exact_composition(state)

    m_out = master.get(out_id)
    if m_out is None:
        raise ValueError(f"out_id {out_id} not in master")
    m_in = master.get(in_id)
    if m_in is None:
        raise ValueError(f"in_id {in_id} not in master")

    i_out = _idx_in_squad(state, out_id)
    if i_out < 0:
        raise ValueError(f"player {out_id} not found in squad")
    p_out = state.squad[i_out]
    cp_out = current_price(m_out, target_short, asof_gw) or p_out.buy_price
    proceeds = fpl_selling_price(p_out.buy_price, cp_out)

    pos_in, team_code_in, _season_used, _name_in = get_pos_team_name(m_in, target_short, pid=in_id)
    cp_in = current_price(m_in, target_short, asof_gw)
    if cp_in is None:
        raise ValueError(f"No price at/<= GW {asof_gw} for player {in_id}")
    cost = float(cp_in)

    bank_before = state.bank
    if bank_before + proceeds - cost < -1e-9:
        need = cost - (bank_before + proceeds)
        raise ValueError(f"Swap would make bank negative by {need:.1f} (proceeds {proceeds:.1f}, cost {cost:.1f}, bank {bank_before:.1f})")

    new_players = state.squad.copy()
    new_players[i_out] = SquadEntry(
        player_id=in_id,
        name=None,
        pos=pos_in,
        team_id="DUMMY",   # composition only; not persisted
        team=None,
        buy_price=0.0,
        sell_price=0.0,
        purchase_gw=asof_gw+1,
    )
    if len(new_players) == 15:
        c = count_positions(new_players)
        for k, need in EXACT_COMPOSITION.items():
            if c[k] != need:
                raise ValueError(f"Swap breaks composition {k}: would have {c[k]}, need {need}")

    proceeds_real = sell_player_from_master(
        state, master, player_id=out_id, season_long=season_long, asof_gw=asof_gw, points_hit=0, log=False
    )
    _ = resolve_team_id_required(team_code_in, teams_map)  # sanity early
    cost_real = buy_player_from_master(
        state, master, player_id=in_id, teams_map=teams_map, season_long=season_long, asof_gw=asof_gw, points_hit=0, log=False
    )

    used = max(0, min(use_free_transfers, state.free_transfers))
    state.free_transfers -= used

    delta = state.bank - bank_before
    state.transfer_log.append(TransferLogEntry(
        gw=asof_gw, outs=[out_id], ins=[in_id], bank_before=bank_before, bank_after=state.bank, points_hit=max(0, points_hit)
    ))
    return proceeds_real, cost_real, delta

# =========================
# Free Hit helpers
# =========================

def fh_begin(state: TeamState, *, gw: Optional[int] = None, chips_log: bool = True) -> None:
    if state.fh_active:
        raise ValueError("Free Hit already active; restore before starting another")
    state.fh_backup_bank = state.bank
    state.fh_backup_squad = copy.deepcopy(state.squad)
    state.fh_active = True
    if chips_log:
        use_gw = int(gw if gw is not None else (state.gw + 1))
        if state.chips.FH:
            state.chips.FH = False
        state.chips_log.append(ChipUsage(gw=use_gw, chip="FH"))

def fh_restore(state: TeamState) -> None:
    if not state.fh_active:
        raise ValueError("Free Hit not active; nothing to restore")
    if state.fh_backup_bank is None or state.fh_backup_squad is None:
        raise ValueError("Free Hit backup missing")
    state.bank = float(state.fh_backup_bank)
    state.squad = copy.deepcopy(state.fh_backup_squad)
    state.fh_active = False
    state.fh_backup_bank = None
    state.fh_backup_squad = None

# =========================
# CLI command impls
# =========================

def load_team_state_and_tick(path: str | Path) -> TeamState:
    state = load_team_state_relaxed(path)
    tick_unavailability(state)
    return state

def _cli_init(args):
    state = TeamState(
        season=args.season,
        gw=args.gw,
        initial_budget=float(args.initial_budget),
        bank=float(args.initial_budget),
        value_liquidation=float(args.initial_budget),
        value_market=float(args.initial_budget),
        free_transfers=args.free_transfers,
        chips=ChipState(TC=True, BB=True, FH=True, WC1=True, WC2=True),
        chips_log=[],
        transfer_log=[],
        fh_active=False,
        fh_backup_bank=None,
        fh_backup_squad=None,
        squad=[],
        unavailable=[],
    )
    save_team_state(state, args.out)
    print(f"Wrote {args.out}")

def _cli_show(args):
    state = load_team_state_strict(args.path)
    print(state.model_dump_json(indent=2))

def _cli_set_bank(args):
    state = load_team_state_and_tick(args.path)
    state.bank = float(args.bank)
    save_team_state(state, args.path)
    print("OK")

def _cli_set_gw(args):
    state = load_team_state_and_tick(args.path)
    state.gw = int(args.gw)
    tick_unavailability(state, to_gw=state.gw)  # UNAVAIL TICK
    save_team_state(state, args.path)
    print("OK")

def _cli_set_chip(args):
    state = load_team_state_and_tick(args.path)
    chip = args.chip
    value = args.value
    if chip not in state.chips.model_fields:
        raise ValueError(f"Unknown chip: {chip}")
    setattr(state.chips, chip, value)
    save_team_state(state, args.path)
    print("OK")

def _cli_use_chip(args):
    state = load_team_state_and_tick(args.path)
    chip = args.chip
    gw = int(args.gw or (state.gw + 1))
    state.chips_log.append(ChipUsage(gw=gw, chip=chip))
    if getattr(state.chips, chip):
        setattr(state.chips, chip, False)
    save_team_state(state, args.path)
    print("OK")

def _cli_add_player(args):
    state = load_team_state_and_tick(args.path)
    if any(p.player_id == args.player_id for p in state.squad):
        raise ValueError(f"player {args.player_id} already in squad")
    if len(state.squad) >= 15:
        raise ValueError("Cannot add more than 15 players to the squad")
    if not is_alnum_id(args.team_id):
        raise ValueError("team_id must be canonical alphanumeric (not the 3-letter code).")
    if args.team and not is_code_like(args.team):
        raise ValueError("team (short code) must be 2–4 letters (e.g., MCI).")
    state.squad.append(
        SquadEntry(
            player_id=args.player_id,
            name=args.name,
            pos=args.pos,
            team_id=args.team_id,
            team=args.team,
            buy_price=float(args.buy_price),
            sell_price=float(args.sell_price if args.sell_price is not None else args.buy_price),
            purchase_gw=int(args.purchase_gw if args.purchase_gw is not None else state.gw),
        )
    )
    save_team_state(state, args.path)
    print("OK")

def _cli_remove_player(args):
    state = load_team_state_and_tick(args.path)
    before = len(state.squad)
    state.squad = [p for p in state.squad if p.player_id != args.player_id]
    if len(state.squad) == before:
        raise ValueError(f"player {args.player_id} not found")
    save_team_state(state, args.path)
    print("OK")

def _cli_clear_squad(args):
    state = load_team_state_and_tick(args.path)
    state.squad = []
    save_team_state(state, args.path)
    print("OK (squad cleared)")

def _cli_add_from_master(args):
    # relaxed load with auto-migration using --teams-map
    state = load_team_state_relaxed(args.path, teams_map_path=args.teams_map, auto_migrate=True)
    master = load_master(args.master)
    teams_map = load_teams_map(args.teams_map)
    if not teams_map:
        raise ValueError("--teams-map is required so team_id is the canonical alphanumeric id.")
    add_player_from_master(
        state,
        master,
        player_id=args.player_id,
        purchase_gw=args.purchase_gw,
        teams_map=teams_map,
        override_buy=args.buy_price,
        override_sell=args.sell_price,
        season_long=args.season or state.season,
        current_gw=args.current_gw or state.gw,
    )
    recompute_values(state, master, season_long=args.season or state.season, current_gw=args.current_gw or state.gw)
    save_team_state(state, args.path)
    print("OK")

def _cli_seed_from_master(args):
    # relaxed load with auto-migration using --teams-map
    state = load_team_state_relaxed(args.path, teams_map_path=args.teams_map, auto_migrate=True)
    master = load_master(args.master)
    teams_map = load_teams_map(args.teams_map)
    if not teams_map:
        raise ValueError("--teams-map is required so team_id is the canonical alphanumeric id.")
    seed_squad_from_master(
        state,
        master,
        csv_list=args.list_csv,
        teams_map=teams_map,
        season_long=args.season or state.season,
        current_gw=args.current_gw or state.gw,
        seed_at_gw=args.seed_at_gw,
        on_duplicate=args.on_duplicate,
        prune_to_csv=args.prune_to_csv,
        reset_bank=args.reset_bank,
    )
    recompute_values(state, master, season_long=args.season or state.season, current_gw=args.current_gw or state.gw)
    save_team_state(state, args.path)
    print("OK")

def _cli_update_sell_from_master(args):
    state = load_team_state_relaxed(args.path)  # no teams-map needed here
    master = load_master(args.master)
    update_sell_from_master(
        state, master,
        season_long=args.season or state.season,
        current_gw=args.current_gw or state.gw,
    )
    recompute_values(state, master, season_long=args.season or state.season, current_gw=args.current_gw or state.gw)
    save_team_state(state, args.path)
    print("OK")

def _cli_prices_snapshot(args):
    state = load_team_state_relaxed(args.path)
    master = load_master(args.master)
    prices_snapshot_at_gw(
        state, master,
        season_long=args.season or state.season,
        snapshot_gw=args.gw
    )
    recompute_values(state, master, season_long=args.season or state.season, current_gw=state.gw)
    save_team_state(state, args.path)
    print("OK")

def _cli_value(args):
    state = load_team_state_relaxed(args.path)
    master = load_master(args.master)
    recompute_values(state, master, season_long=args.season or state.season, current_gw=args.current_gw or state.gw)
    save_team_state(state, args.path)
    print(f"liquidation_value: {state.value_liquidation:.1f}")
    print(f"market_value:      {state.value_market:.1f}")

def _cli_sell_from_master(args):
    state = load_team_state_relaxed(args.path)
    master = load_master(args.master)
    proceeds = sell_player_from_master(
        state, master,
        player_id=args.player_id,
        season_long=args.season or state.season,
        asof_gw=args.asof_gw or state.gw,
        points_hit=args.points_hit or 0
    )
    recompute_values(state, master, season_long=args.season or state.season, current_gw=state.gw)
    save_team_state(state, args.path)
    print(f"OK (proceeds credited: {proceeds:.1f})")

def _cli_buy_from_master(args):
    state = load_team_state_relaxed(args.path, teams_map_path=args.teams_map, auto_migrate=True)
    master = load_master(args.master)
    teams_map = load_teams_map(args.teams_map)
    if not teams_map:
        raise ValueError("--teams-map is required so team_id is the canonical alphanumeric id.")
    cost = buy_player_from_master(
        state, master,
        player_id=args.player_id,
        teams_map=teams_map,
        season_long=args.season or state.season,
        asof_gw=args.asof_gw or state.gw,
        points_hit=args.points_hit or 0
    )
    recompute_values(state, master, season_long=args.season or state.season, current_gw=state.gw)
    save_team_state(state, args.path)
    print(f"OK (cost debited: {cost:.1f})")

def _cli_swap_from_master(args):
    state = load_team_state_relaxed(args.path, teams_map_path=args.teams_map, auto_migrate=True)
    master = load_master(args.master)
    teams_map = load_teams_map(args.teams_map)
    if not teams_map:
        raise ValueError("--teams-map is required so team_id is the canonical alphanumeric id.")
    proceeds, cost, delta = swap_transfer_from_master(
        state, master,
        out_id=args.out_id, in_id=args.in_id,
        teams_map=teams_map,
        season_long=args.season or state.season,
        asof_gw=args.asof_gw or state.gw,
        use_free_transfers=args.use_ft or 0,
        points_hit=args.points_hit or 0
    )
    recompute_values(state, master, season_long=args.season or state.season, current_gw=state.gw)
    save_team_state(state, args.path)
    print(f"OK (sold {args.out_id} for {proceeds:.1f}, bought {args.in_id} for {cost:.1f}, bank Δ {delta:+.1f})")

def _cli_audit(args):
    state = load_team_state_relaxed(args.path)
    try:
        validate_exact_composition(state)
        print("OK: squad has exact 2 GK / 5 DEF / 5 MID / 3 FWD")
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(2)

def _cli_fh_begin(args):
    state = load_team_state_relaxed(args.path)
    fh_begin(state, gw=args.gw, chips_log=not args.no_log)
    save_team_state(state, args.path)
    print("OK (Free Hit started; bank+squad snapshot saved)")

def _cli_fh_restore(args):
    state = load_team_state_relaxed(args.path)
    fh_restore(state)
    save_team_state(state, args.path)
    print("OK (Free Hit restored; bank+squad reverted)")

def _cli_migrate_codes(args):
    js = load_json(args.path)
    code2id = load_teams_map(args.teams_map)
    if not code2id:
        raise ValueError("--teams-map is required for migrate-codes.")
    changed = migrate_codes_inplace(js, code2id, dry_run=args.dry_run)
    if args.dry_run:
        print(f"[dry-run] Would migrate {changed} entr{'y' if changed==1 else 'ies'}.")
        # Also try validation on a copy if requested
        if args.validate:
            try:
                TeamState.model_validate(js)
                print("[dry-run] Validation would PASS after migration.")
            except ValidationError as e:
                print("[dry-run] Validation would FAIL after migration:")
                print(e)
        return
    dump_json(js, args.path)
    # Validate after writing to ensure future commands won't fail
    try:
        _ = TeamState.model_validate(js)
    except ValidationError as e:
        print("Wrote migrated file, but validation failed:", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(2)
    print(f"OK (migrated {changed} entr{'y' if changed==1 else 'ies'}).")

# ---------------- Unavailability CLI ----------------

def _cli_unavail_add(args):
    state = load_team_state_and_tick(args.path)
    # update existing if present, else append
    found = False
    for i, e in enumerate(state.unavailable or []):
        if e.player_id == args.player_id:
            state.unavailable[i].reason = args.reason
            state.unavailable[i].status = args.status
            state.unavailable[i].detail = args.detail
            state.unavailable[i].source = args.source
            state.unavailable[i].until_gw = args.until_gw
            state.unavailable[i].matches_remaining = args.matches
            state.unavailable[i].last_accounted_gw = state.gw
            found = True
            break
    if not found:
        new = UnavailableEntry(
            player_id=args.player_id,
            reason=args.reason,
            status=args.status,
            detail=args.detail,
            source=args.source,
            added_gw=state.gw,
            last_accounted_gw=state.gw,
            until_gw=args.until_gw,
            matches_remaining=args.matches,
            active=True,
        )
        (state.unavailable or []).append(new)
    tick_unavailability(state)
    save_team_state(state, args.path)
    print("OK (unavailable added/updated)")

def _cli_unavail_set_until(args):
    state = load_team_state_and_tick(args.path)
    pid = args.player_id
    found = False
    for i, e in enumerate(state.unavailable or []):
        if e.player_id == pid:
            state.unavailable[i].until_gw = args.until_gw
            found = True
    if not found:
        raise ValueError(f"{pid} not found in unavailable list")
    tick_unavailability(state)
    save_team_state(state, args.path)
    print("OK (until_gw set)")

def _cli_unavail_clear(args):
    state = load_team_state_and_tick(args.path)
    before = len(state.unavailable or [])
    state.unavailable = [e for e in (state.unavailable or []) if e.player_id != args.player_id]
    if len(state.unavailable) == before:
        raise ValueError("player_id not in unavailable list")
    save_team_state(state, args.path)
    print("OK (unavailable entry removed)")

def _cli_unavail_list(args):
    state = load_team_state_and_tick(args.path)
    tick_unavailability(state)
    js = [json.loads(e.model_dump_json()) for e in (state.unavailable or [])]
    print(json.dumps(js, indent=2, ensure_ascii=False))

def _cli_unavail_tick(args):
    state = load_team_state_and_tick(args.path)
    tick_unavailability(state)
    save_team_state(state, args.path)
    print("OK (unavailability ticked)")

# =========================
# Parser
# =========================

def _build_parser():
    ap = argparse.ArgumentParser(
        description="Manage team state (master integration, snapshots, transactions, composition checks, FH backup, legacy migration, unavailability)"
    )
    sub = ap.add_subparsers(required=True)

    # init
    p = sub.add_parser("init", help="create a fresh team_state.json")
    p.add_argument("--season", required=True)
    p.add_argument("--gw", type=int, required=True, help="snapshot_gw (last concluded GW)")
    p.add_argument("--initial-budget", type=float, default=100.0)
    p.add_argument("--free-transfers", type=int, default=1)
    p.add_argument("--out", default="state/team_state.json")
    p.set_defaults(func=_cli_init)

    # show
    p = sub.add_parser("show", help="print current team_state.json")
    p.add_argument("path", help="path to team_state.json", nargs="?")
    p.set_defaults(func=_cli_show)

    # set-bank
    p = sub.add_parser("set-bank")
    p.add_argument("path")
    p.add_argument("--bank", type=float, required=True)
    p.set_defaults(func=_cli_set_bank)

    # set-gw
    p = sub.add_parser("set-gw")
    p.add_argument("path")
    p.add_argument("--gw", type=int, required=True)
    p.set_defaults(func=_cli_set_gw)

    # set-chip
    p = sub.add_parser("set-chip")
    p.add_argument("path")
    p.add_argument("--chip", required=True, choices=["TC", "BB", "FH", "WC1", "WC2"])
    p.add_argument("--value", required=True, type=lambda s: s.lower() in {"1", "true", "yes", "y"})
    p.set_defaults(func=_cli_set_chip)

    # use-chip (log usage)
    p = sub.add_parser("use-chip")
    p.add_argument("path")
    p.add_argument("--chip", required=True, choices=["TC", "BB", "FH", "WC1", "WC2"])
    p.add_argument("--gw", type=int, help="defaults to upcoming GW (state.gw + 1)")
    p.set_defaults(func=_cli_use_chip)

    # add-player (manual)
    p = sub.add_parser("add-player")
    p.add_argument("path")
    p.add_argument("--player-id", required=True)
    p.add_argument("--name", required=False)
    p.add_argument("--pos", required=True, choices=["GK", "DEF", "MID", "FWD"])
    p.add_argument("--team-id", required=True, help="canonical alphanumeric id")
    p.add_argument("--team", required=False, help="short team code (e.g., MCI, ARS)")
    p.add_argument("--buy-price", type=float, required=True)
    p.add_argument("--sell-price", type=float)
    p.add_argument("--purchase-gw", type=int)
    p.set_defaults(func=_cli_add_player)

    # remove-player
    p = sub.add_parser("remove-player")
    p.add_argument("path")
    p.add_argument("--player-id", required=True)
    p.set_defaults(func=_cli_remove_player)

    # clear-squad
    p = sub.add_parser("clear-squad", help="wipe squad but keep season/gw/bank")
    p.add_argument("path")
    p.set_defaults(func=_cli_clear_squad)

    # add-from-master
    p = sub.add_parser("add-from-master", help="add one player by player_id using master_fpl.json data")
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--master", required=True, help="path to master_fpl.json")
    p.add_argument("--player-id", required=True)
    p.add_argument("--purchase-gw", type=int, help="defaults to state.gw if omitted")
    p.add_argument("--season", help="override season for lookup (default: state.season)")
    p.add_argument("--current-gw", type=int, help="override snapshot gw for sell computation (default: state.gw)")
    p.add_argument("--teams-map", required=True, help="JSON mapping {TEAM_CODE -> canonical team_id}")
    p.add_argument("--buy-price", type=float, help="override buy_price")
    p.add_argument("--sell-price", type=float, help="override sell_price")
    p.set_defaults(func=_cli_add_from_master)

    # seed-squad-from-master
    p = sub.add_parser("seed-squad-from-master", help="seed squad from CSV with columns: player_id[,purchase_gw]")
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--master", required=True, help="path to master_fpl.json")
    p.add_argument("--list-csv", required=True, help="CSV with player_id[,purchase_gw]")
    p.add_argument("--season", help="override season (default: state.season)")
    p.add_argument("--current-gw", type=int, help="snapshot gw used to compute sell (default: state.gw)")
    p.add_argument("--seed-at-gw", type=int, help="if purchase_gw missing, bootstrap buy_price at this gw")
    p.add_argument("--teams-map", required=True, help="JSON mapping {TEAM_CODE -> canonical team_id}")
    p.add_argument("--on-duplicate", choices=["error","skip","replace","update-sell"], default="error")
    p.add_argument("--prune-to-csv", action="store_true")
    p.add_argument("--reset-bank", action="store_true", help="after seeding, set bank=initial_budget-sum(buy_price)")
    p.set_defaults(func=_cli_seed_from_master)

    # update-sell-from-master
    p = sub.add_parser("update-sell-from-master", help="recompute sell_price for all players vs current_gw")
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--master", required=True, help="path to master_fpl.json")
    p.add_argument("--season", help="override season (default: state.season)")
    p.add_argument("--current-gw", type=int, help="snapshot gw (default: state.gw)")
    p.set_defaults(func=_cli_update_sell_from_master)

    # prices-snapshot
    p = sub.add_parser("prices-snapshot", help="lock snapshot to last concluded GW (or provided gw) and recompute sells")
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--master", required=True, help="path to master_fpl.json")
    p.add_argument("--season", help="override season (default: state.season)")
    p.add_argument("--gw", type=int, help="snapshot gw; defaults to last concluded GW available in master")
    p.set_defaults(func=_cli_prices_snapshot)

    # value
    p = sub.add_parser("value", help="recompute + print liquidation_value and market_value")
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--master", required=True, help="path to master_fpl.json")
    p.add_argument("--season", help="override season (default: state.season)")
    p.add_argument("--current-gw", type=int, help="snapshot gw (default: state.gw)")
    p.set_defaults(func=_cli_value)

    # sell-from-master
    p = sub.add_parser("sell-from-master", help="sell a player at as-of snapshot gw (credits bank, removes)")
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--master", required=True, help="path to master_fpl.json")
    p.add_argument("--player-id", required=True)
    p.add_argument("--season", help="override season (default: state.season)")
    p.add_argument("--asof-gw", type=int, help="snapshot gw (default: state.gw)")
    p.add_argument("--points-hit", type=int, default=0)
    p.set_defaults(func=_cli_sell_from_master)

    # buy-from-master
    p = sub.add_parser("buy-from-master", help="buy a player at as-of snapshot gw (debits bank, adds)")
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--master", required=True, help="path to master_fpl.json")
    p.add_argument("--player-id", required=True)
    p.add_argument("--season", help="override season (default: state.season)")
    p.add_argument("--asof-gw", type=int, help="snapshot gw (default: state.gw)")
    p.add_argument("--teams-map", required=True, help="JSON mapping {TEAM_CODE -> canonical team_id}")
    p.add_argument("--points-hit", type=int, default=0)
    p.set_defaults(func=_cli_buy_from_master)

    # swap-from-master
    p = sub.add_parser("swap-from-master", help="swap out->in atomically; enforces bank>=0 & composition")
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--master", required=True, help="path to master_fpl.json")
    p.add_argument("--out-id", required=True)
    p.add_argument("--in-id", required=True)
    p.add_argument("--season", help="override season (default: state.season)")
    p.add_argument("--asof-gw", type=int, help="snapshot gw (default: state.gw)")
    p.add_argument("--teams-map", required=True, help="JSON mapping {TEAM_CODE -> canonical team_id}")
    p.add_argument("--use-ft", type=int, default=1, help="free transfers to consume (clamped to available)")
    p.add_argument("--points-hit", type=int, default=0, help="for logging only; does not affect bank")
    p.set_defaults(func=_cli_swap_from_master)

    # audit composition
    p = sub.add_parser("audit", help="validate exact 15-man composition (2 GK / 5 DEF / 5 MID / 3 FWD)")
    p.add_argument("path", help="team_state.json path")
    p.set_defaults(func=_cli_audit)

    # Free Hit
    p = sub.add_parser("fh-begin", help="start Free Hit: snapshot bank+squad; optionally log chip usage")
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--gw", type=int, help="chip usage GW; defaults to upcoming")
    p.add_argument("--no-log", action="store_true", help="do not log FH chip usage")
    p.set_defaults(func=_cli_fh_begin)

    p = sub.add_parser("fh-restore", help="restore bank+squad from Free Hit snapshot")
    p.add_argument("path", help="team_state.json path")
    p.set_defaults(func=_cli_fh_restore)

    # migrate-codes
    p = sub.add_parser("migrate-codes", help="migrate legacy team_id short codes -> canonical ids using --teams-map")
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--teams-map", required=True, help="JSON mapping {TEAM_CODE -> canonical team_id}")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--validate", action="store_true", help="with --dry-run, also attempt validation after migration")
    p.set_defaults(func=_cli_migrate_codes)

    # ---- Unavailability commands ----
    p = sub.add_parser("unavail-add", help="add/update an unavailable record")
    p.add_argument("path")
    p.add_argument("--player-id", required=True)
    p.add_argument("--reason", choices=list(UNAVAIL_REASONS.__args__), default="UNKNOWN")
    p.add_argument("--status", choices=list(UNAVAIL_STATUS.__args__), default="OUT")
    p.add_argument("--detail")
    p.add_argument("--source")
    p.add_argument("--until-gw", type=int)
    p.add_argument("--matches", type=int)
    p.set_defaults(func=_cli_unavail_add)

    p = sub.add_parser("unavail-set-until", help="set until_gw for an unavailable record")
    p.add_argument("path")
    p.add_argument("--player-id", required=True)
    p.add_argument("--until-gw", type=int, required=True)
    p.set_defaults(func=_cli_unavail_set_until)

    p = sub.add_parser("unavail-clear", help="remove an unavailable record")
    p.add_argument("path")
    p.add_argument("--player-id", required=True)
    p.set_defaults(func=_cli_unavail_clear)

    p = sub.add_parser("unavail-list", help="list all unavailable records (active flag auto-updated)")
    p.add_argument("path")
    p.set_defaults(func=_cli_unavail_list)

    p = sub.add_parser("unavail-tick", help="recompute active flags vs current gw")
    p.add_argument("path")
    p.set_defaults(func=_cli_unavail_tick)

    return ap

# =========================
# Main
# =========================

def main():
    parser = _build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValidationError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
