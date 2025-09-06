#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator

# =========================
# Core schema (Team State)
# =========================

class ChipState(BaseModel):
    TC: bool
    BB: bool
    FH: bool
    WC1: bool
    WC2: bool

class SquadEntry(BaseModel):
    player_id: str
    name: Optional[str] = None               # human-readable player name
    pos: Literal["GK", "DEF", "MID", "FWD"]
    team_id: str
    sell_price: float = Field(ge=0)
    buy_price: float = Field(ge=0)
    purchase_gw: int = Field(ge=1)

class TeamState(BaseModel):
    season: str             # e.g., "2025-2026" (long)
    gw: int = Field(ge=1)   # current GW (for sell-price computation)
    bank: float = Field(ge=0)
    free_transfers: int = Field(ge=0)
    chips: ChipState
    squad: List[SquadEntry]

    @field_validator("squad")
    @classmethod
    def _size_ok(cls, v: List[SquadEntry]) -> List[SquadEntry]:
        # Allow empty during bootstrap; downstream build/validate can enforce >=1.
        if not (0 <= len(v) <= 15):
            raise ValueError("squad size must be between 0 and 15")
        return v

# =========================
# IO helpers
# =========================

def load_team_state(path: str | Path) -> TeamState:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return TeamState.model_validate_json(f.read())

def save_team_state(state: TeamState, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    js = json.loads(state.model_dump_json())
    with p.open("w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False)

def load_master(master_path: str | Path) -> Dict[str, dict]:
    p = Path(master_path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_teams_map(teams_map_path: Optional[str | Path]) -> Dict[str, str] | None:
    if not teams_map_path:
        return None
    p = Path(teams_map_path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    # Expect {"MCI":"3333cccc", ...}
    return {str(k): str(v) for k, v in raw.items()}

# =========================
# Season / Position helpers
# =========================

_SEASON_SHORT_RE = re.compile(r"^\d{4}-\d{2}$")

def to_short_season(long_or_short: str) -> str:
    """
    Convert '2025-2026' -> '2025-26'. If already short ('2025-26'), return as-is.
    Also accepts '2025/26' -> '2025-26'.
    """
    s = long_or_short.strip()
    if _SEASON_SHORT_RE.match(s):
        return s
    if "/" in s:
        a, b = s.split("/", 1)
        return f"{a}-{b}"
    if len(s) == 9 and s[4] == "-":
        a = s[:4]
        b = s[-2:]
        return f"{a}-{b}"
    return s  # fallback; caller will see missing season if keys don't match

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
    """
    Prefer target_short (e.g., '2025-26'). If missing, fallback to the latest
    available career season. Returns (season_key_used, season_info_dict).
    """
    career = entry.get("career", {}) or {}
    if target_short in career and isinstance(career[target_short], dict):
        return target_short, career[target_short]
    # fallback: latest season by start year if format matches YYYY-YY
    candidates = []
    for k, v in career.items():
        if isinstance(v, dict) and _SEASON_SHORT_RE.match(k):
            try:
                year = int(k[:4])
                candidates.append((year, k, v))
            except Exception:
                continue
    if candidates:
        candidates.sort()  # ascending
        _, k, v = candidates[-1]
        return k, v
    return None, None

# =========================
# Master-FPL utilities
# =========================

def get_master_name(entry: dict, pid: str) -> str:
    # Prefer "name" if present; else "first_name second_name"; else fallback to pid
    if entry.get("name"):
        return str(entry["name"])
    fn = str(entry.get("first_name") or "").strip()
    sn = str(entry.get("second_name") or "").strip()
    full = (fn + " " + sn).strip()
    return full if full else pid

def price_at_or_before(master_entry: dict, season_short: str, gw: int) -> Optional[float]:
    """
    Return the player's price at GW <= gw for the given short season,
    using the latest known GW not exceeding `gw`.
    """
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

def fpl_selling_price(purchase_price: float, current_price_val: float) -> float:
    """
    FPL selling price rule:
      - If price <= purchase, sell price = purchase.
      - If price > purchase, you gain half of the rise, rounded down to 0.1.
    Implementation:
      rises = current - purchase
      sell = purchase + floor((rises / 0.2)) * 0.1
    """
    pp = Decimal(str(purchase_price))
    cp = Decimal(str(current_price_val))
    if cp <= pp:
        return float(pp)
    rises = cp - pp
    steps = (rises / Decimal("0.2")).to_integral_value(rounding=ROUND_DOWN)
    bonus = steps * Decimal("0.1")
    sell = pp + bonus
    return float(sell.quantize(Decimal("0.1"), rounding=ROUND_DOWN))

def get_pos_team_name(master_entry: dict, target_short: str, *, pid: str) -> Tuple[str, str, str, str]:
    """
    Returns (pos, team_code, season_used, name). Falls back to latest available career season.
    Raises a detailed error if pos/team still missing.
    """
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
            f"Found: {season_info.get('fpl_pos') or season_info.get('position')!r}. "
            f"Available seasons: {list_career_seasons(master_entry)}"
        )
    if not team_code:
        raise ValueError(
            f"Missing team code for player {pid} ({name}) in season '{season_used}'. "
            f"Available seasons: {list_career_seasons(master_entry)}"
        )
    return pos, str(team_code).upper(), season_used, name

def resolve_team_id(team_code: str, teams_map: Dict[str, str] | None) -> str:
    """
    Return a canonical team_id for state:
      - If teams_map provided, map code->id.
      - Else, use team_code itself as team_id (contract allows string).
    """
    if teams_map and team_code in teams_map:
        return teams_map[team_code]
    return team_code

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
    team_id = resolve_team_id(team_code, teams_map)

    if override_buy is not None:
        buy_price = float(override_buy)
    else:
        buy_price = price_at_or_before(entry, target_short, purchase_gw)
        if buy_price is None:
            raise ValueError(
                f"No price data for player {player_id} ({name}) at/<= GW {purchase_gw} in season {target_short}. "
                f"Available price seasons: {list(entry.get('prices', {}).keys())}"
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
        team_id=team_id,
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

    # Duplicate/size checks
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
        new_sell = fpl_selling_price(entry.buy_price, cp)
        state.squad[i].sell_price = new_sell

# =========================
# Value helpers
# =========================

def calc_values(
    state: TeamState,
    master: Dict[str, dict],
    season_long: Optional[str] = None,
    current_gw: Optional[int] = None,
) -> tuple[float, float]:
    """
    Return (selling_value, team_value):
      - selling_value = bank + sum(sell_price)
      - team_value    = bank + sum(current market prices from master at current_gw)
    """
    season_long = season_long or state.season
    target_short = to_short_season(season_long)
    current_gw = current_gw or state.gw

    selling_value = state.bank + sum(p.sell_price for p in state.squad)

    cur_sum = 0.0
    for p in state.squad:
        entry = master.get(p.player_id)
        if entry:
            cp = current_price(entry, target_short, current_gw) or p.buy_price
        else:
            cp = p.buy_price
        cur_sum += cp
    team_value = state.bank + cur_sum

    def _round1(x: float) -> float:
        return float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_DOWN))

    return _round1(selling_value), _round1(team_value)

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
    on_duplicate: Literal["error", "skip", "replace", "update-sell"] = "error",
    prune_to_csv: bool = False,
) -> None:
    """
    CSV columns: player_id,purchase_gw
    purchase_gw optional; if missing, defaults to state.gw

    on_duplicate:
      - error        -> raise (previous behavior)
      - skip         -> ignore CSV rows already present in squad
      - replace      -> rebuild entry from master using the CSV purchase_gw and overwrite existing
      - update-sell  -> keep existing buy/purchase_gw/team/pos/name, recompute sell_price from master at current_gw

    prune_to_csv:
      - True removes any players in state.squad that are NOT in the CSV (after applying duplicates policy)
    """
    season_long = season_long or state.season
    current_gw = current_gw or state.gw

    # Read CSV and de-dup within the CSV itself (keep first occurrence)
    wanted_order: List[Tuple[str, int]] = []
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
                # warn and skip duplicate lines in CSV
                print(f"[warn] duplicate player_id in CSV ignored: {pid}")
                continue
            seen_csv.add(pid)
            pgw = int(row["purchase_gw"]) if row.get("purchase_gw") else state.gw
            wanted_order.append((pid, pgw))

    # Build a quick index of current squad
    idx = {p.player_id: i for i, p in enumerate(state.squad)}

    # Apply rows
    for pid, pgw in wanted_order:
        if pid not in master:
            raise ValueError(f"player_id {pid} not in master_fpl.json")

        if pid not in idx:
            # add if space available
            if len(state.squad) >= 15:
                raise ValueError("Cannot exceed 15 players; consider --prune-to-csv or clear-squad first")
            state.squad.append(
                _build_entry_from_master(
                    state, master,
                    player_id=pid,
                    purchase_gw=pgw,
                    teams_map=teams_map,
                    season_long=season_long,
                    current_gw=current_gw,
                )
            )
            idx[pid] = len(state.squad) - 1
            continue

        # Duplicate handling
        if on_duplicate == "error":
            name = state.squad[idx[pid]].name or pid
            raise ValueError(f"player {pid} ({name}) already in squad")
        elif on_duplicate == "skip":
            continue
        elif on_duplicate == "replace":
            state.squad[idx[pid]] = _build_entry_from_master(
                state, master,
                player_id=pid,
                purchase_gw=pgw,
                teams_map=teams_map,
                season_long=season_long,
                current_gw=current_gw,
            )
        elif on_duplicate == "update-sell":
            # keep buy/purchase_gw etc., just recompute sell from master at current_gw
            m = master[pid]
            target_short = to_short_season(season_long)
            cp = current_price(m, target_short, current_gw) or state.squad[idx[pid]].buy_price
            new_sell = fpl_selling_price(state.squad[idx[pid]].buy_price, cp)
            state.squad[idx[pid]].sell_price = new_sell
        else:
            raise ValueError(f"Unknown on_duplicate policy: {on_duplicate}")

    # Optionally prune players not listed in CSV
    if prune_to_csv:
        keep = set(pid for pid, _ in wanted_order)
        state.squad = [p for p in state.squad if p.player_id in keep]

# =========================
# Basic CLI commands
# =========================

def _cli_init(args):
    state = TeamState(
        season=args.season,
        gw=args.gw,
        bank=args.bank,
        free_transfers=args.free_transfers,
        chips=ChipState(TC=False, BB=False, FH=False, WC1=False, WC2=False),
        squad=[],
    )
    save_team_state(state, args.out)
    print(f"Wrote {args.out}")

def _cli_show(args):
    state = load_team_state(args.path)
    print(state.model_dump_json(indent=2))

def _cli_set_bank(args):
    state = load_team_state(args.path)
    state.bank = float(args.bank)
    save_team_state(state, args.path)
    print("OK")

def _cli_set_gw(args):
    state = load_team_state(args.path)
    state.gw = int(args.gw)
    save_team_state(state, args.path)
    print("OK")

def _cli_set_chip(args):
    state = load_team_state(args.path)
    chip = args.chip
    value = args.value
    if chip not in state.chips.model_fields:
        raise ValueError(f"Unknown chip: {chip}")
    setattr(state.chips, chip, value)
    save_team_state(state, args.path)
    print("OK")

def _cli_add_player(args):
    state = load_team_state(args.path)
    if any(p.player_id == args.player_id for p in state.squad):
        raise ValueError(f"player {args.player_id} already in squad")
    if len(state.squad) >= 15:
        raise ValueError("Cannot add more than 15 players to the squad")
    state.squad.append(
        SquadEntry(
            player_id=args.player_id,
            name=args.name,
            pos=args.pos,
            team_id=args.team_id,
            buy_price=float(args.buy_price),
            sell_price=float(args.sell_price if args.sell_price is not None else args.buy_price),
            purchase_gw=int(args.purchase_gw if args.purchase_gw is not None else state.gw),
        )
    )
    save_team_state(state, args.path)
    print("OK")

def _cli_remove_player(args):
    state = load_team_state(args.path)
    before = len(state.squad)
    state.squad = [p for p in state.squad if p.player_id != args.player_id]
    if len(state.squad) == before:
        raise ValueError(f"player {args.player_id} not found")
    save_team_state(state, args.path)
    print("OK")

def _cli_clear_squad(args):
    state = load_team_state(args.path)
    state.squad = []
    save_team_state(state, args.path)
    print("OK (squad cleared)")

# =========================
# Master-aware CLI commands
# =========================

def _cli_add_from_master(args):
    state = load_team_state(args.path)
    master = load_master(args.master)
    teams_map = load_teams_map(args.teams_map)
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
    save_team_state(state, args.path)
    print("OK")

def _cli_seed_from_master(args):
    state = load_team_state(args.path)
    master = load_master(args.master)
    teams_map = load_teams_map(args.teams_map)
    seed_squad_from_master(
        state,
        master,
        csv_list=args.list_csv,
        teams_map=teams_map,
        season_long=args.season or state.season,
        current_gw=args.current_gw or state.gw,
        on_duplicate=args.on_duplicate,
        prune_to_csv=args.prune_to_csv,
    )
    save_team_state(state, args.path)
    print("OK")

def _cli_update_sell_from_master(args):
    state = load_team_state(args.path)
    master = load_master(args.master)
    update_sell_from_master(
        state,
        master,
        season_long=args.season or state.season,
        current_gw=args.current_gw or state.gw,
    )
    save_team_state(state, args.path)
    print("OK")

def _cli_value(args):
    state = load_team_state(args.path)
    master = load_master(args.master)
    sv, tv = calc_values(
        state,
        master,
        season_long=args.season or state.season,
        current_gw=args.current_gw or state.gw,
    )
    print(f"selling_value: {sv:.1f}")
    print(f"team_value:    {tv:.1f}")

# =========================
# Parser
# =========================

def _build_parser():
    ap = argparse.ArgumentParser(
        description="Manage state/team_state.json (with master_fpl.json integration + idempotent seeding)"
    )
    sub = ap.add_subparsers(required=True)

    # init
    p = sub.add_parser("init", help="create a fresh team_state.json")
    p.add_argument("--season", required=True)
    p.add_argument("--gw", type=int, required=True)
    p.add_argument("--bank", type=float, default=0.0)
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

    # add-player (manual)
    p = sub.add_parser("add-player")
    p.add_argument("path")
    p.add_argument("--player-id", required=True)
    p.add_argument("--name", required=False)
    p.add_argument("--pos", required=True, choices=["GK", "DEF", "MID", "FWD"])
    p.add_argument("--team-id", required=True)
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

    # add-from-master (single)
    p = sub.add_parser("add-from-master", help="add one player by player_id using master_fpl.json data")
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--master", required=True, help="path to master_fpl.json")
    p.add_argument("--player-id", required=True)
    p.add_argument("--purchase-gw", type=int, help="defaults to state.gw if omitted")
    p.add_argument("--season", help="override season for lookup (default: state.season)")
    p.add_argument("--current-gw", type=int, help="override current GW (default: state.gw)")
    p.add_argument("--teams-map", help="optional JSON mapping {TEAM_CODE: team_id}")
    p.add_argument("--buy-price", type=float, help="override buy_price")
    p.add_argument("--sell-price", type=float, help="override sell_price")
    p.set_defaults(func=_cli_add_from_master)

    # seed-squad-from-master (bulk from CSV)
    p = sub.add_parser(
        "seed-squad-from-master",
        help="seed squad from CSV with columns: player_id,purchase_gw",
    )
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--master", required=True, help="path to master_fpl.json")
    p.add_argument("--list-csv", required=True, help="CSV with player_id[,purchase_gw]")
    p.add_argument("--season", help="override season for lookup (default: state.season)")
    p.add_argument("--current-gw", type=int, help="override current GW (default: state.gw)")
    p.add_argument("--teams-map", help="optional JSON mapping {TEAM_CODE: team_id}")
    p.add_argument("--on-duplicate", choices=["error","skip","replace","update-sell"], default="error")
    p.add_argument("--prune-to-csv", action="store_true")
    p.set_defaults(func=_cli_seed_from_master)

    # update-sell-from-master (recompute sell_price for all players)
    p = sub.add_parser(
        "update-sell-from-master", help="recompute sell_price for all players from master_fpl.json"
    )
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--master", required=True, help="path to master_fpl.json")
    p.add_argument("--season", help="override season for lookup (default: state.season)")
    p.add_argument("--current-gw", type=int, help="override current GW (default: state.gw)")
    p.set_defaults(func=_cli_update_sell_from_master)

    # value (selling_value and team_value)
    p = sub.add_parser("value", help="print selling_value and team_value")
    p.add_argument("path", help="team_state.json path")
    p.add_argument("--master", required=True, help="path to master_fpl.json")
    p.add_argument("--season", help="override season for lookup (default: state.season)")
    p.add_argument("--current-gw", type=int, help="override current GW (default: state.gw)")
    p.set_defaults(func=_cli_value)

    return ap

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
