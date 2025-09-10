import os
import csv
import tempfile
import pytest
from scripts.optimizers import team_state as ts

SEASON_LONG = "2025-2026"
SEASON_SHORT = "2025-26"

def _write_csv(rows):
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["player_id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path

def _empty_state(gw=3, budget=100.0):
    return ts.TeamState(
        season=SEASON_LONG,
        gw=gw,
        initial_budget=budget,
        bank=budget,
        value_liquidation=budget,
        value_market=budget,
        free_transfers=1,
        chips=ts.ChipState(TC=True, BB=True, FH=True, WC1=True, WC2=True),
        chips_log=[],
        transfer_log=[],
        fh_active=False,
        fh_backup_bank=None,
        fh_backup_squad=None,
        squad=[],
    )

def _master_basic():
    # Two players to cover simple flows
    return {
        "7b865b65": {
            "name": "Jarrod Bowen",
            "career": { SEASON_SHORT: {"team": "WHU", "fpl_pos": "MID"} },
            "prices": { SEASON_SHORT: {"1": 8.0, "2": 7.9, "3": 7.8, "4": 7.9} },
        },
        "aaaa1111": {
            "name": "Some Defender",
            "career": { SEASON_SHORT: {"team": "ARS", "fpl_pos": "DEF"} },
            "prices": { SEASON_SHORT: {"1": 5.0, "2": 5.0, "3": 5.1, "4": 5.2} },
        },
    }

def _master_full_15():
    # Build a valid 15-player pool to test composition:
    # 2 GK, 5 DEF, 5 MID, 3 FWD (use stable prices at gw3/gw4)
    m = {}
    def add(pid, pos, team, p3, p4):
        m[pid] = {
            "name": pid,
            "career": { SEASON_SHORT: {"team": team, "fpl_pos": pos} },
            "prices": { SEASON_SHORT: {"3": p3, "4": p4} },
        }
    # GK x2
    add("gk1","MID","AAA",4.5,4.5)  # NOTE: set correct pos below; placeholder
    add("gk2","MID","BBB",4.0,4.0)
    # DEF x5
    for i,p in enumerate([4.0,4.5,5.0,5.0,5.5], start=1):
        add(f"def{i}","DEF","DEF",p,p)
    # MID x5
    for i,p in enumerate([6.0,6.5,7.0,7.5,8.0], start=1):
        add(f"mid{i}","MID","MID",p,p)
    # FWD x3
    for i,p in enumerate([6.5,7.0,7.5], start=1):
        add(f"fwd{i}","FWD","FWD",p,p)
    # Fix GK positions
    m["gk1"]["career"][SEASON_SHORT]["fpl_pos"] = "GK"
    m["gk2"]["career"][SEASON_SHORT]["fpl_pos"] = "GK"
    return m

def test_swap_bank_non_negative():
    master = _master_basic()
    state = _empty_state(gw=4, budget=10.0)
    # Seed one MID (Bowen @7.9 at gw4) and set bank small so swap fails
    state.squad.append(ts.SquadEntry(
        player_id="7b865b65", name="Jarrod Bowen", pos="MID", team_id="WHU",
        buy_price=7.8, sell_price=7.9, purchase_gw=3
    ))
    state.bank = 0.0  # no cash
    # Try to swap to DEF costing 5.2 at gw4 => proceeds 7.9, cost 5.2 -> bank would be +2.7 (OK)
    # To force failure, make bank negative by crafting cost > proceeds
    master["aaaa1111"]["prices"][SEASON_SHORT]["4"] = 9.0
    with pytest.raises(ValueError):
        ts.swap_transfer_from_master(state, master, out_id="7b865b65", in_id="aaaa1111", season_long=SEASON_LONG, asof_gw=4)

def test_swap_composition_enforced():
    master = _master_full_15()
    state = _empty_state(gw=3, budget=1000.0)  # enough bank to not be limiting
    # Build valid 15-man squad at gw3
    # 2 GK
    state.squad.append(ts.SquadEntry(
    player_id="gk1", name="gk1", pos="GK", team_id="AAA",
    buy_price=4.5, sell_price=4.5, purchase_gw=3
    ))
    state.squad.append(ts.SquadEntry(
        player_id="gk2", name="gk2", pos="GK", team_id="BBB",
        buy_price=4.0, sell_price=4.0, purchase_gw=3
    ))
    # 5 DEF
    # 5 DEF
    for i in range(1, 6):
        p = 4.0 if i == 1 else 4.5 if i == 2 else 5.0 if i in (3, 4) else 5.5
        state.squad.append(ts.SquadEntry(
            player_id=f"def{i}", name=f"def{i}", pos="DEF", team_id="DEF",
            buy_price=p, sell_price=p, purchase_gw=3
        ))

    # 5 MID
    for i, p in enumerate([6.0, 6.5, 7.0, 7.5, 8.0], start=1):
        state.squad.append(ts.SquadEntry(
            player_id=f"mid{i}", name=f"mid{i}", pos="MID", team_id="MID",
            buy_price=p, sell_price=p, purchase_gw=3
        ))

    # 3 FWD
    for i, p in enumerate([6.5, 7.0, 7.5], start=1):
        state.squad.append(ts.SquadEntry(
            player_id=f"fwd{i}", name=f"fwd{i}", pos="FWD", team_id="FWD",
            buy_price=p, sell_price=p, purchase_gw=3
        ))

    # Audit should pass
    ts.validate_exact_composition(state)

    # Attempt a swap that would break composition: swap out a DEF and bring in a MID
    # Create a new MID in master
    master["new_mid"] = {
        "name": "new_mid",
        "career": { SEASON_SHORT: {"team": "MID", "fpl_pos": "MID"} },
        "prices": { SEASON_SHORT: {"3": 5.0, "4": 5.0} },
    }

    with pytest.raises(ValueError):
        ts.swap_transfer_from_master(state, master, out_id="def1", in_id="new_mid", season_long=SEASON_LONG, asof_gw=3)

def test_fh_restore_bank_and_squad():
    master = _master_basic()
    state = _empty_state(gw=3, budget=100.0)
    # Seed Bowen at gw3
    state.squad.append(ts.SquadEntry(
        player_id="7b865b65", name="Jarrod Bowen", pos="MID", team_id="WHU",
        buy_price=7.8, sell_price=7.8, purchase_gw=3
    ))
    state.bank = 92.2

    # Begin FH and mutate state via buy
    ts.fh_begin(state, gw=4, chips_log=True)
    assert state.fh_active
    # Buy DEF at gw3 price 5.1
    ts.buy_player_from_master(state, master, player_id="aaaa1111", season_long=SEASON_LONG, asof_gw=3, log=False)
    assert any(p.player_id=="aaaa1111" for p in state.squad)
    # Restore FH
    ts.fh_restore(state)
    assert not any(p.player_id=="aaaa1111" for p in state.squad)
    assert state.bank == pytest.approx(92.2, 1e-6)
    assert not state.fh_active

def test_prices_snapshot_lock_to_last_concluded():
    master = _master_basic()
    state = _empty_state(gw=3)
    # Put Bowen only
    state.squad.append(ts.SquadEntry(
        player_id="7b865b65", name="Jarrod Bowen", pos="MID", team_id="WHU",
        buy_price=7.8, sell_price=7.8, purchase_gw=3
    ))
    # No gw given => lock to max known (4)
    ts.prices_snapshot_at_gw(state, master, season_long=SEASON_LONG, snapshot_gw=None)
    assert state.gw == 4
    bowen = [p for p in state.squad if p.player_id=="7b865b65"][0]
    assert bowen.sell_price == pytest.approx(7.9, 1e-6)
