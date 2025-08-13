#!/usr/bin/env python3
"""
three_gw_optimizer.py — v2 (per-GW JSON prices, bench-fodder allowed)

- Maximizes EV over a 3-GW horizon (or any horizon you pass).
- Captain each GW by max yield (objective doubles the captain’s EV).
- DGWs handled by summing XP per player within each GW (independent fixtures).
- Budget enforced using **purchase price at gw_start** (tenths of £m).
- Supports your JSON price registry with per-GW prices.
- Keeps near-zero availability players (no filtering) — useful for cheap bench.

Inputs:
  --xp-csv <path>           expected_points.csv (needs: season, gw_orig, player_id, team_id, pos, player, exp_points_total)
  --prices-json <path>      JSON registry (your format)
  --season 2024-2025
  --gw-start 1
  --horizon 3
  --budget 1000             # tenths of £m (e.g., £100.0m => 1000)
  --out-dir plans/v2

Dependencies: pandas, numpy, pulp
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List
import json
import numpy as np
import pandas as pd
import pulp

POS_ALIASES = {"GKP":"GK","GK":"GK","DEF":"DEF","D":"DEF","MID":"MID","M":"MID","FWD":"FWD","F":"FWD"}

SQUAD_QUOTAS = {"GK":2,"DEF":5,"MID":5,"FWD":3}
XI_MIN = {"GK":1,"DEF":3,"MID":2,"FWD":1}
XI_MAX = {"GK":1,"DEF":5,"MID":5,"FWD":3}

def norm_pos(s: pd.Series) -> pd.Series:
    x = s.fillna("").str.upper().str[:3]
    return x.map(POS_ALIASES).fillna(x)

# ────────────── Data loading ──────────────

def load_xp(path: Path, season: str, gw_start: int, horizon: int) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date_played"], low_memory=False)
    req = {"season","gw_orig","player_id","team_id","pos","player","exp_points_total"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"XP missing: {miss}")
    df["season"] = df["season"].astype(str)
    df = df[df["season"] == str(season)].copy()
    gw_end = gw_start + horizon - 1
    df = df[(df["gw_orig"] >= gw_start) & (df["gw_orig"] <= gw_end)].copy()
    # string ids (your ids can be hex-like)
    df["player_id"] = df["player_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)
    df["pos"] = norm_pos(df["pos"])
    # sum DGW fixtures within GW
    xp = (df.groupby(["player_id","team_id","pos","player","gw_orig"], as_index=False)
            ["exp_points_total"].sum()
            .rename(columns={"exp_points_total":"xp"}))
    return xp

def _pick_price_entry(gw_dict: dict, gw: int):
    sgw = str(gw)
    if sgw in gw_dict: return gw_dict[sgw]
    keys = sorted(int(k) for k in gw_dict.keys() if k.isdigit() and int(k) <= gw)
    return gw_dict[str(keys[-1])] if keys else None

def load_registry_prices_json(path: Path, gws: List[int]) -> pd.DataFrame:
    """
    Returns long DF with per-GW prices:
      player_id, gw, now_cost (tenths), pos_guess, team_code
    """
    reg = json.loads(Path(path).read_text("utf-8"))
    players = reg.get("players", {})
    rows = []
    for pid, pdata in players.items():
        gwd = pdata.get("gw", {})
        for gw in gws:
            ent = _pick_price_entry(gwd, gw)
            if not ent: 
                continue
            price = ent.get("price", None)
            pos   = ent.get("fpl_pos", None)
            tcode = ent.get("team", None)
            if price is None: 
                continue
            rows.append({
                "player_id": str(pid),
                "gw": int(gw),
                "now_cost": int(round(float(price)*10)),  # £m → tenths
                "pos_guess": (str(pos).upper() if pos is not None else None),
                "team_code": tcode
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No prices found in JSON for requested GWs.")
    return df

# --- replace your prepare_universe with this ---
def prepare_universe(xp: pd.DataFrame,
                     price_long: pd.DataFrame,
                     gw_start: int,
                     gws: list[int]) -> tuple[pd.DataFrame, dict[str,dict[int,float]]]:
    """
    Universe from prices at gw_start; XP is optional (0 if missing).
    team_id from XP if available else fallback to team_code; pos from XP else price pos_guess.
    """
    # 1) Purchase-eligible set at gw_start
    purch = (price_long[price_long["gw"] == gw_start]
             .groupby("player_id", as_index=False)
             .last())  # one row per player_id with now_cost, pos_guess, team_code

    purch["player_id"] = purch["player_id"].astype(str)
    purch = purch.rename(columns={"now_cost":"purchase_cost"})

    # 2) Canonical attrs from XP (where available)
    attrs = (xp.sort_values("gw_orig")
               .groupby("player_id", as_index=False)
               .first()[["player_id","team_id","pos","player"]])
    attrs["player_id"] = attrs["player_id"].astype(str)
    attrs["team_id"]   = attrs["team_id"].astype(str)

    # 3) Merge -> base universe
    base = purch.merge(attrs, on="player_id", how="left")
    # Fallbacks
    base["pos"] = base["pos"].fillna(base.get("pos_guess")).fillna("MID")
    base["pos"] = base["pos"].astype(str).str.upper().str[:3].replace({"GKP":"GK","D":"DEF","M":"MID","F":"FWD"})
    # If team_id missing, fall back to team_code (string surrogate ensures team cap still works)
    base["team_id"] = base["team_id"].fillna(base.get("team_code")).fillna("UNK")
    base["player"]  = base["player"].fillna(base["player_id"])

    # 4) Build per-GW XP dict (missing -> 0.0)
    xp_pivot = xp.pivot_table(index="player_id", columns="gw_orig", values="xp", aggfunc="sum", fill_value=0.0)
    xp_pivot.index = xp_pivot.index.astype(str)
    xp_gw = {}
    for pid in base["player_id"]:
        pid = str(pid)
        xp_gw[pid] = {}
        for w in gws:
            xp_gw[pid][int(w)] = float(xp_pivot.loc[pid, w]) if (pid in xp_pivot.index and w in xp_pivot.columns) else 0.0

    # Sanity: keep only rows with purchase_cost and a valid position
    base = base.dropna(subset=["purchase_cost"])
    return base.reset_index(drop=True), xp_gw


# ────────────── Optimizer ──────────────

def solve_three_gw(base: pd.DataFrame,
                   xp_gw: Dict[str,Dict[int,float]],
                   gws: List[int],
                   budget: int,
                   out_dir: Path | None = None):
    players = list(base["player_id"])
    pos_of   = dict(zip(base["player_id"], base["pos"]))
    team_of  = dict(zip(base["player_id"], base["team_id"]))
    name_of  = dict(zip(base["player_id"], base["player"]))
    cost_of  = dict(zip(base["player_id"], base["purchase_cost"]))

    P_by_pos = {k: set(base.loc[base["pos"]==k, "player_id"]) for k in ["GK","DEF","MID","FWD"]}
    teams = sorted(set(base["team_id"]))

    m = pulp.LpProblem("FPL_Three_GW_Optimizer_v2", pulp.LpMaximize)

    # Vars
    y = pulp.LpVariable.dicts("squad", players, 0, 1, cat="Binary")
    s = {(p,w): pulp.LpVariable(f"start_{p}_gw{w}", 0, 1, cat="Binary") for p in players for w in gws}
    c = {(p,w): pulp.LpVariable(f"capt_{p}_gw{w}", 0, 1, cat="Binary") for p in players for w in gws}

    # Objective: XI EV + captain EV (adds one copy)
    m += pulp.lpSum((xp_gw.get(p, {}).get(w, 0.0)) * (s[(p,w)] + c[(p,w)]) for p in players for w in gws)

    # Budget (purchase at gw_start)
    m += pulp.lpSum(cost_of[p] * y[p] for p in players) <= budget, "Budget"

    # Squad size & quotas
    m += pulp.lpSum(y[p] for p in players) == 15, "SquadSize15"
    for pos, quota in SQUAD_QUOTAS.items():
        m += pulp.lpSum(y[p] for p in P_by_pos.get(pos, set())) == quota, f"SquadQuota_{pos}"

    # Team limit ≤3
    for t in teams:
        m += pulp.lpSum(y[p] for p in players if team_of[p]==t) <= 3, f"TeamLimit_{t}"

    # Weekly constraints
    for w in gws:
        for p in players:
            m += s[(p,w)] <= y[p], f"StartSubset_{p}_{w}"
            m += c[(p,w)] <= s[(p,w)], f"CaptainSubset_{p}_{w}"
        m += pulp.lpSum(s[(p,w)] for p in players) == 11, f"XI_size_{w}"
        for pos in ["GK","DEF","MID","FWD"]:
            Ppos = P_by_pos.get(pos, set())
            m += pulp.lpSum(s[(p,w)] for p in Ppos) >= XI_MIN[pos], f"XI_min_{pos}_{w}"
            m += pulp.lpSum(s[(p,w)] for p in Ppos) <= XI_MAX[pos], f"XI_max_{pos}_{w}"
        m += pulp.lpSum(c[(p,w)] for p in players) == 1, f"OneCaptain_{w}"

    def _must_have(base, pos, need): 
        have = (base["pos"]==pos).sum()
        if have < need:
            raise RuntimeError(f"Infeasible: need {need} {pos}s but base has {have}. "
                            f"Check XP merge & price coverage for GW {gws}.")

    # squad quotas
    _require = {"GK":2, "DEF":5, "MID":5, "FWD":3}
    for pos, need in _require.items(): _must_have(base, pos, need)

    # min XI per week
    xi_min = {"GK":1, "DEF":3, "MID":2, "FWD":1}
    for pos, need in xi_min.items(): _must_have(base, pos, need)


    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        raise RuntimeError(f"Solver status: {pulp.LpStatus[m.status]}")

    squad = [p for p in players if y[p].value() > 0.5]
    xi_by_w = {w: [p for p in players if s[(p,w)].value() > 0.5] for w in gws}
    c_by_w  = {w: [p for p in players if c[(p,w)].value() > 0.5][0] for w in gws}

    # Save
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "player_id": p, "player": name_of[p], "pos": pos_of[p], "team_id": team_of[p],
            "purchase_cost": cost_of[p]
        } for p in squad]).sort_values(["pos","purchase_cost","player"]).to_csv(out_dir/"squad.csv", index=False)
        for w in gws:
            dfw = pd.DataFrame([{
                "gw": w, "player_id": p, "player": name_of[p], "pos": pos_of[p],
                "team_id": team_of[p], "starter": 1, "captain": int(p==c_by_w[w]),
                "xp": xp_gw.get(p, {}).get(w, 0.0)
            } for p in xi_by_w[w]]).sort_values(["captain","pos","xp"], ascending=[False, True, False])
            dfw.to_csv(out_dir/f"xi_week{w}.csv", index=False)

    # Console summary
    total_obj = pulp.value(m.objective)
    spent = sum(cost_of[p] for p in squad)
    fmt = lambda x: f"£{x/10:.1f}m"
    print(f"\nOPTIMAL EV over GWs {gws}: {total_obj:.2f} pts\n")
    print(f"SQUAD (15) — Cost {fmt(spent)} / {fmt(budget)}")
    for pos in ["GK","DEF","MID","FWD"]:
        rows = [p for p in squad if pos_of[p]==pos]
        print(f" {pos} ({len(rows)}): " + ", ".join(f"{name_of[p]} ({fmt(cost_of[p])})" for p in rows))
    for w in gws:
        xi = xi_by_w[w]; cap = c_by_w[w]
        ev_xi = sum(xp_gw.get(p, {}).get(w, 0.0) for p in xi)
        ev_cap = xp_gw.get(cap, {}).get(w, 0.0)
        print(f"\nGW {w}: XI EV={ev_xi:.2f} | Captain {name_of[cap]} (+{ev_cap:.2f})")
        for pos in ["GK","DEF","MID","FWD"]:
            print(f"  {pos}: {sum(1 for p in xi if pos_of[p]==pos)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xp-csv", type=Path, required=True)
    ap.add_argument("--prices-json", type=Path, required=True)
    ap.add_argument("--season", type=str, required=True)
    ap.add_argument("--gw-start", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--budget", type=int, default=1000)  # tenths
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    gws = list(range(args.gw_start, args.gw_start + args.horizon))
    xp = load_xp(args.xp_csv, args.season, args.gw_start, args.horizon)
    price_long = load_registry_prices_json(args.prices_json, gws)
    base, xp_gw = prepare_universe(xp, price_long, args.gw_start, gws)
    # NOTE: we keep near-zero-availability players; no filtering here by p1/p60.

    solve_three_gw(base, xp_gw, gws, args.budget, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
