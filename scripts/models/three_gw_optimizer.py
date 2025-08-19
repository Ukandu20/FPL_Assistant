#!/usr/bin/env python3
"""
three_gw_optimizer.py — v3 (transfer-aware, group XP constraint, per-GW prices, universe debug)

What it does
------------
• Maximizes total expected points (EV) over a multi-GW horizon.
• Picks a 15-man squad each GW, a starting XI, and exactly one captain (max yield).
• Allows ≤1 transfer between consecutive GWs (can also roll to 0).
• Budget enforced per GW using per-GW prices from your JSON registry.
• Team limit ≤3 per real team, formation bounds each GW (1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD).
• Group XP constraint: at least K starters with XP ≥ tau each GW (defaults: tau=2.0, K=9).
• Optional bench EV weight; optional epsilon spend tie-breaker.
• Writes universe_debug.csv (by GW: availability, price, XP), squad_week*.csv, xi_week*.csv.

Inputs
------
--xp-csv           expected_points.csv  (season, gw_orig, player_id, team_id, pos, player, exp_points_total)
--prices-json      registry JSON (your per-GW format)
--season           e.g., 2024-2025
--gw-start         integer GW start
--horizon          number of GWs (>=1)
--budget           tenths of £m (e.g., 1000 = £100.0m)

Recommended tunables
--------------------
--min-spend-ratio  0.90       # spend ≥90% of budget per GW (guardrail vs under-spend)
--bench-weight     0.05       # small EV for bench to prefer playable fodder (all bench slots)
--epsilon-spend    0.001      # tiny tie-breaker to prefer spending when EV ties
--high-xp-threshold 2.0       # tau for "high XP" starters
--min-high-xp-starters 9      # K starters each GW with XP ≥ tau
--max-transfers-per-step 1    # ≤1 transfer between consecutive GWs (can be 0)

Notes
-----
• Availability is enforced: a player can be in the GW squad only if a price exists for that GW.
• Price/XP joins are by your hex-like player_id (you confirmed they match).
• Team cap falls back to team_code if team_id missing in XP for a player/GW.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pulp

# ───────────────────────── Config ─────────────────────────

POS_ALIASES = {"GKP":"GK","GK":"GK","DEF":"DEF","D":"DEF","MID":"MID","M":"MID","FWD":"FWD","F":"FWD"}
SQUAD_QUOTAS = {"GK":2,"DEF":5,"MID":5,"FWD":3}
XI_MIN = {"GK":1,"DEF":3,"MID":2,"FWD":1}
XI_MAX = {"GK":1,"DEF":5,"MID":5,"FWD":3}

def norm_pos(s: pd.Series) -> pd.Series:
    x = s.fillna("").str.upper().str[:3]
    return x.map(POS_ALIASES).fillna(x)

# ───────────────────────── Loaders ─────────────────────────

def load_xp(path: Path, season: str, gw_start: int, horizon: int) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date_played"], low_memory=False)
    need = {"season","gw_orig","player_id","team_id","pos","player","exp_points_total"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"XP file missing columns: {miss}")
    df["season"] = df["season"].astype(str)
    df = df[df["season"] == str(season)].copy()
    gw_end = gw_start + horizon - 1
    df = df[(df["gw_orig"] >= gw_start) & (df["gw_orig"] <= gw_end)].copy()
    df["player_id"] = df["player_id"].astype(str)
    df["team_id"]   = df["team_id"].astype(str)
    df["pos"] = norm_pos(df["pos"])
    # sum any DGW fixtures within a GW to a single XP per player/GW
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
    """Long frame: player_id, gw, now_cost (tenths), pos_guess, team_code."""
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
                "now_cost": int(round(float(price)*10)),  # £m -> tenths
                "pos_guess": (str(pos).upper() if pos is not None else None),
                "team_code": tcode
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No prices found in JSON for requested GWs.")
    return df

# ───────────────────────── Universe ─────────────────────────

def build_universe(
    xp: pd.DataFrame,
    price_long: pd.DataFrame,
    gws: List[int]
) -> Tuple[pd.DataFrame, Dict[str,Dict[int,float]], Dict[int,Dict[str,int]], Dict[int,Dict[str,int]]]:
    """
    Returns:
      base: one row per player with attrs (player_id, player, pos, team_id)
      xp_gw[w][p]: expected points for player p in GW w
      cost_w[w][p]: cost (tenths) for player p in GW w (only if available)
      avail_w[w][p]: 1 if player has a price for GW w else 0
    """
    # canonical attrs from XP if present
    attrs = (xp.sort_values("gw_orig")
               .groupby("player_id", as_index=False)
               .first()[["player_id","team_id","pos","player"]])
    attrs["player_id"] = attrs["player_id"].astype(str)
    attrs["team_id"]   = attrs["team_id"].astype(str)
    attrs["pos"] = norm_pos(attrs["pos"])

    # all players that have a price in ANY requested GW
    purch = (price_long[price_long["gw"].isin(gws)]
             .groupby("player_id", as_index=False)
             .last()[["player_id","pos_guess","team_code"]])
    purch["player_id"] = purch["player_id"].astype(str)

    base = purch.merge(attrs, on="player_id", how="left")
    base["pos"] = base["pos"].fillna(base["pos_guess"]).fillna("MID")
    base["pos"] = norm_pos(base["pos"])
    base["team_id"] = base["team_id"].fillna(base["team_code"]).fillna("UNK").astype(str)
    base["player"]  = base["player"].fillna(base["player_id"])

    # Per-GW XP dict
    xp_pivot = xp.pivot_table(index="player_id", columns="gw_orig", values="xp", aggfunc="sum", fill_value=0.0)
    xp_pivot.index = xp_pivot.index.astype(str)
    xp_gw: Dict[str, Dict[int, float]] = {}
    for pid in base["player_id"]:
        pid = str(pid)
        xp_gw[pid] = {int(w): float(xp_pivot.loc[pid, w]) if (pid in xp_pivot.index and w in xp_pivot.columns) else 0.0
                      for w in gws}

    # Per-GW cost and availability
    cost_w: Dict[int, Dict[str, int]]  = {w: {} for w in gws}
    avail_w: Dict[int, Dict[str, int]] = {w: {} for w in gws}
    for _, r in price_long.iterrows():
        w = int(r["gw"]); p = str(r["player_id"])
        if w in cost_w:
            cost_w[w][p]  = int(r["now_cost"])
            avail_w[w][p] = 1
    # fill missing availability as 0
    for w in gws:
        for p in base["player_id"]:
            if p not in avail_w[w]:
                avail_w[w][p] = 0

    return base.reset_index(drop=True), xp_gw, cost_w, avail_w

# ───────────────────────── Optimizer ─────────────────────────

def solve_transfer_aware(
    base: pd.DataFrame,
    xp_gw: Dict[str,Dict[int,float]],
    cost_w: Dict[int,Dict[str,int]],
    avail_w: Dict[int,Dict[str,int]],
    gws: List[int],
    budget: int,
    epsilon_spend: float,
    bench_weight: float,
    min_spend_ratio: float,
    high_xp_threshold: float,
    min_high_xp_starters: int,
    max_transfers_per_step: int,
    debug: bool,
    out_dir: Path | None = None
):
    players = list(base["player_id"])
    pos_of   = dict(zip(base["player_id"], base["pos"]))
    team_of  = dict(zip(base["player_id"], base["team_id"]))
    name_of  = dict(zip(base["player_id"], base["player"]))

    P_by_pos = {k: set(base.loc[base["pos"]==k, "player_id"]) for k in ["GK","DEF","MID","FWD"]}
    teams = sorted(set(base["team_id"]))

    # sanity per week: enough supply by position & availability
    for w in gws:
        for pos, q in SQUAD_QUOTAS.items():
            have = sum(1 for p in P_by_pos[pos] if avail_w[w].get(p,0)==1)
            if have < q:
                raise RuntimeError(f"Infeasible pool: GW {w} needs {q} {pos} but only {have} available with prices.")

    if debug:
        print("Players in base:", len(base))
        print("By position:\n", base["pos"].value_counts().to_string())
        print("Distinct teams:", len(teams))
        for w in gws:
            top = (pd.DataFrame({"player_id": players, "xp": [xp_gw[p].get(w,0.0) for p in players]})
                   .merge(base[["player_id","player","pos","team_id"]], on="player_id", how="left")
                   .sort_values("xp", ascending=False).head(12))
            print(f"\nTop XP GW {w}:\n", top.to_string(index=False))

    # Model
    m = pulp.LpProblem("FPL_Three_GW_Optimizer_v3", pulp.LpMaximize)

    # Vars
    y = {(p,w): pulp.LpVariable(f"squad_{p}_gw{w}", 0, 1, cat="Binary") for p in players for w in gws}
    s = {(p,w): pulp.LpVariable(f"start_{p}_gw{w}", 0, 1, cat="Binary") for p in players for w in gws}
    c = {(p,w): pulp.LpVariable(f"capt_{p}_gw{w}", 0, 1, cat="Binary") for p in players for w in gws}
    invar  = {(p,w): pulp.LpVariable(f"in_{p}_gw{w}", 0, 1, cat="Binary")  for p in players for w in gws[1:]}
    outvar = {(p,w): pulp.LpVariable(f"out_{p}_gw{w}", 0, 1, cat="Binary") for p in players for w in gws[1:]}

    # Objective
    obj_xi   = pulp.lpSum(xp_gw[p].get(w, 0.0) * s[(p,w)] for p in players for w in gws)
    obj_capt = pulp.lpSum(xp_gw[p].get(w, 0.0) * c[(p,w)] for p in players for w in gws)
    obj_bench= bench_weight * pulp.lpSum(xp_gw[p].get(w, 0.0) * (y[(p,w)] - s[(p,w)]) for p in players for w in gws)
    obj_spend= epsilon_spend * pulp.lpSum(cost_w[w].get(p,0) * y[(p,w)] for p in players for w in gws)
    m += obj_xi + obj_capt + obj_bench + obj_spend

    # Weekly constraints
    for w in gws:
        # availability: can only be in squad if priced that GW
        for p in players:
            m += y[(p,w)] <= avail_w[w].get(p,0), f"Avail_{p}_{w}"
        # budget per week: min & max
        m += pulp.lpSum(cost_w[w].get(p,0) * y[(p,w)] for p in players) <= budget,     f"BudgetMax_{w}"
        if min_spend_ratio > 0:
            m += pulp.lpSum(cost_w[w].get(p,0) * y[(p,w)] for p in players) >= int(np.floor(budget * min_spend_ratio)), f"BudgetMin_{w}"
        # squad size & quotas
        m += pulp.lpSum(y[(p,w)] for p in players) == 15, f"SquadSize15_{w}"
        for pos, quota in SQUAD_QUOTAS.items():
            Ppos = [p for p in players if pos_of[p]==pos]
            m += pulp.lpSum(y[(p,w)] for p in Ppos) == quota, f"SquadQuota_{pos}_{w}"
        # team limit ≤3
        for t in teams:
            m += pulp.lpSum(y[(p,w)] for p in players if team_of[p]==t) <= 3, f"TeamLimit_{t}_{w}"
        # starters subset & formation
        for p in players:
            m += s[(p,w)] <= y[(p,w)], f"StartSubset_{p}_{w}"
            m += c[(p,w)] <= s[(p,w)], f"CaptainSubset_{p}_{w}"
        m += pulp.lpSum(s[(p,w)] for p in players) == 11, f"XI_size_{w}"
        for pos in ["GK","DEF","MID","FWD"]:
            Ppos = [p for p in players if pos_of[p]==pos]
            m += pulp.lpSum(s[(p,w)] for p in Ppos) >= XI_MIN[pos], f"XI_min_{pos}_{w}"
            m += pulp.lpSum(s[(p,w)] for p in Ppos) <= XI_MAX[pos], f"XI_max_{pos}_{w}"
        # exactly one captain
        m += pulp.lpSum(c[(p,w)] for p in players) == 1, f"OneCaptain_{w}"
        # group high-XP starters
        elig = [p for p in players if xp_gw[p].get(w,0.0) >= float(high_xp_threshold)]
        if len(elig) < min_high_xp_starters:
            raise RuntimeError(f"Infeasible: GW {w} has only {len(elig)} players with XP ≥ {high_xp_threshold}, "
                               f"but requires {min_high_xp_starters} starters.")
        m += pulp.lpSum(s[(p,w)] for p in elig) >= int(min_high_xp_starters), f"MinHighXPStarters_{w}"

    # Transfers between weeks (≤1, can be 0)
    for w_prev, w in zip(gws[:-1], gws[1:]):
        for p in players:
            m += y[(p,w)] - y[(p,w_prev)] <= invar[(p,w)],  f"InFlag_{p}_{w}"
            m += y[(p,w_prev)] - y[(p,w)] <= outvar[(p,w)], f"OutFlag_{p}_{w}"
        # balance and limit
        m += pulp.lpSum(invar[(p,w)]  for p in players) == pulp.lpSum(outvar[(p,w)] for p in players), f"SwapBalance_{w}"
        m += pulp.lpSum(invar[(p,w)]  for p in players) <= max_transfers_per_step, f"MaxTransfers_{w}"

    # Solve
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        raise RuntimeError(f"Solver status: {pulp.LpStatus[m.status]}")

    # Extract
    xi_by_w  = {w: [p for p in players if s[(p,w)].value() > 0.5] for w in gws}
    cap_by_w = {w: [p for p in players if c[(p,w)].value() > 0.5][0] for w in gws}
    squad_by_w = {w: [p for p in players if y[(p,w)].value() > 0.5] for w in gws}

    # Save outputs
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        # universe debug (long)
        rows = []
        for p in players:
            for w in gws:
                rows.append({
                    "player_id": p,
                    "player": name_of[p],
                    "pos": pos_of[p],
                    "team_id": team_of[p],
                    "gw": w,
                    "available": avail_w[w].get(p,0),
                    "cost": cost_w[w].get(p, np.nan),
                    "xp": xp_gw[p].get(w, 0.0)
                })
        pd.DataFrame(rows).to_csv(out_dir/"universe_debug.csv", index=False)

        # per-week squad and XI
        for w in gws:
            squad = squad_by_w[w]
            df_squad = pd.DataFrame([{
                "gw": w,
                "player_id": p,
                "player": name_of[p],
                "pos": pos_of[p],
                "team_id": team_of[p],
                "now_cost": cost_w[w].get(p, np.nan)
            } for p in squad]).sort_values(["pos","now_cost","player"])
            df_squad.to_csv(out_dir/f"squad_week{w}.csv", index=False)

            xi = xi_by_w[w]; cap = cap_by_w[w]
            df_xi = pd.DataFrame([{
                "gw": w,
                "player_id": p,
                "player": name_of[p],
                "pos": pos_of[p],
                "team_id": team_of[p],
                "starter": 1,
                "captain": int(p==cap),
                "xp": xp_gw[p].get(w, 0.0),
                "now_cost": cost_w[w].get(p, np.nan)
            } for p in xi]).sort_values(["captain","pos","xp"], ascending=[False, True, False])
            df_xi.to_csv(out_dir/f"xi_week{w}.csv", index=False)

    # Console summary
    total_ev = pulp.value(obj_xi + obj_capt)
    bench_ev_val = pulp.value(obj_bench) if bench_weight > 0 else 0.0
    print(f"\nOPTIMAL EV over GWs {gws}: {total_ev:.2f} pts"
          + (f"  (+ bench weighted {bench_weight*bench_ev_val:.2f})" if bench_weight>0 else ""))

    for w in gws:
        xi = xi_by_w[w]; cap = cap_by_w[w]
        ev_xi = sum(xp_gw[p].get(w,0.0) for p in xi)
        ev_cap = xp_gw[cap].get(w,0.0)
        spent = sum(cost_w[w].get(p,0) for p in squad_by_w[w])
        fmt = lambda x: f"£{x/10:.1f}m"
        print(f"\nGW {w}: XI EV={ev_xi:.2f} | Captain {name_of[cap]} (+{ev_cap:.2f}) | Spend {fmt(spent)}")

# ───────────────────────── CLI ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xp-csv", type=Path, required=True)
    ap.add_argument("--prices-json", type=Path, required=True)
    ap.add_argument("--season", type=str, required=True)
    ap.add_argument("--gw-start", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--budget", type=int, default=1000)          # tenths (1000 = £100.0m)
    ap.add_argument("--epsilon-spend", type=float, default=0.001)
    ap.add_argument("--bench-weight", type=float, default=0.05)  # small bench value
    ap.add_argument("--min-spend-ratio", type=float, default=0.90)
    ap.add_argument("--high-xp-threshold", type=float, default=2.0)
    ap.add_argument("--min-high-xp-starters", type=int, default=9)
    ap.add_argument("--max-transfers-per-step", type=int, default=1)  # ≤1 (can roll)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    gws = list(range(args.gw_start, args.gw_start + args.horizon))
    xp = load_xp(args.xp_csv, args.season, args.gw_start, args.horizon)
    price_long = load_registry_prices_json(args.prices_json, gws)
    base, xp_gw, cost_w, avail_w = build_universe(xp, price_long, gws)

    solve_transfer_aware(
        base=base,
        xp_gw=xp_gw,
        cost_w=cost_w,
        avail_w=avail_w,
        gws=gws,
        budget=args.budget,
        epsilon_spend=args.epsilon_spend,
        bench_weight=args.bench_weight,
        min_spend_ratio=args.min_spend_ratio,
        high_xp_threshold=args.high_xp_threshold,
        min_high_xp_starters=args.min_high_xp_starters,
        max_transfers_per_step=args.max_transfers_per_step,
        debug=args.debug,
        out_dir=args.out_dir
    )

if __name__ == "__main__":
    main()
