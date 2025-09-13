#!/usr/bin/env python3
"""
Multi-GW selector (MILP)

Goals
-----
• Horizon planning over multiple GWs with:
  - Budget recursion and squad evolution.
  - Per-GW transfer bounds (min/max); solver decides within bounds.
  - Hits & free transfers modeled across weeks (stack to 5).
  - FH/WC declared by user:
      • FH is unconstrained (temporary squad), zero hits/FT use, reverts squad & keeps bank.
      • WC allows transfers with zero hit cost, FT unaffected; budget updates; bounds still apply if you set them.
  - Optional TC/BB (each ≤1 across the horizon; never on FH/WC). Default disabled.

• Price handling:
  - Auto-detect price units (e.g., tenths like 79 for £7.9) and scale into “millions”.
  - Optional --freeze-prices to reuse one snapshot price for all GWs in the horizon.

• Solver controls:
  - --time-limit (wall seconds), --mip-gap (relative gap), --max-nodes, --threads.
  - Keep the best incumbent if solver stops early (don’t crash).

• Output:
  - Per-GW JSON plans under --out-dir
  - Master summary JSON via --out-master
"""

from __future__ import annotations
import argparse, json, os
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import inspect


try:
    import pulp  # type: ignore
except Exception:
    raise SystemExit("pulp is required (pip install pulp).")

TEAM_COL_CANDIDATES = ("team", "team_quota_key")
MAX_FREE_TRANSFERS_STACK = 5
HIT_COST = 4.0
BIGM_EV_DEFAULT = 1000.0  # EV mode-switch big-M (kept fixed)

# ---------------- helpers ----------------
def _read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet: {path}: {e}")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("optimizer_input must be .parquet or .csv")

def _parse_gws(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a.strip()), int(b.strip())
            out.extend(list(range(min(a, b), max(a, b) + 1)))
        else:
            out.append(int(part))
    out = sorted(set(out))
    if not out:
        raise ValueError("--gws resulted in empty set")
    return out

def _parse_bounds_map(s: Optional[str]) -> Dict[int, Tuple[int, int]]:
    out: Dict[int, Tuple[int, int]] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        g, rng = part.split(":", 1)
        a, b = rng.split("-", 1)
        out[int(g.strip())] = (int(a.strip()), int(b.strip()))
    return out

def _parse_chip_plan(s: Optional[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        g, chip = part.split(":", 1)
        chip = chip.strip().upper()
        if chip not in {"FH", "WC"}:
            raise ValueError("chip-plan supports only FH or WC here (TC/BB are solver-chosen).")
        out[int(g.strip())] = chip
    return out

def _parse_topk(s: str) -> Dict[str, int]:
    out = {"GK": 5, "DEF": 15, "MID": 15, "FWD": 10}
    if not s:
        return out
    for part in s.split(","):
        k, v = part.strip().split(":")
        out[k.strip().upper()] = int(v)
    return out

def _round1(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    val = float(x)
    r = round(val + (1e-12 if val >= 0 else -1e-12), 1)
    return 0.0 if r == 0.0 else r

def _get_team_col_name(df: pd.DataFrame) -> str:
    for c in TEAM_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("optimizer_input must include 'team' (preferred) or 'team_quota_key'.")

def _load_team_lookup_json(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {str(k).upper(): str(v) for k, v in d.items()}  # code -> team_id

def _validate_teams(df: pd.DataFrame, teams_json: Optional[str]) -> None:
    team_col = _get_team_col_name(df)
    if "team_id" not in df.columns:
        raise ValueError("optimizer_input missing 'team_id'")
    chk = df.copy()
    chk[team_col] = chk[team_col].astype(str).str.strip().str.upper()
    chk["team_id"] = chk["team_id"].astype(str)

    nunique = chk.groupby("team_id")[team_col].nunique()
    bad_ids = nunique[nunique > 1]
    if not bad_ids.empty:
        details = (
            chk[chk["team_id"].isin(bad_ids.index)]
            .groupby(["team_id", team_col])
            .size()
            .reset_index(name="rows")
            .sort_values(["team_id", "rows"], ascending=[True, False])
        )
        raise ValueError("Inconsistent team code per team_id:\n" + details.to_string(index=False))

    cmap = _load_team_lookup_json(teams_json)
    if cmap:
        unknown = chk[~chk[team_col].isin(cmap.keys())][[team_col]].drop_duplicates()
        if not unknown.empty:
            raise ValueError("Unknown team code(s) vs mapping:\n" + unknown.to_string(index=False))
        tmp = chk.copy()
        tmp["canon_team_id"] = tmp[team_col].map(cmap)
        mism = tmp[tmp["canon_team_id"].notna() & (tmp["team_id"] != tmp["canon_team_id"])]
        if not mism.empty:
            slim = (
                mism.groupby([team_col, "team_id", "canon_team_id"])
                .size()
                .reset_index(name="rows")
                .sort_values("rows", ascending=False)
            )
            raise ValueError("team_id mismatch vs mapping (code -> team_id):\n" + slim.to_string(index=False))

def _pool_union(df: pd.DataFrame, owned_ids: Set[str], gws: List[int], topk: Dict[str, int]) -> pd.DataFrame:
    keep: List[pd.DataFrame] = []
    for g in gws:
        gg = df[df["gw"] == g].copy()
        gg["owned"] = gg["player_id"].astype(str).isin(owned_ids)
        for pos_name, grp in gg.groupby("pos", as_index=False):
            grp = grp.sort_values(["owned", "exp_pts_mean"], ascending=[False, False])
            k = topk.get(pos_name, 10)
            keep.append(pd.concat([grp[grp["owned"]], grp[~grp["owned"]].head(k)], ignore_index=True))
    pool = (
        pd.concat(keep, ignore_index=True)
        .drop_duplicates(["player_id"])
        .reset_index(drop=True)
    )
    return pool

def _parse_chip_envelope(gws: List[int], plan: Dict[int, str]) -> Tuple[Dict[int, int], Dict[int, int]]:
    fh = {g: 1 if plan.get(g) == "FH" else 0 for g in gws}
    wc = {g: 1 if plan.get(g) == "WC" else 0 for g in gws}
    for g in gws:
        if fh[g] and wc[g]:
            raise ValueError(f"GW{g}: cannot schedule both FH and WC.")
    return fh, wc

def _bounds_per_gw(
    gws: List[int], bounds_map: Dict[int, Tuple[int, int]], default_min: int, default_max: int
) -> Dict[int, Tuple[int, int]]:
    out = {}
    for g in gws:
        out[g] = bounds_map.get(g, (default_min, default_max))
        lo, hi = out[g]
        if lo < 0 or hi < 0 or lo > hi:
            raise ValueError(f"Bad transfer bounds for GW{g}: {lo}-{hi}")
    return out

def _infer_money_scale(df: pd.DataFrame, held_prev_vals: List[float]) -> float:
    """
    Return multiplicative scale to convert stored prices into 'millions'.
    If values look like 75–160, return 0.1; if already 7.5–16, return 1.0.
    """
    mx_df = float(df["price"].max()) if "price" in df.columns and len(df) else 0.0
    mx_hp = float(max(held_prev_vals) if held_prev_vals else 0.0)
    mx = max(mx_df, mx_hp)
    if mx >= 50.0:
        return 0.1
    return 1.0

def _money_stats_guard(
    df_scaled: pd.DataFrame, held_prev_scaled: List[float], verbose: bool
) -> Tuple[float, float]:
    """
    Using scaled data (already multiplied by 'scale'), compute safe PRICE_UB and BIGM_PRICE.
    """
    pmax_df = float(df_scaled["price"].max()) if "price" in df_scaled.columns and len(df_scaled) else 0.0
    pmax_hp = float(max(held_prev_scaled) if held_prev_scaled else 0.0)
    pmax = max(pmax_df, pmax_hp)

    # sanity
    if pmax < 2.0:
        raise ValueError(
            f"Scaled price max looks too small ({pmax:.2f}). Check units in optimizer_input and team_state."
        )

    if pmax > 30.0 and verbose:
        print(f"[warn] Scaled price max looks high ({pmax:.2f}). Confirm no double-scaling upstream.")

    price_ub = max(20.0, round(pmax * 1.10, 2))  # 10% headroom
    bigm_price = max(price_ub, 20.0)
    return price_ub, bigm_price

def _validate_pool_min_positions(pool: pd.DataFrame) -> None:
    req = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    cnt = pool["pos"].value_counts().to_dict()
    missing = {k: v for k, v in req.items() if cnt.get(k, 0) < v}
    if missing:
        raise ValueError(
            "Candidate pool too small by position (increase --topk or expand GWs). "
            f"Needed at least {req}, got {cnt}."
        )

def _make_cbc_solver(
    msg: bool,
    time_limit: Optional[int],
    mip_gap: Optional[float],
    max_nodes: Optional[int],
    threads: Optional[int],
):
    """
    Build a CBC solver compatible with multiple PuLP versions.
    Falls back to raw CBC flags when ctor keywords aren't available.
    """
    params = set(inspect.signature(pulp.PULP_CBC_CMD.__init__).parameters)
    kwargs = {"msg": bool(msg)}
    options: List[str] = []

    # time limit
    if time_limit is not None:
        if "timeLimit" in params:
            kwargs["timeLimit"] = int(time_limit)
        elif "maxSeconds" in params:
            kwargs["maxSeconds"] = int(time_limit)
        else:
            options += ["-seconds", str(int(time_limit))]

    # mip gap (relative)
    if mip_gap is not None:
        if "gapRel" in params:
            kwargs["gapRel"] = float(mip_gap)
        elif "fracGap" in params:
            kwargs["fracGap"] = float(mip_gap)
        else:
            options += ["-ratio", str(float(mip_gap))]

    # node limit
    if max_nodes is not None:
        if "nodeLimit" in params:
            kwargs["nodeLimit"] = int(max_nodes)
        else:
            options += ["-maxNodes", str(int(max_nodes))]

    # threads
    if threads is not None:
        if "threads" in params:
            kwargs["threads"] = int(threads)
        else:
            options += ["-threads", str(int(threads))]

    if options:
        kwargs["options"] = options

    return pulp.PULP_CBC_CMD(**kwargs)

# ---------------- core MILP ----------------
def solve_multi_gw(
    team_state_path: str,
    optimizer_input_path: str,
    out_dir: str,
    out_master: str,
    gws: List[int],
    risk_lambda: float = 0.0,
    topk: Optional[Dict[str, int]] = None,
    allow_hits: bool = True,
    max_extra_transfers: int = 3,
    transfer_bounds: Optional[Dict[int, Tuple[int, int]]] = None,
    chip_plan: Optional[Dict[int, str]] = None,  # only FH/WC here
    enable_tc: bool = False,
    enable_bb: bool = False,
    tc_multiplier: float = 3.0,
    teams_json: Optional[str] = None,
    price_scale: Optional[float] = None,
    freeze_prices: bool = False,
    freeze_prices_gw: Optional[int] = None,
    # Solver controls
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
    max_nodes: Optional[int] = None,
    threads: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    # ---- Load state ----
    with open(team_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    bank0 = float(state.get("bank", 0.0))
    ft0 = int(state.get("free_transfers", 1))
    season = str(state.get("season"))
    snapshot_gw = int(state.get("gw"))
    owned0: List[dict] = list(state.get("squad", []))
    owned_ids0: Set[str] = {str(p["player_id"]) for p in owned0}
    sell_map0_raw: Dict[str, float] = {
        str(p["player_id"]): float(p.get("sell_price", p.get("price", 0.0))) for p in owned0
    }
    held_prev_vals_raw = list(sell_map0_raw.values())

    # ---- Load data ----
    df_all = _read_any(optimizer_input_path)
    req_cols = [
        "season", "gw", "player_id", "team_id", "pos", "price", "sell_price",
        "p60", "exp_pts_mean", "exp_pts_var", "cs_prob", "is_dgw", "captain_uplift",
    ]
    missing = [c for c in req_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"optimizer_input missing columns: {missing}")

    # ---- Determine money scale ----
    scale = float(price_scale) if price_scale is not None else _infer_money_scale(df_all, held_prev_vals_raw)

    # apply scaling consistently
    df_all = df_all.copy()
    df_all["price"] = df_all["price"].astype(float) * scale
    sell_map0 = {k: v * scale for k, v in sell_map0_raw.items()}
    bank0 *= scale

    # ---- Slice by GWs after scaling ----
    df = df_all[df_all["gw"].isin(gws)].copy()
    if df.empty:
        raise ValueError("optimizer_input has no rows for requested GWs")

    # validation of team mapping
    _validate_teams(df, teams_json=teams_json)
    team_col = _get_team_col_name(df)

    # ---- Build candidate pool union across horizon ----
    pool = _pool_union(df, owned_ids0, gws, topk or {"GK": 10, "DEF": 25, "MID": 25, "FWD": 20})
    _validate_pool_min_positions(pool)

    pid_list = pool["player_id"].astype(str).tolist()
    P = len(pid_list)
    pid_index = {pid_list[i]: i for i in range(P)}

    # ---- Per-GW index and attributes ----
    gw_list = gws
    G = len(gw_list)
    gw_index = {gw_list[t]: t for t in range(G)}

    # Static attributes from first GW occurrence
    pos = np.empty(P, dtype=object)
    name = np.array([None] * P, dtype=object)
    team_code = np.empty(P, dtype=object)
    team_id_canon = np.empty(P, dtype=object)

    rep = df.merge(pool[["player_id"]], on="player_id", how="inner")
    rep = rep.sort_values(["player_id", "gw"]).drop_duplicates("player_id", keep="first")
    pos_map = dict(zip(rep["player_id"].astype(str), rep["pos"].astype(str)))
    name_map = dict(zip(rep["player_id"].astype(str), rep.get("player", rep["player_id"]).astype(object)))
    code_map = dict(zip(rep["player_id"].astype(str), rep[team_col].astype(str)))
    id_map = dict(zip(rep["player_id"].astype(str), rep["team_id"].astype(str)))
    for p in pid_list:
        pos[pid_index[p]] = pos_map.get(p, "MID")
        name[pid_index[p]] = name_map.get(p, None)
        team_code[pid_index[p]] = code_map.get(p, "UNK")
        team_id_canon[pid_index[p]] = id_map.get(p, "UNK")

    # ---- Freeze prices (optional): build snapshot map ----
    freeze_gw = int(freeze_prices_gw) if freeze_prices_gw is not None else snapshot_gw
    base_price_map: Dict[str, float] = {}
    if freeze_prices:
        sub = df_all[df_all["player_id"].astype(str).isin(pid_list)].copy()
        # choose nearest price at/before freeze_gw; else nearest after
        for p in pid_list:
            rp = sub[sub["player_id"].astype(str) == p]
            before = rp[rp["gw"] <= freeze_gw].sort_values("gw")
            if len(before) > 0:
                v = float(before["price"].iloc[-1])
            else:
                after = rp[rp["gw"] >= freeze_gw].sort_values("gw")
                if len(after) > 0:
                    v = float(after["price"].iloc[0])
                else:
                    v = np.nan  # no price available near snapshot
            base_price_map[p] = v
        missing_bp = [p for p, v in base_price_map.items() if not np.isfinite(v)]
        if missing_bp:
            raise ValueError(
                f"Missing snapshot price for {len(missing_bp)} players (e.g., {missing_bp[:5]}). "
                "Ensure optimizer_input has prices near the snapshot GW."
            )

    # ---- Compute price big-M bounds (after knowing price mode) ----
    if freeze_prices:
        pmax = max(max(base_price_map.values()) if base_price_map else 0.0,
                   max(sell_map0.values()) if sell_map0 else 0.0)
        if pmax < 2.0:
            raise ValueError(f"Frozen scaled price max looks too small ({pmax:.2f}). Check units.")
        if verbose and pmax > 30.0:
            print(f"[warn] Frozen scaled price max looks high ({pmax:.2f}) — check upstream scaling.")
        PRICE_UB = max(20.0, round(pmax * 1.10, 2))
        BIGM_PRICE = max(PRICE_UB, 20.0)
    else:
        PRICE_UB, BIGM_PRICE = _money_stats_guard(df_all, list(sell_map0.values()), verbose=bool(verbose))
    BIGM_EV = BIGM_EV_DEFAULT

    # ---- Build per-GW matrices ----
    price = np.zeros((G, P), dtype=float)
    ev = np.zeros((G, P), dtype=float)
    var = np.zeros((G, P), dtype=float)
    capup = np.zeros((G, P), dtype=float)

    if freeze_prices:
        for g in gw_list:
            gg = df[df["gw"] == g]
            t = gw_index[g]
            for _, row in gg.iterrows():
                p = str(row["player_id"])
                if p not in pid_index:
                    continue
                i = pid_index[p]
                price[t, i] = float(base_price_map[p])  # frozen price
                ev[t, i] = float(row["exp_pts_mean"])
                var[t, i] = max(0.0, float(row["exp_pts_var"]))
                capup[t, i] = max(0.0, float(row["captain_uplift"]))
    else:
        for g in gw_list:
            gg = df[df["gw"] == g]
            t = gw_index[g]
            for _, row in gg.iterrows():
                p = str(row["player_id"])
                if p not in pid_index:
                    continue
                i = pid_index[p]
                price[t, i] = float(row["price"])
                ev[t, i] = float(row["exp_pts_mean"])
                var[t, i] = max(0.0, float(row["exp_pts_var"]))
                capup[t, i] = max(0.0, float(row["captain_uplift"]))

    # ---- Chip plan & bounds ----
    fh_map, wc_map = _parse_chip_envelope(gw_list, chip_plan or {})
    bounds = _bounds_per_gw(gw_list, transfer_bounds or {}, default_min=0, default_max=3)

    # ---- Model ----
    m = pulp.LpProblem("multi_gw_selector", pulp.LpMaximize)

    # Decision variables
    in_squad = pulp.LpVariable.dicts("in", [(t, i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)
    buy = pulp.LpVariable.dicts("buy", [(t, i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)
    sell = pulp.LpVariable.dicts("sell", [(t, i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)

    start = pulp.LpVariable.dicts("start", [(t, i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)
    cap_norm = pulp.LpVariable.dicts("cap_n", [(t, i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)
    cap_tc = pulp.LpVariable.dicts("cap_tc", [(t, i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)

    # FH temporary squad/XI
    tmp_in = pulp.LpVariable.dicts("fh_in", [(t, i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)
    tmp_start = pulp.LpVariable.dicts("fh_start", [(t, i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)
    tmp_capn = pulp.LpVariable.dicts("fh_cap_n", [(t, i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)
    tmp_capt = pulp.LpVariable.dicts("fh_cap_tc", [(t, i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)

    # Bank and FT/hits per GW
    bank = pulp.LpVariable.dicts("bank", [t for t in range(G)], lowBound=-0.0001)
    ft = pulp.LpVariable.dicts("ft", [t for t in range(G + 1)], lowBound=0, upBound=MAX_FREE_TRANSFERS_STACK, cat=pulp.LpInteger)
    ft_used = pulp.LpVariable.dicts("ft_used", [t for t in range(G)], lowBound=0, upBound=MAX_FREE_TRANSFERS_STACK, cat=pulp.LpInteger)
    hits = pulp.LpVariable.dicts("hits", [t for t in range(G)], lowBound=0, upBound=max_extra_transfers, cat=pulp.LpInteger)
    cap_tc_flag = pulp.LpVariable.dicts("tc", [t for t in range(G)], 0, 1, cat=pulp.LpBinary)
    bb_flag = pulp.LpVariable.dicts("bb", [t for t in range(G)], 0, 1, cat=pulp.LpBinary)

    # Shadow price carried when owned (value realized on sell)
    held = pulp.LpVariable.dicts("held_price", [(t, i) for t in range(G) for i in range(P)], lowBound=0, upBound=None, cat=pulp.LpContinuous)

    # ---------------- Constraints ----------------

    # Initial FT and bank
    m += ft[0] == min(MAX_FREE_TRANSFERS_STACK, ft0), "ft_init"
    m += bank[0] == bank0, "bank_init"

    # Initial squad constants
    in_prev = np.array([1 if p in owned_ids0 else 0 for p in pid_list], dtype=int)
    held_prev = np.array([float(sell_map0.get(p, 0.0)) for p in pid_list], dtype=float)

    # One TC total; one BB total; exclusions
    if enable_tc:
        m += pulp.lpSum(cap_tc_flag[t] for t in range(G)) <= 1
    else:
        for t in range(G): m += cap_tc_flag[t] == 0

    if enable_bb:
        m += pulp.lpSum(bb_flag[t] for t in range(G)) <= 1
    else:
        for t in range(G): m += bb_flag[t] == 0
    for t in range(G):
        if fh_map[gw_list[t]] or wc_map[gw_list[t]]:
            m += cap_tc_flag[t] == 0, f"no_tc_on_chip_{t}"
            m += bb_flag[t] == 0, f"no_bb_on_chip_{t}"
        m += cap_tc_flag[t] + bb_flag[t] <= 1, f"no_tc_and_bb_same_week_{t}"

    # Per-week squad & XI constraints
    def _sum_pos_on(vec, t, pname):
        return pulp.lpSum(vec[(t, i)] for i in range(P) if pos[i] == pname)

    for t in range(G):
        # Normal-week squad
        m += pulp.lpSum(in_squad[(t, i)] for i in range(P)) == 15, f"squad15_{t}"
        m += _sum_pos_on(in_squad, t, "GK") == 2, f"comp_gk2_{t}"
        m += _sum_pos_on(in_squad, t, "DEF") == 5, f"comp_def5_{t}"
        m += _sum_pos_on(in_squad, t, "MID") == 5, f"comp_mid5_{t}"
        m += _sum_pos_on(in_squad, t, "FWD") == 3, f"comp_fwd3_{t}"
        # Team cap ≤3
        teams = list(set(team_code.tolist()))
        for T in teams:
            m += pulp.lpSum(in_squad[(t, i)] for i in range(P) if team_code[i] == T) <= 3, f"teamcap_{t}_{T}"

        # XI
        m += pulp.lpSum(start[(t, i)] for i in range(P)) == 11, f"xi11_{t}"
        for i in range(P):
            m += start[(t, i)] <= in_squad[(t, i)], f"start_in_{t}_{i}"

        # Formation ranges
        m += _sum_pos_on(start, t, "GK") == 1, f"xi_gk1_{t}"
        m += _sum_pos_on(start, t, "DEF") >= 3
        m += _sum_pos_on(start, t, "DEF") <= 5
        m += _sum_pos_on(start, t, "MID") >= 2
        m += _sum_pos_on(start, t, "MID") <= 5
        m += _sum_pos_on(start, t, "FWD") >= 1
        m += _sum_pos_on(start, t, "FWD") <= 3

        # Captain mode split (normal vs TC)
        m += pulp.lpSum(cap_norm[(t, i)] for i in range(P)) == (1 - cap_tc_flag[t]), f"cap_norm_count_{t}"
        m += pulp.lpSum(cap_tc[(t, i)] for i in range(P)) == cap_tc_flag[t], f"cap_tc_count_{t}"
        for i in range(P):
            m += cap_norm[(t, i)] <= start[(t, i)], f"capn_start_{t}_{i}"
            m += cap_tc[(t, i)] <= start[(t, i)], f"captc_start_{t}_{i}"

        # FH temporary squad if FH week
        if fh_map[gw_list[t]] == 1:
            m += pulp.lpSum(tmp_in[(t, i)] for i in range(P)) == 15, f"fh_squad15_{t}"
            m += _sum_pos_on(tmp_in, t, "GK") == 2
            m += _sum_pos_on(tmp_in, t, "DEF") == 5
            m += _sum_pos_on(tmp_in, t, "MID") == 5
            m += _sum_pos_on(tmp_in, t, "FWD") == 3
            for T in teams:
                m += pulp.lpSum(tmp_in[(t, i)] for i in range(P) if team_code[i] == T) <= 3, f"fh_teamcap_{t}_{T}"

            m += pulp.lpSum(tmp_start[(t, i)] for i in range(P)) == 11, f"fh_xi11_{t}"
            for i in range(P):
                m += tmp_start[(t, i)] <= tmp_in[(t, i)], f"fh_start_in_{t}_{i}"
            m += _sum_pos_on(tmp_start, t, "GK") == 1
            m += _sum_pos_on(tmp_start, t, "DEF") >= 3
            m += _sum_pos_on(tmp_start, t, "DEF") <= 5
            m += _sum_pos_on(tmp_start, t, "MID") >= 2
            m += _sum_pos_on(tmp_start, t, "MID") <= 5
            m += _sum_pos_on(tmp_start, t, "FWD") >= 1
            m += _sum_pos_on(tmp_start, t, "FWD") <= 3

            m += pulp.lpSum(tmp_capn[(t, i)] for i in range(P)) == (1 - cap_tc_flag[t]), f"fh_cap_norm_count_{t}"
            m += pulp.lpSum(tmp_capt[(t, i)] for i in range(P)) == cap_tc_flag[t], f"fh_cap_tc_count_{t}"
            for i in range(P):
                m += tmp_capn[(t, i)] <= tmp_start[(t, i)], f"fh_capn_start_{t}_{i}"
                m += tmp_capt[(t, i)] <= tmp_start[(t, i)], f"fh_capt_start_{t}_{i}"

    # Transitions, budget, FT/hits recursion
    for t in range(G):
        gw = gw_list[t]
        is_fh = fh_map[gw] == 1
        is_wc = wc_map[gw] == 1

        # Ownership transition
        for i in range(P):
            if t == 0:
                m += in_squad[(t, i)] == in_prev[i] + buy[(t, i)] - sell[(t, i)], f"trans_{t}_{i}"
            else:
                m += in_squad[(t, i)] == in_squad[(t - 1, i)] + buy[(t, i)] - sell[(t, i)], f"trans_{t}_{i}"

        # FH week: no persistent buys/sells
        if is_fh:
            for i in range(P):
                m += buy[(t, i)] == 0, f"no_buy_fh_{t}_{i}"
                m += sell[(t, i)] == 0, f"no_sell_fh_{t}_{i}"

        # --- Realized proceeds (linearize z = sell * held_prev_or_last) ---
        proceeds_vars = []
        for i in range(P):
            z = pulp.LpVariable(f"proceeds_{t}_{i}", lowBound=0, upBound=PRICE_UB, cat=pulp.LpContinuous)
            if t == 0:
                y_const = held_prev[i]  # already scaled
                m += z <= y_const, f"proc_t{t}_i{i}_ub_y0"
                m += z <= PRICE_UB * sell[(t, i)], f"proc_t{t}_i{i}_ub_x0"
                m += z >= y_const - PRICE_UB * (1 - sell[(t, i)]), f"proc_t{t}_i{i}_lb_bigM0"
            else:
                y = held[(t - 1, i)]  # bounded later by in_squad
                m += z <= y, f"proc_t{t}_i{i}_ub_y"
                m += z <= PRICE_UB * sell[(t, i)], f"proc_t{t}_i{i}_ub_x"
                m += z >= y - PRICE_UB * (1 - sell[(t, i)]), f"proc_t{t}_i{i}_lb_bigM"
            proceeds_vars.append(z)
        proceeds = pulp.lpSum(proceeds_vars)

        # Buys cost
        buys_cost = pulp.lpSum(buy[(t, i)] * price[t, i] for i in range(P))

        # Budget recursion
        if is_fh:
            # bank unchanged this GW
            if t == 0:
                m += bank[t] == bank0, f"bank_fh_{t}"
            else:
                m += bank[t] == bank[t - 1], f"bank_fh_{t}"
        else:
            if t == 0:
                m += bank[t] == bank0 + proceeds - buys_cost, f"bank_{t}"
            else:
                m += bank[t] == bank[t - 1] + proceeds - buys_cost, f"bank_{t}"
        m += bank[t] >= -1e-6, f"bank_nonneg_{t}"

        # Shadow price carry (held)
        for i in range(P):
            if t == 0:
                # held_t equals either new buy price (if buy) or previous held (if kept), when owned
                m += held[(t, i)] <= price[t, i] + BIGM_PRICE * (1 - buy[(t, i)]) + BIGM_PRICE * (1 - in_squad[(t, i)])
                m += held[(t, i)] >= price[t, i] - BIGM_PRICE * (1 - buy[(t, i)]) - BIGM_PRICE * (1 - in_squad[(t, i)])
                m += held[(t, i)] <= held_prev[i] + BIGM_PRICE * buy[(t, i)] + BIGM_PRICE * (1 - in_squad[(t, i)])
                m += held[(t, i)] >= held_prev[i] - BIGM_PRICE * buy[(t, i)] - BIGM_PRICE * (1 - in_squad[(t, i)])
            else:
                m += held[(t, i)] <= price[t, i] + BIGM_PRICE * (1 - buy[(t, i)]) + BIGM_PRICE * (1 - in_squad[(t, i)])
                m += held[(t, i)] >= price[t, i] - BIGM_PRICE * (1 - buy[(t, i)]) - BIGM_PRICE * (1 - in_squad[(t, i)])
                m += held[(t, i)] <= held[(t - 1, i)] + BIGM_PRICE * buy[(t, i)] + BIGM_PRICE * (1 - in_squad[(t, i)])
                m += held[(t, i)] >= held[(t - 1, i)] - BIGM_PRICE * buy[(t, i)] - BIGM_PRICE * (1 - in_squad[(t, i)])
            # If not owned after t, force held to 0 upper bound (guard)
            m += held[(t, i)] <= PRICE_UB * in_squad[(t, i)], f"held_guard_{t}_{i}"

        # Transfer counts for bounds
        transfers_cnt = pulp.lpSum(buy[(t, i)] for i in range(P))
        lo, hi = bounds[gw]
        if not is_fh:  # FH unconstrained by bounds (temporary squad)
            m += transfers_cnt >= lo, f"tmin_{t}"
            m += transfers_cnt <= hi, f"tmax_{t}"

        # FT and hits
        if is_fh or is_wc:
            # No consumption; hits=0; ft_next = min(5, ft + 1)
            m += ft_used[t] == 0, f"ft_used_chip_{t}"
            m += hits[t] == 0, f"hits_chip_{t}"
        else:
            # ft_used <= ft and <= transfers_cnt
            m += ft_used[t] <= ft[t], f"ft_used_le_ft_{t}"
            m += ft_used[t] <= transfers_cnt, f"ft_used_le_tr_{t}"
            # hits = transfers_cnt - ft_used
            m += hits[t] == transfers_cnt - ft_used[t], f"hits_def_{t}"
            if not allow_hits:
                m += hits[t] == 0, f"no_hits_{t}"
            else:
                m += hits[t] <= max_extra_transfers, f"hits_cap_{t}"

        # FT roll-forward with cap 5: ft[t+1] = min(5, ft[t] - ft_used[t] + 1)
        c = pulp.LpVariable(f"ft_clip_{t}", lowBound=0)
        m += ft[t + 1] == ft[t] - ft_used[t] + 1 - c, f"ft_recur_{t}"
        m += c >= ft[t] - ft_used[t] + 1 - MAX_FREE_TRANSFERS_STACK, f"ft_clip_lb_{t}"
        m += ft[t + 1] <= MAX_FREE_TRANSFERS_STACK, f"ft_next_le_5_{t}"
        m += ft[t + 1] >= 0, f"ft_next_ge_0_{t}"

    # Objective: sum over GWs
    obj = 0
    for t in range(G):
        gw = gw_list[t]
        is_fh = fh_map[gw] == 1

        # XI EV / All-15 EV
        xi_ev = pulp.lpSum(start[(t, i)] * ev[t, i] for i in range(P))
        all_ev = pulp.lpSum(in_squad[(t, i)] * ev[t, i] for i in range(P))
        xi_var = pulp.lpSum(start[(t, i)] * var[t, i] for i in range(P))

        if is_fh:
            xi_ev_fh = pulp.lpSum(tmp_start[(t, i)] * ev[t, i] for i in range(P))
            all_ev_fh = pulp.lpSum(tmp_in[(t, i)] * ev[t, i] for i in range(P))
            xi_var_fh = pulp.lpSum(tmp_start[(t, i)] * var[t, i] for i in range(P))

        # Captain uplift with TC split (normal vs TC)
        uplift_norm = pulp.lpSum(cap_norm[(t, i)] * capup[t, i] for i in range(P))
        uplift_tc = pulp.lpSum(cap_tc[(t, i)] * ((tc_multiplier - 1.0) * capup[t, i]) for i in range(P))
        if is_fh:
            uplift_norm_fh = pulp.lpSum(tmp_capn[(t, i)] * capup[t, i] for i in range(P))
            uplift_tc_fh = pulp.lpSum(tmp_capt[(t, i)] * ((tc_multiplier - 1.0) * capup[t, i]) for i in range(P))

        # BB mode switch (≤1 per horizon, never with FH/WC)
        if is_fh:
            ev_term = pulp.LpVariable(f"ev_term_fh_{t}", lowBound=-BIGM_EV, upBound=BIGM_EV)
            m += ev_term <= xi_ev_fh + BIGM_EV * bb_flag[t]
            m += ev_term >= xi_ev_fh - BIGM_EV * bb_flag[t]
            m += ev_term <= all_ev_fh + BIGM_EV * (1 - bb_flag[t])
            m += ev_term >= all_ev_fh - BIGM_EV * (1 - bb_flag[t])
            obj += ev_term + uplift_norm_fh + uplift_tc_fh - float(risk_lambda) * xi_var_fh
        else:
            ev_term = pulp.LpVariable(f"ev_term_{t}", lowBound=-BIGM_EV, upBound=BIGM_EV)
            m += ev_term <= xi_ev + BIGM_EV * bb_flag[t]
            m += ev_term >= xi_ev - BIGM_EV * bb_flag[t]
            m += ev_term <= all_ev + BIGM_EV * (1 - bb_flag[t])
            m += ev_term >= all_ev - BIGM_EV * (1 - bb_flag[t])
            hits_penalty = 0.0 if wc_map[gw] == 1 else HIT_COST
            obj += ev_term + uplift_norm + uplift_tc - float(risk_lambda) * xi_var - hits_penalty * hits[t]

    m += obj

    # ---------------- Solve with limits ----------------
    # Build a solver using wrapper that adapts to available pulp versions
    solver = _make_cbc_solver(
        msg=bool(verbose),
        time_limit=time_limit,   # seconds (None means unlimited)
        mip_gap=mip_gap,         # relative mip gap (e.g., 0.005 = 0.5%)
        max_nodes=max_nodes,     # cap B&B nodes
        threads=threads          # number of threads
    )
    res = m.solve(solver)
    status_str = pulp.LpStatus[res]

    # Accept "Optimal" OR any status that left a feasible incumbent (e.g., "Not Solved").
    if status_str == "Infeasible":
        raise RuntimeError("MILP infeasible (check bounds/chips/budget).")
    elif status_str == "Unbounded":
        raise RuntimeError("MILP unbounded (check constraints).")
    else:
        # If solver stopped early, ensure we at least have values.
        some_values = any(pulp.value(v) is not None for v in in_squad.values())
        if not some_values:
            raise RuntimeError(f"MILP stopped early with no incumbent: status={status_str}")

    # --------------- Extract & write outputs ---------------
    os.makedirs(out_dir, exist_ok=True)

    def _pobj(pid: Optional[str], t: int, use_tmp: bool = False) -> Optional[dict]:
        if pid is None:
            return None
        i = pid_index[pid]
        return {
            "id": pid,
            "name": None if name[i] is None else str(name[i]),
            "pos": str(pos[i]),
            "team": str(team_code[i]),
            "xPts": _round1(ev[t, i]),
        }

    plans: Dict[int, dict] = {}
    ft_path = []
    bank_path = []

    for t, gw in enumerate(gw_list):
        is_fh = fh_map[gw] == 1
        is_wc = wc_map[gw] == 1

        bank_val = float(pulp.value(bank[t]))
        ft_now = int(pulp.value(ft[t]))
        ft_used_val = int(pulp.value(ft_used[t]))
        hits_val = int(pulp.value(hits[t]))
        ft_next = int(pulp.value(ft[t + 1]))
        bank_path.append({"gw": gw, "bank": _round1(bank_val)})
        ft_path.append(
            {
                "gw": gw,
                "ft_before": ft_now,
                "ft_used": ft_used_val,
                "ft_after": max(0, ft_now - ft_used_val),
                "ft_next": ft_next,
            }
        )

        def picks(mask, use_tmp=False):
            if use_tmp:
                if mask == "in":
                    return [pid_list[i] for i in range(P) if pulp.value(tmp_in[(t, i)]) > 0.5]
                if mask == "start":
                    return [pid_list[i] for i in range(P) if pulp.value(tmp_start[(t, i)]) > 0.5]
                if mask == "capn":
                    return [pid_list[i] for i in range(P) if pulp.value(tmp_capn[(t, i)]) > 0.5]
                if mask == "capt":
                    return [pid_list[i] for i in range(P) if pulp.value(tmp_capt[(t, i)]) > 0.5]
            else:
                if mask == "in":
                    return [pid_list[i] for i in range(P) if pulp.value(in_squad[(t, i)]) > 0.5]
                if mask == "start":
                    return [pid_list[i] for i in range(P) if pulp.value(start[(t, i)]) > 0.5]
                if mask == "capn":
                    return [pid_list[i] for i in range(P) if pulp.value(cap_norm[(t, i)]) > 0.5]
                if mask == "capt":
                    return [pid_list[i] for i in range(P) if pulp.value(cap_tc[(t, i)]) > 0.5]
            return []

        xi_ids = picks("start", use_tmp=is_fh)
        cap_id_n = picks("capn", use_tmp=is_fh)
        cap_id_t = picks("capt", use_tmp=is_fh)
        cap_pid = (cap_id_t or cap_id_n)[0] if (cap_id_t or cap_id_n) else None

        buy_ids = [] if is_fh else [pid_list[i] for i in range(P) if pulp.value(buy[(t, i)]) > 0.5]
        sell_ids = [] if is_fh else [pid_list[i] for i in range(P) if pulp.value(sell[(t, i)]) > 0.5]

        def _count(pname: str) -> int:
            if is_fh:
                return sum(1 for i in range(P) if pos[i] == pname and pulp.value(tmp_start[(t, i)]) > 0.5)
            else:
                return sum(1 for i in range(P) if pos[i] == pname and pulp.value(start[(t, i)]) > 0.5)

        formation = f"{_count('DEF')}-{_count('MID')}-{_count('FWD')}"

        ev_xi = sum(ev[t, i] * (pulp.value(tmp_start[(t, i)]) if is_fh else pulp.value(start[(t, i)])) for i in range(P))
        ev_15 = sum(ev[t, i] * (pulp.value(tmp_in[(t, i)]) if is_fh else pulp.value(in_squad[(t, i)])) for i in range(P))
        bb_on = int(pulp.value(bb_flag[t]))
        ev_team = ev_15 if bb_on else ev_xi

        uplift_base = sum(
            capup[t, i] * (pulp.value(tmp_capn[(t, i)]) if is_fh else pulp.value(cap_norm[(t, i)])) for i in range(P)
        )
        uplift_tc_val = sum(
            ((tc_multiplier - 1.0) * capup[t, i]) * (pulp.value(tmp_capt[(t, i)]) if is_fh else pulp.value(cap_tc[(t, i)]))
            for i in range(P)
        )
        uplift_total = uplift_base + uplift_tc_val

        var_pen = float(risk_lambda) * sum(
            var[t, i] * (pulp.value(tmp_start[(t, i)]) if is_fh else pulp.value(start[(t, i)])) for i in range(P)
        )
        hit_cost = 0.0 if (is_fh or is_wc) else HIT_COST * hits_val
        total = ev_team + uplift_total - var_pen - hit_cost

        buys_cost_val = sum(price[t, i] for i in range(P) if not is_fh and pulp.value(buy[(t, i)]) > 0.5)
        if t == 0:
            sells_proceeds_val = sum(held_prev[i] for i in range(P) if not is_fh and pulp.value(sell[(t, i)]) > 0.5)
        else:
            sells_proceeds_val = sum(
                float(pulp.value(held[(t - 1, i)])) for i in range(P) if not is_fh and pulp.value(sell[(t, i)]) > 0.5
            )
        bank_before = float(bank0 if t == 0 else pulp.value(bank[t - 1])) if not is_fh else float(pulp.value(bank[t]))
        bank_after = float(pulp.value(bank[t]))

        transfers_out = []
        rb = list(buy_ids)
        for sid in sell_ids:
            in_id = rb.pop(0) if rb else None
            bp = float(price[t, pid_index[in_id]]) if in_id else None
            sp = float(held_prev[pid_index[sid]] if t == 0 else pulp.value(held[(t - 1, pid_index[sid])]))
            pair_net = (bp if bp is not None else 0.0) - float(sp)
            transfers_out.append(
                {
                    "out": sid,
                    "out_name": None if name[pid_index[sid]] is None else str(name[pid_index[sid]]),
                    "out_pos": str(pos[pid_index[sid]]),
                    "out_team": str(team_code[pid_index[sid]]),
                    "in": in_id,
                    "in_name": None if (in_id is None or name[pid_index[in_id]] is None) else str(name[pid_index[in_id]]),
                    "in_pos": None if in_id is None else str(pos[pid_index[in_id]]),
                    "in_team": None if in_id is None else str(team_code[pid_index[in_id]]),
                    "sell_value": _round1(sp),
                    "buy_price": _round1(bp),
                    "pair_net": _round1(pair_net),
                }
            )
        for in_id in rb:
            bp = float(price[t, pid_index[in_id]])
            transfers_out.append(
                {
                    "out": None,
                    "out_name": None,
                    "out_pos": None,
                    "out_team": None,
                    "in": in_id,
                    "in_name": None if name[pid_index[in_id]] is None else str(name[pid_index[in_id]]),
                    "in_pos": str(pos[pid_index[in_id]]),
                    "in_team": str(team_code[pid_index[in_id]]),
                    "sell_value": None,
                    "buy_price": _round1(bp),
                    "pair_net": _round1(bp),
                }
            )

        plan = {
            "objective": {
                "ev": _round1(ev_team),
                "captain_uplift": _round1(uplift_total),
                "risk_penalty": _round1(var_pen),
                "hit_cost": _round1(hit_cost),
                "total": _round1(total),
            },
            "meta": {
                "season": season,
                "gw": gw,
                "snapshot_gw": snapshot_gw,
                "chip": ("FH" if is_fh else ("WC" if is_wc else ("TC" if int(pulp.value(cap_tc_flag[t])) else ("BB" if bb_on else None)))),
                "formation": formation,
                "free_transfers_before": ft_now,
                "free_transfers_used": ft_used_val,
                "free_transfers_after": max(0, ft_now - ft_used_val),
                "free_transfers_next": ft_next,
                "max_free_transfers_stack": MAX_FREE_TRANSFERS_STACK,
                "bank_before": _round1(bank_before),
                "bank_after": _round1(bank_after),
                "budget": {
                    "buys_cost": _round1(buys_cost_val),
                    "sells_proceeds": _round1(sells_proceeds_val),
                    "net_spend": _round1(buys_cost_val - sells_proceeds_val),
                },
                "transfers_used": 0 if is_fh else len(buy_ids),
                "hits_charged": 0 if (is_fh or is_wc) else hits_val,
                "prices_mode": "frozen" if freeze_prices else "per-gw",
                "prices_frozen_from_gw": (freeze_gw if freeze_prices else None),
                "price_scale": scale,
            },
            "xi": [_pobj(x, t, use_tmp=is_fh) for x in xi_ids],
            "captain": _pobj(cap_pid, t, use_tmp=is_fh),
            "transfers": transfers_out,
        }
        out_path = os.path.join(out_dir, f"gw{gw:02d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        plans[gw] = plan

    # Master summary
    master = {
        "season": season,
        "gws": gw_list,
        "chips": {
            "FH": [g for g in gw_list if fh_map[g] == 1],
            "WC": [g for g in gw_list if wc_map[g] == 1],
            "TC": [gw_list[t] for t in range(G) if int(pulp.value(cap_tc_flag[t])) == 1],
            "BB": [gw_list[t] for t in range(G) if int(pulp.value(bb_flag[t])) == 1],
        },
        "paths": {
            "bank": bank_path,
            "free_transfers": ft_path,
        },
        "objective_total": _round1(sum((plans[g]["objective"]["total"] or 0.0) for g in gw_list)),
        "out_dir": out_dir,
        "prices_mode": "frozen" if freeze_prices else "per-gw",
        "prices_frozen_from_gw": (freeze_gw if freeze_prices else None),
        "price_scale": scale,
        "solver": {
            "status": status_str,
            "time_limit_sec": time_limit,
            "mip_gap_target": mip_gap,
            "max_nodes": max_nodes,
            "threads": threads,
        },
    }
    with open(out_master, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)

    return master

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Multi-GW MILP selector (transfers + XI + chips)")
    ap.add_argument("--team-state", required=True)
    ap.add_argument("--optimizer-input", required=True)
    ap.add_argument("--out-dir", required=True, help="Directory to write per-GW plans")
    ap.add_argument("--out-master", required=True, help="Path to write master summary JSON")
    ap.add_argument("--gws", required=True, help='GW list/range, e.g. "4-6" or "4,5,6"')

    ap.add_argument("--risk-lambda", type=float, default=0.0)
    ap.add_argument("--topk", default="GK:5,DEF:15,MID:15,FWD:10")

    ap.add_argument("--allow-hits", action="store_true")
    ap.add_argument("--max-extra-transfers", type=int, default=3)

    ap.add_argument("--transfer-bounds", help='Per-GW bounds map, e.g. "4:0-3,5:0-2"')
    ap.add_argument("--min-transfers", type=int, default=0, help="Default lower bound when map not given")
    ap.add_argument("--max-transfers", type=int, default=3, help="Default upper bound when map not given")

    ap.add_argument("--chip-plan", help='FH/WC map, e.g. "4:FH,6:WC" (TC/BB are solver-chosen)')
    ap.add_argument("--enable-tc", action="store_true", help="Allow Triple Captain search (≤1 across horizon)")
    ap.add_argument("--enable-bb", action="store_true", help="Allow Bench Boost search (≤1 across horizon)")
    ap.add_argument("--tc-multiplier", type=float, default=3.0)

    ap.add_argument("--teams-json", help="Path to _id_lookup_teams.json for team code validation")
    ap.add_argument("--price-scale", type=float, default=None,
                    help="Override auto price scale (e.g., 0.1 if prices are in tenths)")
    ap.add_argument("--freeze-prices", action="store_true",
                    help="Use snapshot prices for all future GWs (no price changes).")
    ap.add_argument("--freeze-prices-gw", type=int, default=None,
                    help="GW to snapshot prices from; defaults to team_state['gw'] if omitted.")

    # CBC controls
    ap.add_argument("--time-limit", type=int, default=None,
                    help="CBC wall time limit in seconds (e.g., 600 = 10 min)")
    ap.add_argument("--mip-gap", type=float, default=None,
                    help="Relative MIP optimality gap target (e.g., 0.005 = 0.5%)")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="Max branch-and-bound nodes for CBC")
    ap.add_argument("--threads", type=int, default=None,
                    help="CBC threads (default: solver decides)")

    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gws = _parse_gws(args.gws)
    bounds_map = _parse_bounds_map(args.transfer_bounds)
    chip_plan = _parse_chip_plan(args.chip_plan)
    bounds = _bounds_per_gw(gws, bounds_map, args.min_transfers, args.max_transfers)

    summary = solve_multi_gw(
        team_state_path=args.team_state,
        optimizer_input_path=args.optimizer_input,
        out_dir=args.out_dir,
        out_master=args.out_master,
        gws=gws,
        risk_lambda=args.risk_lambda,
        topk=_parse_topk(args.topk),
        allow_hits=bool(args.allow_hits),
        max_extra_transfers=int(args.max_extra_transfers),
        transfer_bounds=bounds,
        chip_plan=chip_plan,
        enable_tc=bool(args.enable_tc),   # FIXED: no inversion
        enable_bb=bool(args.enable_bb),   # FIXED: no inversion
        tc_multiplier=args.tc_multiplier,
        teams_json=args.teams_json,
        price_scale=args.price_scale,
        freeze_prices=bool(args.freeze_prices),
        freeze_prices_gw=args.freeze_prices_gw,
        # CBC limits
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        max_nodes=args.max_nodes,
        threads=args.threads,
        verbose=bool(args.verbose),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
