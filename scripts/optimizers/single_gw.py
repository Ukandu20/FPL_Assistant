#!/usr/bin/env python3
"""
Single-GW selector (MILP) — chip-aware, sweep runner, team validator, rich JSON + Markdown output.

Core features:
• EV auto-detection: picks the first non-empty among ['exp_pts_mean','xPts','xpts','exp_points'].
• 'ha' renamed to 'venue' across output objects. No 'is_home' in output.
• Strict FPL formations only (3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-3-2, 5-4-1) via selector binaries.
• Team cap ≤3, legal squad composition (2/5/5/3).
• Chips: NONE, TC, BB, WC, FH.
• Sweep runner: NONE/TC/BB by FT/FT+hits; WC/FH try K=1..15 exact transfers (hits=0).
• Outputs: transfers split into "out" and "in", fdr as int (or null), bench order+gk, xi (sorted) and xi_by_pos, team_summary.

New in this version:
• --ban <csv>: hard exclude player_ids.
• --lock <csv>: force player_ids to start in XI.
• --soft-lock <csv>: discourage **ownership** (apply EV penalty if in squad; not just if starting).
• --soft-lock-penalty <float>: penalty magnitude (default 5.0) applied per soft-locked player owned.
• --preview N: show quick preview of pool, XI, and transfers in stdout.
• --report md: write a shareable Markdown file next to the JSON plan (works in sweep).
• Sweep skips infeasible K's automatically and **reports infeasible K's** (especially useful when --lock is set).
"""
from __future__ import annotations

import argparse, json, os, re
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

try:
    import pulp  # type: ignore
except Exception:
    raise SystemExit("pulp is required (pip install pulp).")

TEAM_COL_CANDIDATES = ("team", "team_quota_key")
MAX_FREE_TRANSFERS_STACK = 5

OPP_COLS = ("opponent", "opponent_id", "is_home")
FDR_COLS = ("fdr",)

# Valid formations per FPL
VALID_FORMATIONS: Set[Tuple[int, int, int]] = {
    (3, 4, 3), (3, 5, 2),
    (4, 3, 3), (4, 4, 2), (4, 5, 1),
    (5, 3, 2), (5, 4, 1),
}

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


def _parse_topk(s: str) -> Dict[str, int]:
    out = {"GK": 20, "DEF": 60, "MID": 60, "FWD": 40}
    if not s:
        return out
    for part in s.split(","):
        k, v = part.strip().split(":")
        out[k.strip().upper()] = int(v)
    return out


def _parse_csv_ids(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


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


def _is_na_like(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x)
    except Exception:
        return x is None


def _to_str(x) -> str:
    return "" if _is_na_like(x) else str(x)


def _first_nonempty_str(*vals) -> str:
    for v in vals:
        s = _to_str(v).strip()
        if s:
            return s
    return ""


def _validate_teams(df: pd.DataFrame) -> None:
    team_col = _get_team_col_name(df)
    if "team_id" not in df.columns:
        raise ValueError("optimizer_input missing 'team_id'")
    chk = df.copy()
    chk[team_col] = chk[team_col].astype(str).str.strip().str.upper()
    chk["team_id"] = chk["team_id"].astype(str)

    nunique = chk.groupby("team_id")[team_col].nunique()
    bad_ids = nunique[nunique > 1]
    if not bad_ids.empty:
        details = (chk[chk["team_id"].isin(bad_ids.index)]
                   .groupby(["team_id", team_col]).size()
                   .reset_index(name="rows").sort_values(["team_id", "rows"], ascending=[True, False]))
        raise ValueError("Inconsistent team code per team_id:\n" + details.to_string(index=False))


def _preflight_required_transfers(df: pd.DataFrame, state: dict, gw: int) -> tuple:
    team_col = _get_team_col_name(df)
    owned = pd.DataFrame(state.get("squad", []))
    if owned.empty:
        raise ValueError("team_state.squad is empty.")
    owned["player_id"] = owned["player_id"].astype(str).str.strip()

    present = (df[["player_id", team_col, "pos"]]
               .assign(player_id=lambda x: x["player_id"].astype(str).str.strip()))
    j = owned.merge(present, on="player_id", how="left", suffixes=("_state", ""))

    missing = j[j[team_col].isna()]
    missing_ids = missing["player_id"].tolist()

    counts = (j.dropna(subset=[team_col])[team_col].astype(str).str.upper().value_counts())
    teamcap_excess_lb = int(sum(max(0, int(c) - 3) for c in counts.tolist()))

    lower_bound = max(len(missing_ids), teamcap_excess_lb)
    return missing_ids, teamcap_excess_lb, lower_bound, j


def _derive_code_and_id_from_state_row(row: pd.Series) -> Tuple[str, str]:
    code = _first_nonempty_str(row.get("team_state"), row.get("team")).upper()
    canon_id = _first_nonempty_str(row.get("team_id_state"), row.get("team_id"))
    if not code or not canon_id:
        raise ValueError(
            "Cannot derive team code/id from state row; ensure team_state.json is migrated "
            "(team_id canonical alphanumeric; short code in 'team')."
        )
    return code, canon_id


def _preflight_budget_lb(df: pd.DataFrame, state: dict, gw: int, exact_transfers: Optional[int], missing_ids: List[str]) -> None:
    if exact_transfers is None:
        return
    E = int(exact_transfers)
    M = len(missing_ids)
    if M == 0 or E <= 0:
        return

    sells_allowed = max(0, E - M)
    present = df.assign(player_id=lambda x: x["player_id"].astype(str))
    owned_ids = {str(p["player_id"]) for p in state.get("squad", [])}
    non_owned = present[~present["player_id"].isin(owned_ids)].copy()

    cheapest_buys_cost = (non_owned["price"].astype(float).nsmallest(E).sum()
                          if len(non_owned) >= E else float("inf"))

    id_to_sell = {str(p["player_id"]): float(p.get("sell_price", p.get("price", 0.0))) for p in state.get("squad", [])}
    present_owned = present[present["player_id"].isin(owned_ids)].copy()
    present_owned["sell_val"] = present_owned["player_id"].map(id_to_sell).astype(float)
    best_proceeds = present_owned["sell_val"].nlargest(sells_allowed).sum() if sells_allowed > 0 else 0.0

    bank = float(state.get("bank", 0.0))
    if bank + best_proceeds + 1e-9 < cheapest_buys_cost:
        raise ValueError(
            "Infeasible by budget lower bound with missing-owned and exact transfers.\n"
            f"- missing_owned = {M} → sells forced to E-M = {sells_allowed}\n"
            f"- bank + max proceeds from {sells_allowed} sells = {bank + best_proceeds:.1f}\n"
            f"- cheapest cost of {E} buys = {cheapest_buys_cost:.1f}\n"
            "Fix: include missing owned in optimizer_input or use --pad-missing-owned."
        )


# --------------- core MILP ----------------
def solve_single_gw(
    team_state_path: str,
    optimizer_input_path: str,
    out_path: str,
    gw: Optional[int] = None,
    risk_lambda: float = 0.0,
    topk: Optional[Dict[str, int]] = None,
    allow_hits: bool = True,
    max_extra_transfers: int = 3,
    cap_cannot_equal_vice: bool = True,
    formation_bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = ((3, 5), (2, 5), (1, 3)),  # unused; kept for API compat
    chip: Optional[str] = None,         # None|WC|FH|TC|BB
    tc_multiplier: float = 3.0,
    verbose: bool = False,
    # Deterministic transfer controls
    exact_transfers: Optional[int] = None,
    max_total_transfers: Optional[int] = None,
    min_total_transfers: Optional[int] = None,
    # Preflight/diagnostics
    pad_missing_owned: bool = False,
    dry_run_validate: bool = False,
    # Locks/Bans
    ban_ids: Optional[List[str]] = None,
    lock_ids: Optional[List[str]] = None,
    soft_lock_ids: Optional[List[str]] = None,
    soft_lock_penalty: float = 5.0,
    # Preview/Report
    preview: int = 0,
    report_kind: Optional[str] = None,
) -> dict:
    # State
    with open(team_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    bank = float(state.get("bank", 0.0))
    free_transfers_before = int(state.get("free_transfers", 1))
    season = str(state.get("season"))
    snapshot_gw = int(state.get("gw"))
    squad_owned: Set[str] = {str(p["player_id"]) for p in state.get("squad", [])}
    owned_sell_map: Dict[str, float] = {
        str(p["player_id"]): float(p.get("sell_price", p.get("price", 0.0)))
        for p in state.get("squad", [])
    }

    ban_ids = ban_ids or []
    lock_ids = lock_ids or []
    soft_lock_ids = soft_lock_ids or []

    # Data
    df = _read_any(optimizer_input_path)

    # Required core columns
    core_required = [
        "season", "gw", "player_id", "team_id", "pos", "price", "sell_price",
        "p60", "exp_pts_var", "cs_prob", "is_dgw", "captain_uplift", "player",
        "team", "opponent", "opponent_id", "is_home"
    ]
    missing = [c for c in core_required if c not in df.columns]
    if missing:
        raise ValueError(f"optimizer_input missing columns: {missing}")

    # EV detection (accept several column names)
    ev_candidates = ["exp_pts_mean", "xPts", "xpts", "exp_points"]
    ev_col = None
    for c in ev_candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any() and (s.fillna(0).abs().sum() > 0):
                ev_col = c
                break
    if ev_col is None:
        df["__ev__"] = 0.0
        ev_col = "__ev__"

    # Optional FDR
    has_fdr = any(c in df.columns for c in FDR_COLS)
    fdr_col = next((c for c in FDR_COLS if c in df.columns), None)

    team_col_name = _get_team_col_name(df)

    # Filter GW
    if gw is not None:
        df = df[df["gw"] == gw].copy()
        if df.empty:
            raise ValueError(f"No optimizer_input rows for gw={gw}")
    else:
        gws = sorted(df["gw"].unique().tolist())
        if len(gws) != 1:
            raise ValueError("optimizer_input has multiple GWs; specify --gw")
        gw = int(gws[0])

    # Preflight feasibility
    missing_ids, teamcap_lb, lb, owned_view = _preflight_required_transfers(df, state, gw)

    # Optional auto-pad missing owned — preserve dtypes
    if missing_ids and pad_missing_owned:
        pads: List[dict] = []
        all_cols = list(df.columns)
        for pid_m in missing_ids:
            row = owned_view.loc[owned_view["player_id"] == pid_m].iloc[0]
            code, canon_team_id = _derive_code_and_id_from_state_row(row)
            pos_state = _first_nonempty_str(row.get("pos_state"), row.get("pos")).upper()
            name = _first_nonempty_str(row.get("name"))
            if not pos_state:
                raise ValueError(f"Cannot pad missing owned {pid_m}: missing pos in state.")
            buy_p = float(row.get("buy_price") or row.get("sell_price") or 0.0)
            sell_p = float(row.get("sell_price") or row.get("buy_price") or 0.0)

            pad_row = {c: np.nan for c in all_cols}
            pad_row.update({
                "season": season, "gw": gw, "player_id": pid_m,
                team_col_name: code, "team_id": str(canon_team_id), "pos": pos_state,
                "price": buy_p, "sell_price": sell_p,
                "p60": 0.0, ev_col: 0.0, "exp_pts_var": 0.0,
                "cs_prob": 0.0, "is_dgw": 0, "captain_uplift": 0.0,
                "player": name,
                "opponent": np.nan, "opponent_id": np.nan, "is_home": np.nan,
            })
            if has_fdr and fdr_col:
                pad_row[fdr_col] = np.nan
            pads.append(pad_row)

        if pads:
            pad_df = pd.DataFrame(pads, columns=all_cols)
            df = pd.concat([df, pad_df], ignore_index=True)
            # Recompute bounds after padding
            missing_ids, teamcap_lb, lb, owned_view = _preflight_required_transfers(df, state, gw)

    # Dry-run diagnostics
    if dry_run_validate:
        diag = {
            "gw": gw,
            "owned_missing_count": len(missing_ids),
            "owned_missing_ids": missing_ids,
            "teamcap_excess_lower_bound": teamcap_lb,
            "min_required_transfers": lb
        }
        print(json.dumps(diag, indent=2))
        return {"diagnostics": diag}

    # Exact K feasibility
    if exact_transfers is not None and lb > int(exact_transfers):
        raise ValueError(
            f"Infeasible: exact_transfers={exact_transfers} but minimum required={lb} "
            f"(missing_owned={len(missing_ids)}, teamcap_lb={teamcap_lb}). "
            f"{'Use --pad-missing-owned or fix optimizer_input.' if missing_ids else 'Adjust K or fix team caps.'}"
        )

    if missing_ids and not pad_missing_owned:
        _preflight_budget_lb(df, state, gw, exact_transfers, missing_ids)

    _validate_teams(df)

    # Candidate pool: all owned + topK per pos by EV
    df["owned"] = df["player_id"].astype(str).isin(squad_owned)
    df[team_col_name] = df[team_col_name].astype(str).str.strip().str.upper()

    keep_rows: List[pd.DataFrame] = []
    kmap = topk or {"GK": 20, "DEF": 60, "MID": 60, "FWD": 40}
    for pos_name, g in df.groupby("pos", as_index=False):
        g = g.sort_values(["owned", ev_col], ascending=[False, False])
        k = kmap.get(pos_name, 50)
        keep_rows.append(pd.concat([g[g["owned"]], g[~g["owned"]].head(k)], ignore_index=True))
    pool = pd.concat(keep_rows, ignore_index=True).drop_duplicates("player_id").reset_index(drop=True)

    # Arrays / safe dtypes
    pid = pool["player_id"].astype(str).tolist()
    pos = pool["pos"].astype(str).tolist()
    teams = pool[team_col_name].astype(str).tolist()
    ownedm = pool["owned"].astype(bool).to_numpy()
    price = pd.to_numeric(pool["price"], errors="coerce").fillna(0.0).astype(float).to_numpy()

    # EV and variance
    ev_series = pd.to_numeric(pool[ev_col], errors="coerce").fillna(0.0)
    ev = ev_series.astype(float).to_numpy()

    var_series = pd.to_numeric(pool["exp_pts_var"], errors="coerce").fillna(0.0).clip(lower=0.0)
    var = var_series.astype(float).to_numpy()

    capup = pd.to_numeric(pool["captain_uplift"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float).to_numpy()

    # Opponent metadata
    opp = pool["opponent"].astype(object).where(~pool["opponent"].isna(), None).tolist()

    def _to_bool_or_none(v):
        if _is_na_like(v):
            return None
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return bool(int(v))
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "t", "1", "h", "home"}:
                return True
            if s in {"false", "f", "0", "a", "away"}:
                return False
        return None

    is_home_vals = [_to_bool_or_none(v) for v in pool["is_home"].tolist()]
    venue = [("H" if b is True else ("A" if b is False else None)) for b in is_home_vals]

    fdr_vals = None
    if has_fdr and fdr_col:
        fdr_vals = pd.to_numeric(pool[fdr_col], errors="coerce").where(~pool[fdr_col].isna(), None).tolist()

    names: List[Optional[str]] = [None] * len(pool)
    if "player" in pool.columns:
        names = [None if pd.isna(v) else str(v) for v in pool["player"].astype(object)]

    # index maps
    id2idx = {pid[i]: i for i in range(len(pid))}

    # Check locks present in pool
    missing_locks = [x for x in lock_ids if x not in id2idx]
    if missing_locks:
        raise ValueError(f"--lock contains ids not present in candidate pool (gw={gw}): {missing_locks}")
    # bans not in pool are fine

    N = len(pool)
    m = pulp.LpProblem("single_gw_selector", pulp.LpMaximize)

    # Vars
    in_squad = pulp.LpVariable.dicts("in_squad", range(N), 0, 1, cat=pulp.LpBinary)
    buy = pulp.LpVariable.dicts("buy", range(N), 0, 1, cat=pulp.LpBinary)
    sell = pulp.LpVariable.dicts("sell", range(N), 0, 1, cat=pulp.LpBinary)
    start = pulp.LpVariable.dicts("start", range(N), 0, 1, cat=pulp.LpBinary)
    cap = pulp.LpVariable.dicts("cap", range(N), 0, 1, cat=pulp.LpBinary)
    vcap = pulp.LpVariable.dicts("vcap", range(N), 0, 1, cat=pulp.LpBinary)
    hits = pulp.LpVariable("hits", lowBound=0, upBound=max_extra_transfers, cat=pulp.LpInteger)

    bench_ranks = [1, 2, 3]
    bench = {r: pulp.LpVariable.dicts(f"bench_r{r}", range(N), 0, 1, cat=pulp.LpBinary) for r in bench_ranks}

    # Ownership transitions
    for i in range(N):
        if ownedm[i]:
            m += in_squad[i] + sell[i] == 1, f"owned_transition_{i}"
            m += buy[i] == 0, f"cant_buy_owned_{i}"
        else:
            m += in_squad[i] == buy[i], f"new_transition_{i}"
            m += sell[i] == 0, f"cant_sell_not_owned_{i}"

    # Hard bans: in_squad == 0
    for bid in ban_ids:
        if bid in id2idx:
            i = id2idx[bid]
            m += in_squad[i] == 0, f"ban_in_squad_{bid}"

    # Squad size & composition
    def _sum_pos(pname: str, vec):
        return pulp.lpSum(vec[i] for i in range(N) if pos[i] == pname)

    m += pulp.lpSum(in_squad[i] for i in range(N)) == 15, "squad_size_15"
    m += _sum_pos("GK", in_squad) == 2, "comp_gk_2"
    m += _sum_pos("DEF", in_squad) == 5, "comp_def_5"
    m += _sum_pos("MID", in_squad) == 5, "comp_mid_5"
    m += _sum_pos("FWD", in_squad) == 3, "comp_fwd_3"

    # Team cap ≤3
    for t in sorted(set(teams)):
        m += pulp.lpSum(in_squad[i] for i in range(N) if teams[i] == t) <= 3, f"teamcap_{t}"

    # XI & formation (strict)
    m += pulp.lpSum(start[i] for i in range(N)) == 11, "xi_size_11"
    for i in range(N):
        m += start[i] <= in_squad[i], f"start_in_squad_{i}"
    m += _sum_pos("GK", start) == 1, "xi_gk_1"

    # counts
    DEF_cnt = _sum_pos("DEF", start)
    MID_cnt = _sum_pos("MID", start)
    FWD_cnt = _sum_pos("FWD", start)

    # formation selector binaries, one per VALID_FORMATIONS
    y_form = {f: pulp.LpVariable(f"form_{f[0]}_{f[1]}_{f[2]}", lowBound=0, upBound=1, cat=pulp.LpBinary)
              for f in sorted(list(VALID_FORMATIONS))}
    m += pulp.lpSum(y_form[f] for f in y_form) == 1, "one_valid_formation"

    # tie position counts to exactly one chosen tuple
    m += DEF_cnt == pulp.lpSum(f[0] * y_form[f] for f in y_form), "def_count_match_formation"
    m += MID_cnt == pulp.lpSum(f[1] * y_form[f] for f in y_form), "mid_count_match_formation"
    m += FWD_cnt == pulp.lpSum(f[2] * y_form[f] for f in y_form), "fwd_count_match_formation"

    # Locks: force start==1 for specified ids
    for lid in lock_ids:
        i = id2idx[lid]
        m += start[i] == 1, f"lock_start_{lid}"

    # C/VC
    m += pulp.lpSum(cap[i] for i in range(N)) == 1, "one_captain"
    m += pulp.lpSum(vcap[i] for i in range(N)) == 1, "one_vice"
    for i in range(N):
        m += cap[i] <= start[i]
        m += vcap[i] <= start[i]
        if cap_cannot_equal_vice:
            m += cap[i] + vcap[i] <= 1

    # Transfers & hits (chip-aware)
    transfers_cnt = pulp.lpSum(buy[i] for i in range(N))
    chip_norm = (chip or "").upper() or None

    def _int_or_none(x):
        return None if x is None else int(x)

    E = _int_or_none(exact_transfers)
    Tmax = _int_or_none(max_total_transfers)
    Tmin = _int_or_none(min_total_transfers)

    if chip_norm in {"WC", "FH"}:
        m += hits == 0, "hits_zero_chip"
        if E is not None:
            if E < 0:
                raise ValueError("--exact-transfers must be >= 0")
            m += transfers_cnt == E, "exact_transfers"
        else:
            if Tmax is not None:
                m += transfers_cnt <= Tmax, "max_total_transfers"
            if Tmin is not None:
                m += transfers_cnt >= Tmin, "min_total_transfers"
    else:
        if E is not None:
            if E < 0:
                raise ValueError("--exact-transfers must be >= 0")
            m += transfers_cnt == E, "exact_transfers"
            if not allow_hits and E > free_transfers_before:
                raise ValueError("exact_transfers > free_transfers but --allow-hits is not set")
            if E <= free_transfers_before or not allow_hits:
                m += hits == 0, "hits_exact_zero"
            else:
                m += hits == (E - free_transfers_before), "hits_exact_match"
                m += hits <= max_extra_transfers, "hits_cap_exact"
        else:
            if not allow_hits:
                m += transfers_cnt <= free_transfers_before, "no_hits_allowed"
                m += hits == 0, "hits_zero"
            else:
                m += hits >= transfers_cnt - free_transfers_before, "hits_lb"
                m += hits <= max_extra_transfers, "hits_cap"
                m += transfers_cnt <= free_transfers_before + max_extra_transfers, "transfer_cap_total"
            if Tmax is not None:
                m += transfers_cnt <= Tmax, "max_total_transfers"
            if Tmin is not None:
                m += transfers_cnt >= Tmin, "min_total_transfers"

    # Budget
    proceeds_expr = pulp.lpSum(sell[i] * float((owned_sell_map.get(pid[i], 0.0))) for i in range(N))
    cost_expr = pulp.lpSum(buy[i] * float(price[i]) for i in range(N))
    m += cost_expr <= bank + proceeds_expr, "budget"

    # Bench ordering (outfield only)
    bench_ranks = [1, 2, 3]
    for r in bench_ranks:
        m += pulp.lpSum(bench[r][i] for i in range(N) if pos[i] != "GK") == 1
    for i in range(N):
        if pos[i] == "GK":
            for r in bench_ranks:
                m += bench[r][i] == 0
        else:
            m += pulp.lpSum(bench[r][i] for r in bench_ranks) <= 1
            for r in bench_ranks:
                m += bench[r][i] <= in_squad[i] - start[i]

    # Objective (chip-aware) + soft-lock ownership penalty (discourages owning these ids)
    team_ev_term = pulp.lpSum(start[i] * ev[i] for i in range(N))
    cap_uplift = pulp.lpSum(cap[i] * capup[i] for i in range(N))
    if chip_norm == "TC":
        factor = max(0.0, float(tc_multiplier) - 1.0)  # 3x => +2x uplift
        cap_uplift = pulp.lpSum(cap[i] * (factor * capup[i]) for i in range(N))
    if chip_norm == "BB":
        team_ev_term = pulp.lpSum(in_squad[i] * ev[i] for i in range(N))

    # soft-lock penalty applies to OWNERSHIP (in_squad), not just starting
    soft_ids_set = set(soft_lock_ids)
    soft_pen_vec = []
    for i in range(N):
        if pid[i] in soft_ids_set:
            soft_pen_vec.append(in_squad[i] * float(soft_lock_penalty))
    soft_pen_term = pulp.lpSum(soft_pen_vec) if soft_pen_vec else 0.0

    obj = team_ev_term + cap_uplift - float(risk_lambda) * pulp.lpSum(start[i] * var[i] for i in range(N)) - soft_pen_term
    if chip_norm not in {"WC", "FH"}:
        obj = obj - 4.0 * hits
    m += obj

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=bool(verbose))
    res = m.solve(solver)
    if pulp.LpStatus[res] != "Optimal":
        msg = f"MILP not optimal: status={pulp.LpStatus[res]}"
        if lock_ids:
            msg += " | Hint: a LOCK may conflict with budget, 3-per-team, formation, or exact K; try higher K or use --soft-lock."
        raise RuntimeError(msg)

    # Extract helpers
    def picks(mask):
        return [pid[i] for i in range(N) if pulp.value(mask[i]) > 0.5]

    def _int_or_none_fdr(v):
        if v is None or _is_na_like(v):
            return None
        try:
            return int(float(v))
        except Exception:
            return None

    def to_player_obj(xid: Optional[str]) -> Optional[dict]:
        if xid is None:
            return None
        i = id2idx[xid]
        base = {
            "id": xid,
            "name": (None if names[i] is None else names[i]),
            "pos": pos[i],
            "team": teams[i],
            "xPts": _round1(ev[i]),
            "opp": opp[i],
            "venue": venue[i],  # 'H'/'A' or None
        }
        if has_fdr and fdr_col:
            base["fdr"] = _int_or_none_fdr(fdr_vals[i])
        return base

    xi_ids = picks(start)
    cap_pid = picks(cap)[0]
    vcap_pid = picks(vcap)[0]
    buy_ids = picks(buy)
    sell_ids = picks(sell)

    # Bench order & bench GK
    outfield = [i for i in range(N) if pos[i] != "GK"]
    bench_order_ids: List[Optional[str]] = [None, None, None]
    for r in bench_ranks:
        for i in outfield:
            if pulp.value(bench[r][i]) > 0.5:
                bench_order_ids[r - 1] = pid[i]
                break
    gk_ids = [pid[i] for i in range(N) if pos[i] == "GK" and pulp.value(in_squad[i]) > 0.5]
    gk_start = [pid[i] for i in range(N) if pos[i] == "GK" and pulp.value(start[i]) > 0.5]
    gk_bench = [x for x in gk_ids if x not in gk_start]
    bench_gk_id = gk_bench[0] if gk_bench else None

    # Maps
    name_map = {pid[i]: (None if names[i] is None else names[i]) for i in range(N)}
    pos_map = {pid[i]: pos[i] for i in range(N)}
    team_map = {pid[i]: teams[i] for i in range(N)}
    ev_map = {pid[i]: float(ev[i]) for i in range(N)}
    price_map = {pid[i]: float(price[i]) for i in range(N)}
    sell_map = {pid[i]: float(owned_sell_map.get(pid[i], 0.0)) for i in range(N)}
    opp_map = {pid[i]: opp[i] for i in range(N)}
    venue_map = {pid[i]: venue[i] for i in range(N)}
    fdr_map = {pid[i]: (None if (not has_fdr or fdr_vals is None) else fdr_vals[i]) for i in range(N)}

    # Transfers enriched (split OUT and IN; no is_home in output)
    transfers_out_list: List[dict] = []
    for out_id in sell_ids:
        sell_value = float(sell_map.get(out_id, 0.0))
        payload_out = {
            "id": out_id,
            "name": name_map.get(out_id),
            "pos": pos_map.get(out_id),
            "team": team_map.get(out_id),
            "xPts": _round1(ev_map.get(out_id)),
            "opp": opp_map.get(out_id),
            "venue": venue_map.get(out_id),
            "fdr": _int_or_none_fdr(fdr_map.get(out_id)),
            "sell_value": sell_value,
        }
        transfers_out_list.append(payload_out)

    transfers_in_list: List[dict] = []
    for in_id in buy_ids:
        buy_price = float(price_map.get(in_id, 0.0))
        payload_in = {
            "id": in_id,
            "name": name_map.get(in_id),
            "pos": pos_map.get(in_id),
            "team": team_map.get(in_id),
            "xPts": _round1(ev_map.get(in_id)),
            "opp": opp_map.get(in_id),
            "venue": venue_map.get(in_id),
            "fdr": _int_or_none_fdr(fdr_map.get(in_id)),
            "buy_price": buy_price,
        }
        transfers_in_list.append(payload_in)

    # Objective parts (chip-aware EV term for reporting)
    if chip_norm == "BB":
        ev_start = float(sum(ev[i] * pulp.value(in_squad[i]) for i in range(N)))
    else:
        ev_start = float(sum(ev[i] * pulp.value(start[i]) for i in range(N)))
    if chip_norm == "TC":
        factor = max(0.0, float(tc_multiplier) - 1.0)
        ev_cap = float(sum((factor * capup[i]) * pulp.value(cap[i]) for i in range(N)))
    else:
        ev_cap = float(sum(capup[i] * pulp.value(cap[i]) for i in range(N)))
    var_pen = float(risk_lambda * sum(var[i] * pulp.value(start[i]) for i in range(N)))
    # Ownership-based soft-lock penalty value (count of soft-ids owned * penalty)
    soft_pen_val = float(sum((pid[i] in soft_ids_set) * pulp.value(in_squad[i]) for i in range(N))) * float(soft_lock_penalty)
    hits_val = 0.0 if chip_norm in {"WC", "FH"} else float(4.0 * pulp.value(hits))
    total = ev_start + ev_cap - var_pen - soft_pen_val - hits_val

    # Budget math
    buys_cost = sum(price_map[x] for x in buy_ids)
    sells_proceeds = sum(sell_map[x] for x in sell_ids)
    bank_after = bank + sells_proceeds - buys_cost

    # Formation string from chosen y_form
    def _val(v): return float(pulp.value(v))
    chosen = None
    for f, y in y_form.items():
        if _val(y) > 0.5:
            chosen = f
            break
    if chosen is None:
        # fallback (should not happen)
        nDEF = int(sum(1 for i in range(N) if pos[i] == "DEF" and pulp.value(start[i]) > 0.5))
        nMID = int(sum(1 for i in range(N) if pos[i] == "MID" and pulp.value(start[i]) > 0.5))
        nFWD = int(sum(1 for i in range(N) if pos[i] == "FWD" and pulp.value(start[i]) > 0.5))
        formation_str = f"{nDEF}-{nMID}-{nFWD}"
    else:
        formation_str = f"{chosen[0]}-{chosen[1]}-{chosen[2]}"

    # Binding detection (heuristic)
    bindings: List[str] = []
    if abs(sum(pulp.value(in_squad[i]) for i in range(N)) - 15) <= 1e-6:
        bindings.append("squad_size")
    if abs(sum(pulp.value(start[i]) for i in range(N)) - 11) <= 1e-6:
        bindings.append("xi_size")
    lhs_cost = sum(pulp.value(buy[i]) * price[i] for i in range(N))
    rhs_budget = bank + sum(pulp.value(sell[i]) * (owned_sell_map.get(pid[i], 0.0)) for i in range(N))
    if abs(lhs_cost - rhs_budget) <= 1e-5 or lhs_cost > rhs_budget - 1e-5:
        bindings.append("budget")
    for t in sorted(set(teams)):
        team_count = sum(pulp.value(in_squad[i]) for i in range(N) if teams[i] == t)
        if team_count >= 3 - 1e-5:
            bindings.append("3-per-team")
            break

    if exact_transfers is not None:
        bindings.append("exact_transfers")
    if (max_total_transfers is not None):
        bindings.append("max_total_transfers")
    if (min_total_transfers is not None):
        bindings.append("min_total_transfers")
    if chip_norm not in {"WC", "FH"} and allow_hits:
        bindings.append("transfer_cap_total")

    # Free transfer accounting
    transfers_used = len(buy_ids)
    if chip_norm in {"WC", "FH"}:
        free_used = 0
        free_after = free_transfers_before
        free_next = min(MAX_FREE_TRANSFERS_STACK, free_after + 1)
        extra_used = 0
    else:
        free_used = min(free_transfers_before, transfers_used)
        free_after = max(0, free_transfers_before - free_used)
        free_next = min(MAX_FREE_TRANSFERS_STACK, free_after + 1)
        extra_used = max(0, transfers_used - free_transfers_before)

    # Team summaries
    def _team_counts(mask_vec):
        counts: Dict[str, int] = {}
        for i in range(N):
            if pulp.value(mask_vec[i]) > 0.5:
                counts[teams[i]] = counts.get(teams[i], 0) + 1
        return [{"team": t, "count": int(c)} for t, c in sorted(counts.items())]

    squad_team_summary = _team_counts(in_squad)
    xi_team_summary = _team_counts(start)

    # Compute XI objects once; then group/sort
    xi_objs = [to_player_obj(x) for x in xi_ids]
    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    xi_sorted = sorted(xi_objs, key=lambda r: pos_order.get(r["pos"], 9) if r else 9)
    xi_by_pos = {
        "GK": [p for p in xi_objs if p and p["pos"] == "GK"],
        "DEF": [p for p in xi_objs if p and p["pos"] == "DEF"],
        "MID": [p for p in xi_objs if p and p["pos"] == "MID"],
        "FWD": [p for p in xi_objs if p and p["pos"] == "FWD"],
    }

    meta_locks = {
        "ban": ban_ids,
        "lock": lock_ids,
        "soft_lock": soft_lock_ids,
    }

    # Compose plan JSON
    plan = {
        "objective": {
            "ev": _round1(ev_start),             # base EV (XI or all-15 for BB)
            "hit_cost": _round1(hits_val),       # 4 * hits (0 for WC/FH)
            "risk_penalty": _round1(var_pen),    # lambda * variance term
            "soft_lock_penalty": _round1(soft_pen_val),  # ownership penalty sum
            "total": _round1(total),             # ev + cap_uplift - risk_penalty - soft_lock_penalty - hit_cost
        },
        "meta": {
            "season": season,
            "gw": gw,
            "snapshot_gw": snapshot_gw,
            "formation": formation_str,
            "bank_before": _round1(bank),
            "bank_after": _round1(bank_after),
            "team_summary": {
                "squad": squad_team_summary,
                "xi": xi_team_summary,
            },
            "locks": meta_locks,
            "budget": {
                "buys_cost": _round1(buys_cost),
                "sells_proceeds": _round1(sells_proceeds),
                "net_spend": _round1(buys_cost - sells_proceeds),
            },
            "transfer_controls": {
                "allow_hits": bool(allow_hits),
                "max_extra_transfers": int(max_extra_transfers),
                "exact_transfers": (None if exact_transfers is None else int(exact_transfers)),
                "max_total_transfers": (None if max_total_transfers is None else int(max_total_transfers)),
                "min_total_transfers": (None if min_total_transfers is None else int(min_total_transfers)),
                "free_transfers_before": free_transfers_before,
                "free_transfers_used": free_used,
                "free_transfers_after": free_after,
                "free_transfers_next": free_next,
                "max_free_transfers_stack": MAX_FREE_TRANSFERS_STACK,
                "transfers_used": transfers_used,
                "extra_transfers": extra_used,
                "hits_charged": int(pulp.value(hits)),
            },
        },
        "chip": chip_norm or None,
        "transfers": {
            "out": transfers_out_list,
            "in": transfers_in_list
        },
        "xi": xi_sorted,
        "xi_by_pos": xi_by_pos,
        "bench": {"order": [to_player_obj(x) for x in bench_order_ids], "gk": to_player_obj(bench_gk_id)},
        "captain": to_player_obj(cap_pid),
        "vice": to_player_obj(vcap_pid),
        "explanations": {
            "binding_constraints": sorted(set(bindings)),
            "notes": [
                f"EV source column: {ev_col}",
                "Captain adds captain_uplift on top of XI EV (TC scales uplift).",
                "WC/FH: hits=0; K is enforced only if provided (sweep will run many exact K).",
                "BB: counts EV of all 15 players.",
                "Outfield bench ranks are 1..3; GK bench is implied.",
                "Team column validated from optimizer_input.",
                "Next-GW free transfers projection assumes +1 carry, capped at 5.",
                "Player objects expose opponent and venue ('H'/'A'); no is_home field in output.",
                "Formation is restricted to the FPL-valid set: "
                + ", ".join([f"{d}-{m}-{f}" for (d, m, f) in sorted(list(VALID_FORMATIONS))]),
                "Soft-lock applies a negative EV penalty if these players are OWNED (in squad).",
            ],
        },
    }

    # Ensure directory and write JSON
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    # Preview (stdout)
    if preview and preview > 0:
        print("\n=== PREVIEW ===")
        print(f"GW {gw} • Formation {plan['meta']['formation']} • Chip {plan['chip']}")
        print(f"Objective: EV={plan['objective']['ev']} | Cap+={_round1(ev_cap)} | Risk-={plan['objective']['risk_penalty']} | SoftOwn-={plan['objective']['soft_lock_penalty']} | Hits-={plan['objective']['hit_cost']} | Total={plan['objective']['total']}")
        print("\nXI (pos, team, name, xPts, opp, venue, fdr) — top N:")
        for p in plan["xi"][:preview]:
            print(f"  {p['pos']:<3} {p['team']:<3}  {p['name']:<20}  xPts={p['xPts']:<4}  vs {p['opp'] or '-':<3}  {p['venue'] or '-':<1}  fdr={p.get('fdr')}")
        print("\nTransfers OUT:")
        for t in plan["transfers"]["out"][:preview]:
            print(f"  OUT {t['pos']:<3} {t['team']:<3}  {t['name']:<20} sell={_round1(t['sell_value'])}  xPts={t['xPts']}  vs {t['opp'] or '-':<3} {t['venue'] or '-'} fdr={t.get('fdr')}")
        print("\nTransfers IN:")
        for t in plan["transfers"]["in"][:preview]:
            print(f"  IN  {t['pos']:<3} {t['team']:<3}  {t['name']:<20} buy={_round1(t['buy_price'])}  xPts={t['xPts']}  vs {t['opp'] or '-':<3} {t['venue'] or '-'} fdr={t.get('fdr')}")
        print("=== /PREVIEW ===\n")

    # Report
    if report_kind and report_kind.lower() == "md":
        md_path = os.path.splitext(out_path)[0] + ".md"
        _write_markdown_report(plan, md_path, infeasible_notes=None, sweep_alias=None, base_single_dir=None, add_infeasible_section=False)
        if verbose:
            print(f"[report] wrote {md_path}")

    return plan


def _write_markdown_report(plan: dict, out_md: str,
                           infeasible_notes: Optional[List[dict]],
                           sweep_alias: Optional[str],
                           base_single_dir: Optional[str],
                           add_infeasible_section: bool) -> None:
    """Emit a concise Markdown report for easy sharing. Optionally append infeasible K summary."""
    meta = plan["meta"]
    obj = plan["objective"]
    chip = plan.get("chip") or "NONE"
    xi = plan["xi"]
    bench = plan["bench"]
    transfers = plan["transfers"]

    def fmt_player(p):
        fdr = p.get("fdr")
        fdrs = "" if fdr is None else f" | FDR {fdr}"
        return f"- **{p['name']}** ({p['pos']} {p['team']}) — xPts {p['xPts']} vs {p['opp'] or '-'} {p['venue'] or ''}{fdrs}"

    def fmt_transfer_out(t):
        fdr = t.get("fdr")
        fdrs = "" if fdr is None else f" | FDR {fdr}"
        return f"- OUT: **{t['name']}** ({t['pos']} {t['team']}) — sell {t['sell_value']} | xPts {t['xPts']} vs {t['opp'] or '-'} {t['venue'] or ''}{fdrs}"

    def fmt_transfer_in(t):
        fdr = t.get("fdr")
        fdrs = "" if fdr is None else f" | FDR {fdr}"
        return f"- IN: **{t['name']}** ({t['pos']} {t['team']}) — buy {t['buy_price']} | xPts {t['xPts']} vs {t['opp'] or '-'} {t['venue'] or ''}{fdrs}"

    lines = []
    title_alias = f"{sweep_alias} — " if sweep_alias else ""
    lines.append(f"# {title_alias}GW{meta['gw']} Plan — {meta['formation']}  \n")
    lines.append(f"**Chip:** {chip}  \n")
    lines.append(f"**Objective:** EV {obj['ev']}  + Cap  − Risk {obj['risk_penalty']} − SoftOwn {obj.get('soft_lock_penalty', 0)} − Hits {obj['hit_cost']}  = **{obj['total']}**  \n")
    lines.append("")
    lines.append("## Team Summary")
    for scope in ("xi", "squad"):
        ts = meta["team_summary"].get(scope, [])
        lines.append(f"- **{scope.upper()}**: " + ", ".join([f"{x['team']}×{x['count']}" for x in ts]))
    lines.append("")
    lines.append("## XI")
    for p in xi:
        lines.append(fmt_player(p))
    lines.append("")
    lines.append("**Captain:** " + (f"{plan['captain']['name']} ({plan['captain']['team']})" if plan.get("captain") else "-"))
    lines.append("**Vice:** " + (f"{plan['vice']['name']} ({plan['vice']['team']})" if plan.get("vice") else "-"))
    lines.append("")
    lines.append("## Bench")
    bo = bench["order"]
    bg = bench["gk"]
    for i, p in enumerate(bo, start=1):
        if p:
            lines.append(f"- **B{i}**: {p['name']} ({p['pos']} {p['team']}) — xPts {p['xPts']}")
    if bg:
        lines.append(f"- **GK**: {bg['name']} ({bg['team']}) — xPts {bg['xPts']}")
    lines.append("")
    lines.append("## Transfers")
    if transfers["out"]:
        lines.append("### Out")
        for t in transfers["out"]:
            lines.append(fmt_transfer_out(t))
    if transfers["in"]:
        lines.append("### In")
        for t in transfers["in"]:
            lines.append(fmt_transfer_in(t))
    lines.append("")
    lines.append("## Budget")
    b = meta["budget"]
    lines.append(f"- Buys cost: **{b['buys_cost']}** | Sells proceeds: **{b['sells_proceeds']}** | Net spend: **{b['net_spend']}**  ")
    lines.append(f"- Bank: **{meta['bank_before']} → {meta['bank_after']}**")
    lines.append("")
    lines.append("## Controls & Notes")
    tc = meta["transfer_controls"]
    lines.append(f"- Transfers used: **{tc['transfers_used']}** (free used {tc['free_transfers_used']}, extra {tc['extra_transfers']}, hits {tc['hits_charged']})")
    lines.append(f"- Free next: **{tc['free_transfers_next']}**")
    lines.append("- " + " | ".join(plan["explanations"]["notes"]))
    if add_infeasible_section and infeasible_notes:
        lines.append("")
        lines.append("## Infeasible K Runs (skipped)")
        for item in infeasible_notes:
            lines.append(f"- **{item['chip']} {item['alias']}** — {item['reason']}")
    lines.append("")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------- sweep runner ----------------
def run_sweep(
    team_state_path: str,
    optimizer_input_path: str,
    out_base_dir: str,
    gw: int,
    allow_hits: bool,
    max_extra_transfers: int,
    sweep_free_transfers: bool,
    sweep_include_hits: bool,
    sweep_chips_csv: str,
    risk_lambda: float,
    topk: Dict[str, int],
    pad_missing_owned: bool,
    verbose: bool,
    # new:
    ban_ids: List[str],
    lock_ids: List[str],
    soft_lock_ids: List[str],
    soft_lock_penalty: float,
    preview: int,
    report_kind: Optional[str],
) -> Dict[str, str]:
    """
    Produce multiple plans in one call. Returns dict alias -> path, plus '_infeasible' list in the dict.
    Layout:
      <out_base_dir>/single/gw{gw}/
        ├─ NONE plans (e.g., 1t.json, 2t.json, ...)
        └─ chips/
            ├─ TC/TC_1t.json, ...
            ├─ BB/BB_1t.json, ...
            ├─ WC/1t.json, 2t.json, ..., 15t.json (only feasible ones)
            └─ FH/1t.json, 2t.json, ..., 15t.json (only feasible ones)
    """
    with open(team_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    free_before = int(state.get("free_transfers", 1))

    # Base K list for NONE/TC/BB
    ks: List[int] = []
    if sweep_free_transfers:
        ks.extend(range(1, max(1, free_before) + 1))
    else:
        ks.append(min(1, max(1, free_before)))

    if sweep_include_hits and allow_hits:
        ks.extend(range(free_before + 1, free_before + max(1, max_extra_transfers) + 1))

    chips_list = [c.strip().upper() for c in (sweep_chips_csv or "NONE").split(",") if c.strip()]
    chips_list = [c for c in chips_list if c in {"NONE", "TC", "BB", "FH", "WC"}]

    out_paths: Dict[str, str] = {}
    infeasible_notes: List[dict] = []

    base_single = os.path.join(out_base_dir, "single", f"gw{gw}")
    base_chips = os.path.join(base_single, "chips")

    os.makedirs(base_single, exist_ok=True)

    # Helper to record infeasible
    def _record_infeasible(chip_lbl: str, alias: str, exc: Exception):
        reason = str(exc)
        infeasible_notes.append({"chip": chip_lbl, "alias": alias, "reason": reason})
        if verbose:
            print(f"[sweep] {chip_lbl} {alias} infeasible/skipped: {reason}")

    # NONE
    if "NONE" in chips_list:
        for K in ks:
            alias = f"{K}t"
            out_path = os.path.join(base_single, f"{alias}.json")
            try:
                plan = solve_single_gw(
                    team_state_path=team_state_path,
                    optimizer_input_path=optimizer_input_path,
                    out_path=out_path,
                    gw=gw,
                    risk_lambda=risk_lambda,
                    topk=topk,
                    allow_hits=allow_hits,
                    max_extra_transfers=max_extra_transfers,
                    cap_cannot_equal_vice=True,
                    chip=None,
                    tc_multiplier=3.0,
                    verbose=verbose,
                    exact_transfers=K,
                    max_total_transfers=None,
                    min_total_transfers=None,
                    pad_missing_owned=pad_missing_owned,
                    dry_run_validate=False,
                    ban_ids=ban_ids,
                    lock_ids=lock_ids,
                    soft_lock_ids=soft_lock_ids,
                    soft_lock_penalty=soft_lock_penalty,
                    preview=preview,
                    report_kind=report_kind,
                )
                out_paths[f"NONE_{alias}"] = out_path
                if verbose:
                    print(f"[sweep] wrote NONE {alias} -> {out_path}")
                if report_kind and report_kind.lower() == "md":
                    _write_markdown_report(plan, os.path.splitext(out_path)[0] + ".md",
                                           infeasible_notes=None, sweep_alias=f"NONE {alias}",
                                           base_single_dir=base_single, add_infeasible_section=False)
            except Exception as e:
                _record_infeasible("NONE", alias, e)

    # TC / BB
    for chip in ["TC", "BB"]:
        if chip not in chips_list:
            continue
        chip_dir = os.path.join(base_chips, chip)
        os.makedirs(chip_dir, exist_ok=True)
        for K in ks:
            alias = f"{chip}_{K}t"
            out_path = os.path.join(chip_dir, f"{alias}.json")
            try:
                plan = solve_single_gw(
                    team_state_path=team_state_path,
                    optimizer_input_path=optimizer_input_path,
                    out_path=out_path,
                    gw=gw,
                    risk_lambda=risk_lambda,
                    topk=topk,
                    allow_hits=allow_hits,
                    max_extra_transfers=max_extra_transfers,
                    cap_cannot_equal_vice=True,
                    chip=chip,
                    tc_multiplier=3.0,
                    verbose=verbose,
                    exact_transfers=K,
                    max_total_transfers=None,
                    min_total_transfers=None,
                    pad_missing_owned=pad_missing_owned,
                    dry_run_validate=False,
                    ban_ids=ban_ids,
                    lock_ids=lock_ids,
                    soft_lock_ids=soft_lock_ids,
                    soft_lock_penalty=soft_lock_penalty,
                    preview=preview,
                    report_kind=report_kind,
                )
                out_paths[alias] = out_path
                if verbose:
                    print(f"[sweep] wrote {alias} -> {out_path}")
                if report_kind and report_kind.lower() == "md":
                    _write_markdown_report(plan, os.path.splitext(out_path)[0] + ".md",
                                           infeasible_notes=None, sweep_alias=alias,
                                           base_single_dir=base_single, add_infeasible_section=False)
            except Exception as e:
                _record_infeasible(chip, alias, e)

    # WC / FH: try K=1..15
    for chip in ["WC", "FH"]:
        if chip not in chips_list:
            continue
        chip_dir = os.path.join(base_chips, chip)
        os.makedirs(chip_dir, exist_ok=True)
        for K in range(1, 16):
            alias = f"{chip}_{K}t"
            out_path = os.path.join(chip_dir, f"{K}t.json")
            try:
                plan = solve_single_gw(
                    team_state_path=team_state_path,
                    optimizer_input_path=optimizer_input_path,
                    out_path=out_path,
                    gw=gw,
                    risk_lambda=risk_lambda,
                    topk=topk,
                    allow_hits=True,                   # irrelevant (hits=0 inside)
                    max_extra_transfers=999,           # irrelevant (hits=0 inside)
                    cap_cannot_equal_vice=True,
                    chip=chip,
                    tc_multiplier=3.0,
                    verbose=verbose,
                    exact_transfers=K,                 # enforce exact K on WC/FH
                    max_total_transfers=None,
                    min_total_transfers=None,
                    pad_missing_owned=pad_missing_owned,
                    dry_run_validate=False,
                    ban_ids=ban_ids,
                    lock_ids=lock_ids,
                    soft_lock_ids=soft_lock_ids,
                    soft_lock_penalty=soft_lock_penalty,
                    preview=preview,
                    report_kind=report_kind,
                )
                out_paths[alias] = out_path
                if verbose:
                    print(f"[sweep] wrote {alias} -> {out_path}")
                if report_kind and report_kind.lower() == "md":
                    _write_markdown_report(plan, os.path.splitext(out_path)[0] + ".md",
                                           infeasible_notes=None, sweep_alias=alias,
                                           base_single_dir=base_single, add_infeasible_section=False)
            except Exception as e:
                _record_infeasible(chip, alias, e)

    # If requested, write one consolidated infeasible summary markdown
    if report_kind and report_kind.lower() == "md" and infeasible_notes:
        md_path = os.path.join(base_single, "infeasible_runs.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Infeasible K Runs (skipped)\n\n")
            for item in infeasible_notes:
                f.write(f"- **{item['chip']} {item['alias']}** — {item['reason']}\n")
        if verbose:
            print(f"[report] wrote infeasible summary: {md_path}")

    # Include infeasible metadata in return
    out_paths["_infeasible"] = infeasible_notes
    return out_paths


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Single-GW MILP selector (transfers + XI + C/VC) with sweep runner")
    ap.add_argument("--team-state", required=True)
    ap.add_argument("--optimizer-input", required=True)
    ap.add_argument("--out-base", default="data/plans", help="root output dir (default: data/plans)")
    ap.add_argument("--out", help="single-plan path; ignored when sweep flags are used")
    ap.add_argument("--gw", type=int)
    ap.add_argument("--risk-lambda", type=float, default=0.0)
    ap.add_argument("--topk", default="GK:20,DEF:60,MID:60,FWD:40")
    ap.add_argument("--allow-hits", action="store_true")
    ap.add_argument("--max-extra-transfers", type=int, default=3)
    ap.add_argument("--no-cap-neq-vice", action="store_true",
                    help="allow same player as C and VC (not recommended)")
    ap.add_argument("--chip", choices=["WC", "FH", "TC", "BB"], help="Apply chip logic (single run)")
    ap.add_argument("--tc-multiplier", type=float, default=3.0, help="Triple Captain multiplier (default 3x)")
    ap.add_argument("--verbose", action="store_true")

    # Deterministic transfer controls (single run)
    ap.add_argument("--exact-transfers", type=int, help="Force exactly K transfers total")
    ap.add_argument("--max-total-transfers", type=int, help="Cap total transfers to ≤ K")
    ap.add_argument("--min-total-transfers", type=int, help="Lower bound on total transfers")

    # Preflight/diagnostics
    ap.add_argument("--pad-missing-owned", action="store_true",
                    help="Auto-inject zero-EV rows for missing owned players to keep feasibility")
    ap.add_argument("--dry-run-validate", action="store_true",
                    help="Run preflight checks and exit (no solve); prints min required transfers")

    # Sweep controls
    ap.add_argument("--sweep-free-transfers", action="store_true",
                    help="Produce K=1..FT plans")
    ap.add_argument("--sweep-include-hits", action="store_true",
                    help="Additionally produce K=FT+1..FT+max_extra_transfers (requires --allow-hits)")
    ap.add_argument("--sweep-chips", default="NONE", help='Comma list among NONE,TC,BB,FH,WC. Plans go to .../chips/<CHIP>/.')
    # Locks/Bans
    ap.add_argument("--ban", help="Comma-separated player_ids to hard BAN (exclude from squad)")
    ap.add_argument("--lock", help="Comma-separated player_ids to LOCK into starting XI")
    ap.add_argument("--soft-lock", dest="soft_lock", help="Comma-separated player_ids to discourage (apply EV penalty if OWNED)")
    ap.add_argument("--soft-lock-penalty", type=float, default=5.0, help="Penalty subtracted from objective if a soft-locked player is OWNED (default 5.0)")
    # Preview/Report
    ap.add_argument("--preview", type=int, default=0, help="Preview N rows of XI and transfers in stdout")
    ap.add_argument("--report", choices=["md"], help="Emit a shareable report alongside JSON (e.g., 'md')")

    args = ap.parse_args()

    ban_ids = _parse_csv_ids(args.ban)
    lock_ids = _parse_csv_ids(args.lock)
    soft_lock_ids = _parse_csv_ids(args.soft_lock)

    # If sweep flags are present, run sweep
    if args.sweep_free_transfers or args.sweep_include_hits or (args.sweep_chips and args.sweep_chips.upper() != "NONE"):
        if args.dry_run_validate:
            _ = solve_single_gw(
                team_state_path=args.team_state,
                optimizer_input_path=args.optimizer_input,
                out_path=os.path.join(args.out_base, "tmp_preflight.json"),
                gw=args.gw,
                risk_lambda=args.risk_lambda,
                topk=_parse_topk(args.topk),
                allow_hits=bool(args.allow_hits),
                max_extra_transfers=int(args.max_extra_transfers),
                cap_cannot_equal_vice=not args.no_cap_neq_vice,
                chip=None,
                tc_multiplier=args.tc_multiplier,
                verbose=bool(args.verbose),
                exact_transfers=1,
                pad_missing_owned=bool(args.pad_missing_owned),
                dry_run_validate=True,
                ban_ids=ban_ids,
                lock_ids=lock_ids,
                soft_lock_ids=soft_lock_ids,
                soft_lock_penalty=float(args.soft_lock_penalty),
                preview=int(args.preview),
                report_kind=args.report,
            )
        res = run_sweep(
            team_state_path=args.team_state,
            optimizer_input_path=args.optimizer_input,
            out_base_dir=args.out_base,
            gw=args.gw,
            allow_hits=bool(args.allow_hits),
            max_extra_transfers=int(args.max_extra_transfers),
            sweep_free_transfers=bool(args.sweep_free_transfers),
            sweep_include_hits=bool(args.sweep_include_hits),
            sweep_chips_csv=str(args.sweep_chips or "NONE"),
            risk_lambda=args.risk_lambda,
            topk=_parse_topk(args.topk),
            pad_missing_owned=bool(args.pad_missing_owned),
            verbose=bool(args.verbose),
            ban_ids=ban_ids,
            lock_ids=lock_ids,
            soft_lock_ids=soft_lock_ids,
            soft_lock_penalty=float(args.soft_lock_penalty),
            preview=int(args.preview),
            report_kind=args.report,
        )
        print(json.dumps(res, indent=2))
        return

    # Otherwise single plan
    out_path = args.out or os.path.join(args.out_base, "single", f"gw{args.gw}", "plan.json")
    plan = solve_single_gw(
        team_state_path=args.team_state,
        optimizer_input_path=args.optimizer_input,
        out_path=out_path,
        gw=args.gw,
        risk_lambda=args.risk_lambda,
        topk=_parse_topk(args.topk),
        allow_hits=bool(args.allow_hits),
        max_extra_transfers=int(args.max_extra_transfers),
        cap_cannot_equal_vice=not args.no_cap_neq_vice,
        chip=args.chip,
        tc_multiplier=args.tc_multiplier,
        verbose=bool(args.verbose),
        exact_transfers=args.exact_transfers,
        max_total_transfers=args.max_total_transfers,
        min_total_transfers=args.min_total_transfers,
        pad_missing_owned=bool(args.pad_missing_owned),
        dry_run_validate=bool(args.dry_run_validate),
        ban_ids=ban_ids,
        lock_ids=lock_ids,
        soft_lock_ids=soft_lock_ids,
        soft_lock_penalty=float(args.soft_lock_penalty),
        preview=int(args.preview),
        report_kind=args.report,
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
