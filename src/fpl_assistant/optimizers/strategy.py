#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Evaluator v0.1 — transfers vs wildcard via MC over horizon H

What this does
--------------
Given a set of candidate rosters per action (A0..A5, AWC), each represented as
per-GW MC-ready sim_input Parquets, this script:

1) For each candidate roster and each GW in [gw_start .. gw_start+H-1]:
   • Chooses the captain *ex-ante* by evaluating each XI player as (C) with the SAME RNG seed
     (common random numbers), then locking the best by EV (SD tiebreak if EV within eps).
   • Runs the mc_sim_v01.run_sim on the resulting sim_input.
   • Extracts the per-sim team points array from the returned samples_df (sum over players).
2) Sums team points arrays across the horizon, subtracts the hit cost for that action.
3) Computes metrics: EV, SD, VaR@10, CVaR@10, P(>X) thresholds.
4) Picks the best candidate within each action (EV primary, lower SD tiebreak),
   then ranks actions and writes a JSON+CSV report.

Assumptions & choices (v0.1)
----------------------------
• Bench policy: like-for-like autosubs (handled in mc_sim_v01).
• Captain choice: per GW per candidate roster (not per-sim), using a fair tie.
• Horizon independence: we use independent RNG seeds per GW; within a GW, captain
  comparisons use common seeds for fairness.
• Candidate generation: out of scope here — feed us the candidate sim_input files
  your optimizer emits.

Directory layout expected
-------------------------
candidates_root/
  A0/hold/           gw4.parquet gw5.parquet ...
  A1/plan_a/         gw4.parquet gw5.parquet ...
  A1/plan_b/         gw4.parquet gw5.parquet ...
  A2/plan_c/         gw4.parquet ...
  AWC/wc_squad_01/   gw4.parquet ...

File schema (each gwX.parquet)
------------------------------
One row per (player, fixture) with the columns required by mc_sim_v01.run_sim:
[
  'gw','fixture_id','team_id','opp_team_id','player_id','player','pos',
  'xg_share','xa_share','lambda_goals_for','lambda_goals_against',
  'pred_minutes','p60','pred_saves_mean','is_start_xi',
  'is_captain','is_vice'
]
Notes:
• Captain/Vice will be overwritten by this script during evaluation. Only XI/bench
  flags are consumed from file.
• If a candidate has DGW fixtures, include them as multiple rows per player.

CLI example
-----------
python strategy_evaluator.py \
  --candidates-root decisions/candidates \
  --gw-start 4 \
  --h 3 \
  --ft-bank 3 \
  --allow-wc true \
  --actions A0,A1,A2,A3,AWC \
  --limit-k 20 \
  --nsims 20000 \
  --captain-ev-eps 0.2 \
  --objective ev --sd-tiebreak \
  --out-json decisions/gw04_strategy_report.json \
  --out-csv decisions/gw04_strategy_report.csv

Integration hooks (later)
-------------------------
• If you want the evaluator to call your MILP automatically: plug into
  `load_candidates()` to synthesize candidates on the fly and write temp Parquets.
• If you move to formation-aware autosubs: adjust mc_sim_v01 first.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ---- MC engine import -------------------------------------------------------
try:
    import mc_sim_v01  # must expose run_sim(df, cfg, write_player_samples_path=None) and SimConfig
except Exception as e:
    raise SystemExit(
        "ERROR: Could not import mc_sim_v01. Ensure mc_sim_v01.py is on PYTHONPATH.\n"
        f"Underlying import error: {e}"
    )


# ---- Data structures --------------------------------------------------------

@dataclass
class Candidate:
    action: str          # e.g., "A0", "A1", ..., "AWC"
    name: str            # subfolder name under action
    gw_files: Dict[int, Path]  # gw -> parquet file path
    m_transfers: Optional[int] = None  # deduced from action
    hit_cost: int = 0
    ft_next: int = 0     # FT bank next week after making m transfers

@dataclass
class CandidateResult:
    candidate: Candidate
    chosen_captains: Dict[int, str]  # gw -> player_id
    team_points: np.ndarray          # length = nsims; summed over horizon (hit applied)
    metrics: Dict[str, float]        # ev, sd, var10, cvar10, thresholds...
    per_gw_points: Dict[int, np.ndarray]  # optional diagnostics


# ---- Utilities --------------------------------------------------------------

def _list_candidate_dirs(candidates_root: Path, actions_filter: Optional[List[str]]) -> Dict[str, List[Path]]:
    """
    Return map: action -> list of candidate subdirectories.
    Example:
      { 'A1': [ Path('.../A1/plan_a'), Path('.../A1/plan_b') ], 'AWC': [ Path('.../AWC/wc1') ], ... }
    """
    out: Dict[str, List[Path]] = {}
    if actions_filter is None:
        actions = [p.name for p in candidates_root.iterdir() if p.is_dir()]
    else:
        actions = actions_filter

    for a in actions:
        adir = candidates_root / a
        if not adir.exists() or not adir.is_dir():
            continue
        kids = [p for p in adir.iterdir() if p.is_dir()]
        kids.sort(key=lambda p: p.name)
        out[a] = kids
    return out


def _parse_action_to_m(action: str) -> Optional[int]:
    """Return number of transfers for 'A0'...'A5'. For 'AWC' return None."""
    if action.upper().startswith("A") and action.upper() != "AWC":
        try:
            return int(action[1:])
        except ValueError:
            return None
    return None  # AWC or others


def _compute_hit_and_ftnext(ft_bank_now: int, m_transfers: Optional[int], is_wc: bool) -> Tuple[int, int]:
    if is_wc:
        return 0, 1  # WC uses no hits and resets future FT to 1 next week
    if m_transfers is None:
        raise ValueError("m_transfers must be provided for non-WC actions")
    extra = max(0, m_transfers - ft_bank_now)
    hit = -4 * extra
    ft_next = min(5, max(0, ft_bank_now - m_transfers) + 1)
    return hit, ft_next


def _load_gw_files(candidate_dir: Path, gw_list: List[int]) -> Dict[int, Path]:
    """
    Expect files named like: gw4.parquet, gw5.parquet, ...
    """
    mapping: Dict[int, Path] = {}
    for gw in gw_list:
        f = candidate_dir / f"gw{gw}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"Missing file for candidate '{candidate_dir.name}': {f}")
        mapping[gw] = f
    return mapping


def _team_points_from_samples(samples_df: pd.DataFrame) -> np.ndarray:
    """
    mc_sim_v01 returns a samples_df with columns: ['player_id','player','pos','sim','points']
    We sum by 'sim' to get team totals. Return as numpy array aligned to sim index order (0..S-1).
    """
    required = {'player_id', 'sim', 'points'}
    if not required.issubset(set(samples_df.columns)):
        raise ValueError(f"samples_df missing columns; has {list(samples_df.columns)}")

    # Ensure all sims present from 0..max
    grp = samples_df.groupby('sim', sort=True)['points'].sum()
    # Fill missing sims (shouldn't happen)
    S = int(samples_df['sim'].max()) + 1
    out = np.zeros(S, dtype=float)
    out[grp.index.to_numpy(dtype=int)] = grp.to_numpy()
    return out


def _choose_captain_for_gw(
    df: pd.DataFrame,
    cfg: mc_sim_v01.SimConfig,
    base_seed: int,
    captain_ev_eps: float = 0.2,
    sd_tiebreak: bool = True,
) -> Tuple[str, np.ndarray]:
    """
    Evaluate each XI player as captain, using the same RNG seed for fairness.
    Return (chosen_captain_id, team_points_array_for_gw).
    """
    xi = df[df['is_start_xi'] == True].copy()
    if xi.empty:
        raise ValueError("No starters (is_start_xi=True) found in sim_input for this GW.")

    # If vice not provided, pick vice as the XI player with max pred_minutes (excluding captain).
    # We'll set per-captain below.

    best_cap = None
    best_ev = -1e9
    best_sd = 1e9
    best_team_points = None

    # Evaluate each XI player as the captain
    for pid in xi['player_id'].unique().tolist():
        df_cap = df.copy()

        # Reset captain/vice flags
        df_cap['is_captain'] = False
        df_cap['is_vice'] = False

        # Set selected captain
        mask_pid = df_cap['player_id'] == pid
        df_cap.loc[mask_pid, 'is_captain'] = True

        # Pick vice: highest pred_minutes in XI excluding captain (fallback to any XI if pred_minutes missing)
        xi_other = df_cap[(df_cap['is_start_xi'] == True) & (~mask_pid)]
        if 'pred_minutes' in xi_other.columns and not xi_other['pred_minutes'].isna().all():
            vice_id = xi_other.sort_values('pred_minutes', ascending=False)['player_id'].iloc[0]
        else:
            vice_id = xi_other['player_id'].iloc[0]
        df_cap.loc[df_cap['player_id'] == vice_id, 'is_vice'] = True

        # Run MC with SAME seed for fairness across captain choices
        local_cfg = mc_sim_v01.SimConfig(
            n_sims=cfg.n_sims,
            seed=base_seed,  # common random numbers
            minutes_sigma=cfg.minutes_sigma,
            cameo_nonzero_prob=cfg.cameo_nonzero_prob,
            p_assist_per_goal=cfg.p_assist_per_goal,
            triple_captain_transfers=cfg.triple_captain_transfers,
            enforce_no_self_assist=cfg.enforce_no_self_assist,
            write_player_samples=True,
        )
        summary, samples = mc_sim_v01.run_sim(df_cap, local_cfg, write_player_samples_path=None)
        if samples is None:
            raise ValueError("mc_sim_v01 returned no samples_df; ensure write_player_samples=True.")
        team_points = _team_points_from_samples(samples)

        ev = float(team_points.mean())
        sd = float(team_points.std(ddof=1))

        # Compare by EV, tie by SD if within eps
        if ev > best_ev + 1e-12:
            best_ev = ev
            best_sd = sd
            best_cap = pid
            best_team_points = team_points
        elif abs(ev - best_ev) <= captain_ev_eps and sd_tiebreak:
            if sd < best_sd - 1e-12:
                best_sd = sd
                best_cap = pid
                best_team_points = team_points

    assert best_cap is not None
    assert best_team_points is not None
    return best_cap, best_team_points


def _metrics_from_array(arr: np.ndarray, horizon: int) -> Dict[str, float]:
    ev = float(arr.mean())
    sd = float(arr.std(ddof=1))
    var10 = float(np.quantile(arr, 0.10))
    cvar10 = float(arr[arr <= var10].mean()) if (arr <= var10).any() else var10
    # Thresholds (sensible defaults)
    if horizon == 1:
        p60 = float((arr > 60.0).mean())
        p80 = float((arr > 80.0).mean())
        return dict(ev=round(ev, 3), sd=round(sd, 3),
                    var10=round(var10, 3), cvar10=round(cvar10, 3),
                    p_gt_60=round(p60, 4), p_gt_80=round(p80, 4))
    else:
        thr = 200.0 if horizon <= 3 else 300.0
        pthr = float((arr > thr).mean())
        return dict(ev=round(ev, 3), sd=round(sd, 3),
                    var10=round(var10, 3), cvar10=round(cvar10, 3),
                    p_gt_threshold=round(pthr, 4), threshold=thr)


# ---- Core evaluation --------------------------------------------------------

def evaluate_candidate(
    cand: Candidate,
    gw_list: List[int],
    nsims: int,
    captain_ev_eps: float,
    sd_tiebreak: bool,
    minutes_sigma: float = 12.0,
    cameo_nonzero_prob: float = 0.5,
    p_assist_per_goal: float = 0.75,
    triple_captain_transfers: bool = True,
    enforce_no_self_assist: bool = False,
    base_seed: int = 42,
) -> CandidateResult:
    """
    For a single candidate, choose captains per GW, run MC per GW with common numbers for captain choice,
    sum team points across the horizon, subtract hit once (hit_cost on candidate), and compute metrics.
    """
    cfg = mc_sim_v01.SimConfig(
        n_sims=nsims,
        seed=base_seed,  # will be overridden per-GW for independence across GWs
        minutes_sigma=minutes_sigma,
        cameo_nonzero_prob=cameo_nonzero_prob,
        p_assist_per_goal=p_assist_per_goal,
        triple_captain_transfers=triple_captain_transfers,
        enforce_no_self_assist=enforce_no_self_assist,
        write_player_samples=True,
    )

    per_gw_points: Dict[int, np.ndarray] = {}
    chosen_caps: Dict[int, str] = {}

    # Evaluate each GW independently (independent seeds across GWs)
    for i, gw in enumerate(gw_list):
        df = pd.read_parquet(cand.gw_files[gw])

        # Ensure booleans are booleans (robustness)
        for col in ['is_start_xi', 'is_captain', 'is_vice']:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        # Fair captain evaluation with common random numbers
        gw_seed = base_seed + 1000 * (gw - gw_list[0])  # stable per-GW independence
        cap_id, team_points_gw = _choose_captain_for_gw(
            df=df, cfg=cfg, base_seed=gw_seed,
            captain_ev_eps=captain_ev_eps, sd_tiebreak=sd_tiebreak
        )
        chosen_caps[gw] = cap_id
        per_gw_points[gw] = team_points_gw

    # Sum across horizon (same length arrays)
    sims_len = len(next(iter(per_gw_points.values())))
    total_arr = np.zeros(sims_len, dtype=float)
    for gw in gw_list:
        total_arr += per_gw_points[gw]

    # Apply hit once (GW start)
    total_arr += cand.hit_cost  # hit_cost is negative or zero

    metrics = _metrics_from_array(total_arr, horizon=len(gw_list))
    return CandidateResult(
        candidate=cand,
        chosen_captains=chosen_caps,
        team_points=total_arr,
        metrics=metrics,
        per_gw_points=per_gw_points,
    )


def pick_best_candidate(results: List[CandidateResult], ev_eps_for_sd: float = 1e-9) -> CandidateResult:
    """
    Select best by EV, with lower SD tiebreak if EV within eps.
    """
    assert results
    best = results[0]
    for r in results[1:]:
        ev_b, sd_b = best.metrics['ev'], best.metrics['sd']
        ev_r, sd_r = r.metrics['ev'], r.metrics['sd']
        if ev_r > ev_b + 1e-12:
            best = r
        elif abs(ev_r - ev_b) <= ev_eps_for_sd:
            if sd_r < sd_b - 1e-12:
                best = r
    return best


# ---- Loading candidates -----------------------------------------------------

def load_candidates(
    candidates_root: Path,
    gw_start: int,
    horizon: int,
    ft_bank_now: int,
    actions_filter: Optional[List[str]],
    limit_k: Optional[int],
) -> Dict[str, List[Candidate]]:
    """
    Walk the directory tree and construct Candidate objects, computing hit_cost & ft_next from action and ft bank.
    """
    gw_list = list(range(gw_start, gw_start + horizon))
    action_dirs = _list_candidate_dirs(candidates_root, actions_filter)

    out: Dict[str, List[Candidate]] = {}
    for action, cand_dirs in action_dirs.items():
        if not cand_dirs:
            continue

        # Optional top-K per action
        if limit_k is not None and len(cand_dirs) > limit_k:
            cand_dirs = cand_dirs[:limit_k]

        m = _parse_action_to_m(action)
        is_wc = (action.upper() == "AWC")
        hit, ft_next = _compute_hit_and_ftnext(ft_bank_now, m, is_wc)

        cands: List[Candidate] = []
        for cdir in cand_dirs:
            gw_files = _load_gw_files(cdir, gw_list)
            cands.append(Candidate(
                action=action,
                name=cdir.name,
                gw_files=gw_files,
                m_transfers=m,
                hit_cost=hit,
                ft_next=ft_next
            ))
        out[action] = cands

    return out


# ---- Main orchestration -----------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Transfers vs Wildcard strategy evaluator via MC")
    ap.add_argument("--candidates-root", required=True, help="Root folder containing action subfolders (A0..A5, AWC)")
    ap.add_argument("--gw-start", type=int, required=True, help="Starting GW (e.g., 4)")
    ap.add_argument("--h", type=int, default=3, help="Horizon length (1..6)")
    ap.add_argument("--ft-bank", type=int, required=True, help="Free transfers currently banked (0..5)")
    ap.add_argument("--actions", default=None, help="Comma-separated subset of actions to evaluate (e.g., A0,A1,A2,AWC)")
    ap.add_argument("--limit-k", type=int, default=None, help="Optional cap of candidates per action")
    ap.add_argument("--nsims", type=int, default=20000, help="Number of MC sims per evaluation")
    ap.add_argument("--captain-ev-eps", type=float, default=0.2, help="EV tie window within which SD breaks ties")
    ap.add_argument("--sd-tiebreak", action="store_true", help="Use lower SD to break EV ties (recommended)")
    ap.add_argument("--minutes-sigma", type=float, default=12.0, help="σ for minutes truncated normal when started")
    ap.add_argument("--cameo-nonzero-prob", type=float, default=0.5, help="Prob of non-zero cameo if not started")
    ap.add_argument("--p-assist-per-goal", type=float, default=0.75, help="Probability a goal has an assist")
    ap.add_argument("--triple-captain-transfers", action="store_true", help="Transfer TC multiplier to vice on C DNP")
    ap.add_argument("--enforce-no-self-assist", action="store_true", help="Disallow scorer=assister (slower if enabled)")
    ap.add_argument("--base-seed", type=int, default=42, help="Base RNG seed")
    ap.add_argument("--objective", choices=["ev", "cvar10", "ev_sd"], default="ev",
                    help="Primary objective for ranking actions (EV, CVaR10, or EV with SD tiebreak)")
    ap.add_argument("--out-json", required=True, help="Path to write JSON report")
    ap.add_argument("--out-csv", required=False, help="Optional CSV summary path")
    args = ap.parse_args()

    root = Path(args.candidates_root)
    if not root.exists():
        raise SystemExit(f"candidates-root not found: {root}")

    if not (1 <= args.h <= 6):
        raise SystemExit("H (horizon) must be between 1 and 6.")

    actions_filter = None
    if args.actions:
        actions_filter = [s.strip() for s in args.actions.split(",") if s.strip()]

    # Load candidate definitions from disk
    candidates_by_action = load_candidates(
        candidates_root=root,
        gw_start=args.gw_start,
        horizon=args.h,
        ft_bank_now=args.ft_bank,
        actions_filter=actions_filter,
        limit_k=args.limit_k,
    )

    if not candidates_by_action:
        raise SystemExit("No candidates found. Check --candidates-root structure and --actions filter.")

    gw_list = list(range(args.gw_start, args.gw_start + args.h))

    # Evaluate all candidates
    all_results: Dict[str, List[CandidateResult]] = {}
    for action, cands in candidates_by_action.items():
        results: List[CandidateResult] = []
        for cand in cands:
            res = evaluate_candidate(
                cand=cand,
                gw_list=gw_list,
                nsims=args.nsims,
                captain_ev_eps=args.captain_ev_eps,
                sd_tiebreak=args.sd_tiebreak or (args.objective == "ev_sd"),
                minutes_sigma=args.minutes_sigma,
                cameo_nonzero_prob=args.cameo_nonzero_prob,
                p_assist_per_goal=args.p_assist_per_goal,
                triple_captain_transfers=args.triple_captain_transfers,
                enforce_no_self_assist=args.enforce_no_self_assist,
                base_seed=args.base_seed,
            )
            results.append(res)
        all_results[action] = results

    # Pick best per action per your objective
    best_per_action: Dict[str, CandidateResult] = {}
    for action, results in all_results.items():
        if args.objective == "cvar10":
            # Replace EV with CVaR10 for primary comparison; SD tiebreak still on SD
            # We reuse pick_best_candidate by temporarily mapping ev <- cvar10
            cloned = []
            for r in results:
                r2 = CandidateResult(
                    candidate=r.candidate,
                    chosen_captains=r.chosen_captains,
                    team_points=r.team_points,
                    metrics=dict(r.metrics),
                    per_gw_points=r.per_gw_points,
                )
                r2.metrics['ev'], r2.metrics['sd'] = r.metrics['cvar10'], r.metrics['sd']
                cloned.append(r2)
            best = pick_best_candidate(cloned, ev_eps_for_sd=1e-9)
            # Best refers to cloned ev=cvar; we want original
            # Find original by candidate identity
            orig = next(r for r in results if r.candidate.name == best.candidate.name and r.candidate.action == best.candidate.action)
            best_per_action[action] = orig
        else:
            best_per_action[action] = pick_best_candidate(results, ev_eps_for_sd=1e-9)

    # Rank actions
    ranked = sorted(best_per_action.values(), key=lambda r: (r.metrics['ev'], -r.metrics['sd']), reverse=True)

    # Compose report
    report = {
        "config": {
            "gw_start": args.gw_start,
            "horizon": args.h,
            "ft_bank_now": args.ft_bank,
            "hits_per_extra": -4,
            "objective": args.objective,
            "captain_ev_tie_eps": args.captain_ev_eps,
            "sd_tiebreak": (args.sd_tiebreak or args.objective == "ev_sd"),
            "nsims": args.nsims,
            "bench_policy": "likeforlike",
        },
        "gw_list": gw_list,
        "actions": [],
        "notes": "EV primary with SD tiebreak unless objective=cvar10."
    }

    # Fill actions
    for res in ranked:
        a = res.candidate.action
        m = res.candidate.m_transfers if res.candidate.m_transfers is not None else "∞"
        hit = res.candidate.hit_cost
        ft_next = res.candidate.ft_next
        entry = {
            "action": a,
            "candidate": res.candidate.name,
            "m_transfers": m,
            "hit": hit,
            "ft_next": ft_next,
            "metrics": res.metrics,
            "captains": res.chosen_captains,
        }
        report["actions"].append(entry)

    # Write JSON
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Optional CSV summary
    if args.out_csv:
        rows = []
        for res in ranked:
            row = dict(
                action=res.candidate.action,
                candidate=res.candidate.name,
                m_transfers=res.candidate.m_transfers if res.candidate.m_transfers is not None else -1,
                hit=res.candidate.hit_cost,
                ft_next=res.candidate.ft_next,
                ev=res.metrics['ev'],
                sd=res.metrics['sd'],
                var10=res.metrics['var10'],
                cvar10=res.metrics['cvar10'],
            )
            # threshold fields vary by H; include if present
            for k in ('p_gt_60', 'p_gt_80', 'p_gt_threshold', 'threshold'):
                if k in res.metrics:
                    row[k] = res.metrics[k]
            rows.append(row)
        df = pd.DataFrame(rows)
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    # Print a short console summary
    print(f"Wrote report: {out_json}")
    if args.out_csv:
        print(f"Wrote CSV: {out_csv}")
    print("\nTop actions (by EV, SD tiebreak):")
    for i, res in enumerate(ranked[:5], 1):
        m = res.candidate.m_transfers if res.candidate.m_transfers is not None else "∞"
        print(f"{i}. {res.candidate.action} / {res.candidate.name} | m={m} hit={res.candidate.hit_cost} ft_next={res.candidate.ft_next} "
              f"| EV={res.metrics['ev']:.2f} SD={res.metrics['sd']:.2f} CVaR10={res.metrics['cvar10']:.2f}")

if __name__ == "__main__":
    main()
