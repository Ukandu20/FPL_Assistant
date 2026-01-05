#!/usr/bin/env python3
# apps/streamlit_app.py
from __future__ import annotations
import json, subprocess, sys, shutil, time, re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterable, Counter

import pandas as pd
import streamlit as st
from collections import Counter

# ----------------------------- CONFIG ----------------------------------------
REPO_ROOT         = Path(".").resolve()
DATA_ROOT         = REPO_ROOT / "data"
PRED_ROOT         = DATA_ROOT / "predictions"
PLANS_ROOT        = DATA_ROOT / "plans"
REGISTRY_ROOT     = DATA_ROOT / "processed" / "registry"
TEAM_STATE_PATH   = DATA_ROOT / "state" / "team_state.json"
TEAMS_JSON        = REGISTRY_ROOT / "master_teams.json"

DEFAULT_LEAGUE    = "ENG-Premier League"

# ----------------------------- HELPERS ----------------------------------------
def _load_json(p: Path, fallback: dict | None = None) -> dict:
    if not p.exists(): return fallback or {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(p)

def _backup_file(p: Path) -> None:
    if p.exists():
        backup = p.with_suffix(".bak")
        shutil.copy2(p, backup)

def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def detect_latest_season() -> Optional[str]:
    # Prefer seasons present under data/plans/<season> or data/predictions/*/<season>
    season_pat = re.compile(r"^\d{4}-\d{4}$")
    candidates: set[str] = set()
    # from plans
    if PLANS_ROOT.exists():
        for d in PLANS_ROOT.iterdir():
            if d.is_dir():
                # single/<season> or <season>/multi
                if d.name == "single":
                    for sd in d.iterdir():
                        if sd.is_dir() and season_pat.match(sd.name):
                            candidates.add(sd.name)
                elif season_pat.match(d.name):
                    candidates.add(d.name)
    # from predictions
    if PRED_ROOT.exists():
        for sub in PRED_ROOT.iterdir():
            if sub.is_dir():
                for d in sub.iterdir():
                    if d.is_dir() and season_pat.match(d.name):
                        candidates.add(d.name)
    if not candidates:
        return None
    return sorted(candidates)[-1]

@st.cache_data(show_spinner=False)
def read_table(path_csv_or_parquet: Path) -> pd.DataFrame:
    if not path_csv_or_parquet or not path_csv_or_parquet.exists():
        return pd.DataFrame()
    if path_csv_or_parquet.suffix.lower() == ".parquet":
        return pd.read_parquet(path_csv_or_parquet)
    return pd.read_csv(path_csv_or_parquet)

def find_latest_pred_path(season: str, subdir: str, stem: str) -> Optional[Path]:
    cand = []
    season_dir = PRED_ROOT / subdir / season
    if season_dir.exists():
        for ext in (".parquet", ".csv"):
            cand.append(season_dir / f"{stem}{ext}")
        gw_files = sorted(season_dir.glob("GW*.parquet")) + sorted(season_dir.glob("GW*.csv"))
        if gw_files:
            cand.insert(0, gw_files[-1])
    for ext in (".parquet", ".csv"):
        cand.append(PRED_ROOT / subdir / f"{stem}{ext}")
    return _first_existing(cand)

def find_latest_plan(season: str, mode: str) -> Optional[Path]:
    if mode == "Single-GW":
        base = PLANS_ROOT / "single" / season
        if not base.exists(): return None
        gws = sorted([d for d in base.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda p: int(p.name))
        for gw_dir in reversed(gws):
            p = gw_dir / "plan.json"
            if p.exists(): return p
            chips_dir = gw_dir / "chips"
            if chips_dir.exists():
                for chip_dir in sorted(chips_dir.iterdir(), reverse=True):
                    p = chip_dir / "plan.json"
                    if p.exists(): return p
        return None
    else:
        base = PLANS_ROOT / season / "multi" / "hold"
        if not base.exists(): return None
        windows = sorted([d for d in base.iterdir() if d.is_dir() and "gw" in d.name], key=lambda p: p.name)
        for w in reversed(windows):
            chips_root = w / "chips"
            if chips_root.exists():
                for chip_dir in sorted(chips_root.iterdir(), reverse=True):
                    for k_dir in sorted(chip_dir.iterdir(), reverse=True):
                        p = k_dir / "plan.json"
                        if p.exists(): return p
            base_root = w / "base"
            if base_root.exists():
                for k_dir in sorted(base_root.iterdir(), reverse=True):
                    p = k_dir / "plan.json"
                    if p.exists(): return p
        return None

def pretty_money(x: float | int | None) -> str:
    if x is None: return "-"
    return f"{x:.1f}"

def run_cli_stream(argv: List[str], live_area, spinner_label="Running...") -> int:
    """
    Stream subprocess stdout+stderr live into a Streamlit area.
    Returns return code.
    """
    live_area.expander("Live logs (click to expand)", expanded=True).write(" ".join(argv))
    exp = live_area.expander("Output", expanded=True)
    buf = []
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    except Exception as e:
        exp.error(f"[launcher-error] {e}")
        return 1
    with st.spinner(spinner_label):
        for line in iter(proc.stdout.readline, ''):
            buf.append(line.rstrip("\n"))
            # Keep last ~400 lines to avoid huge DOM
            if len(buf) > 400:
                buf = buf[-400:]
            exp.code("\n".join(buf))
        proc.stdout.close()
        rc = proc.wait()
    if rc == 0:
        exp.success("Completed.")
    else:
        exp.error(f"Process exited with code {rc}")
    return rc

# ---------- Registry helpers for validation (best-effort; optional files) -----
@st.cache_data(show_spinner=False)
def load_teams_map() -> dict[str, str]:
    """
    Return map of team_id -> team short name if available.
    """
    teams = _load_json(TEAMS_JSON, {})
    # Accept either {team_id:{...,"short":"ARS"}} or {"ARS":{"id":...}}
    if isinstance(teams, dict):
        # try to build reverse maps
        if "teams" in teams and isinstance(teams["teams"], list):
            out = {}
            for t in teams["teams"]:
                tid = str(t.get("team_id") or t.get("id") or t.get("code") or t.get("hex_id") or "")
                short = t.get("short") or t.get("short_name") or t.get("name") or ""
                if tid: out[tid] = short
            return out
        else:
            # maybe flat dict keyed by id
            out = {}
            for k,v in teams.items():
                if isinstance(v, dict):
                    short = v.get("short") or v.get("short_name") or v.get("name") or ""
                    out[str(k)] = short
            if out: return out
    return {}

def _norm_pos(p: str|None) -> Optional[str]:
    if not p: return None
    p = str(p).upper()
    aliases = {"GKP":"GK","GOALKEEPER":"GK","DEFENDER":"DEF","MIDFIELDER":"MID","FORWARD":"FWD","STRIKER":"FWD"}
    return aliases.get(p, p) if p in {"GK","DEF","MID","FWD"} else aliases.get(p)

def validate_squad_rules(squad: list, teams_map: dict[str,str]) -> tuple[bool, list[str]]:
    """
    Expect each squad entry to have at least:
      - player_id (any)
      - pos (GK/DEF/MID/FWD)  OR we cannot validate composition
      - team_id (string/int)  OR we cannot validate team≤3
    Returns (ok, errors[])
    """
    errs: list[str] = []
    if not isinstance(squad, list) or len(squad) != 15:
        errs.append("Squad must be a list of exactly 15 entries.")
        return False, errs

    pos_counts = Counter()
    team_counts = Counter()

    missing_pos, missing_team = [], []

    for i, item in enumerate(squad, start=1):
        if isinstance(item, dict):
            pos = _norm_pos(item.get("pos"))
            tid = item.get("team_id")
        else:
            # Unsupported minimal format → cannot validate; flag it
            pos = None
            tid = None
        if not pos:
            missing_pos.append(i)
        else:
            pos_counts[pos] += 1
        if tid is None:
            missing_team.append(i)
        else:
            team_counts[str(tid)] += 1

    # Composition
    want = {"GK":2, "DEF":5, "MID":5, "FWD":3}
    for k, need in want.items():
        if pos_counts.get(k,0) != need:
            errs.append(f"Position count invalid: {k}={pos_counts.get(k,0)} (must be {need}).")

    # Team ≤3 rule
    for tid, n in team_counts.items():
        if n > 3:
            short = teams_map.get(str(tid), str(tid))
            errs.append(f"Team cap exceeded: {short} has {n} players (max 3).")

    # Missing info warnings → make them errors (you asked “always validate”)
    if missing_pos:
        errs.append(f"Missing 'pos' for entries at indexes: {missing_pos}. Add pos for all players.")
    if missing_team:
        errs.append(f"Missing 'team_id' for entries at indexes: {missing_team}. Add team_id for all players.")

    return (len(errs) == 0), errs

# ---------- Apply plan -> team_state ------------------------------------------
def apply_plan_to_state(plan: dict, state: dict) -> dict:
    """
    Update team_state using plan.json
    - Uses plan['meta'] if present (bank_before/after, free_transfers_before/after/next, hits, chip, gw)
    - Applies 'transfers' list into state['history'] with timestamp-free records
    - Does NOT mutate squad automatically (too risky without the full player registry);
      you can extend this if your plan JSON includes final squad.
    """
    out = json.loads(json.dumps(state))  # deep copy

    meta = plan.get("meta", {})
    # Update GW/season if present
    if "season" in meta: out["season"] = meta["season"]
    if "gw" in meta: out["gw"] = meta["gw"]

    # Bank / transfers — prefer meta authoritative values if present
    if "bank_after" in meta:
        out["bank"] = float(meta["bank_after"])
    elif "budget" in meta and isinstance(meta["budget"], dict):
        # fallback: bank_before - net_spend (consistent with your JSON structure)
        try:
            b_before = float(meta.get("bank_before", out.get("bank", 0.0)))
            net_spend = float(meta["budget"].get("net_spend", 0.0))
            out["bank"] = b_before - net_spend
        except Exception:
            pass

    if "free_transfers_after" in meta:
        out["free_transfers"] = int(meta["free_transfers_after"])
    elif "transfers_used" in meta:
        try:
            before = int(meta.get("free_transfers_before", out.get("free_transfers", 1)))
            used   = int(meta.get("transfers_used", 0))
            after  = max(0, before - used)
            out["free_transfers"] = after
        except Exception:
            pass

    # Chip usage
    if "chip_used" in meta and meta["chip_used"]:
        chip = str(meta["chip_used"]).upper()
        chips = out.get("chips", {"WC": True, "FH": True, "TC": True, "BB": True})
        if chip in chips and chips[chip] is True:
            chips[chip] = False
        out["chips"] = chips

    # History append
    history = out.get("history", [])
    history.append({
        "gw": meta.get("gw"),
        "season": meta.get("season"),
        "objective": plan.get("objective", {}),
        "transfers": plan.get("transfers", []),
        "chip_used": meta.get("chip_used"),
        "bank_before": meta.get("bank_before"),
        "bank_after": out.get("bank"),
        "free_transfers_before": meta.get("free_transfers_before"),
        "free_transfers_after": out.get("free_transfers")
    })
    out["history"] = history
    return out

# ----------------------------- UI SETUP ---------------------------------------
st.set_page_config(page_title="FPL Assistant — Control Panel", layout="wide")

# Auto-detect season
auto_season = detect_latest_season() or "2025-2026"

# Sidebar
st.sidebar.title("FPL Control")
season = st.sidebar.text_input("Season", value=auto_season)
league = st.sidebar.text_input("League", value=DEFAULT_LEAGUE)
mode = st.sidebar.radio("Mode", ["Single-GW", "Multi-GW (hold)"], horizontal=False)

st.sidebar.markdown("---")
exec_mode = st.sidebar.toggle("Enable command execution", value=False,
                              help="If off, this app is read-only and won't run any scripts.")
profile_path = st.sidebar.text_input("Default profile (TOML)", value="profiles/prem_gw6_fast.toml")
as_of_gw = st.sidebar.number_input("as_of_gw (for fresh runs)", value=6, step=1, min_value=1, max_value=60)
n_future  = st.sidebar.number_input("n_future (for fresh runs)", value=3, step=1, min_value=1, max_value=10)

st.sidebar.markdown("---")
st.sidebar.caption("Reads existing outputs by default. Click run to regenerate.")

# Overview cards
st.title("FPL Assistant — Plans & State")
colA, colB, colC = st.columns(3)
with colA:
    st.subheader("Latest Predictions")
    ep_path = find_latest_pred_path(season, subdir="expected_points", stem="expected_points")
    st.write(f"`{ep_path}`" if ep_path else "No expected points output found.")
with colB:
    st.subheader("Latest Plan")
    plan_path = find_latest_plan(season, "Single-GW" if mode == "Single-GW" else "Multi-GW")
    st.write(f"`{plan_path}`" if plan_path else "No plan.json found.")
with colC:
    st.subheader("Team State")
    st.write(f"`{TEAM_STATE_PATH}`" if TEAM_STATE_PATH.exists() else "team_state.json not found.")

# Tabs
tabs = st.tabs(["📈 Predictions", "🧮 Plan", "🧾 Team State", "⚙️ Run Models/Optimizers"])

# 1) Predictions
with tabs[0]:
    st.header("Predictions preview")
    pred_kind = st.selectbox("File to preview", [
        "expected_points",
        "goals_assists",
        "minutes",
        "saves",
        "defense",
    ])
    if pred_kind == "expected_points":
        p = find_latest_pred_path(season, "expected_points", "expected_points")
    else:
        season_dir = PRED_ROOT / pred_kind / season
        p = None
        if season_dir.exists():
            gw_files = sorted(season_dir.glob("GW*.parquet")) + sorted(season_dir.glob("GW*.csv"))
            p = gw_files[-1] if gw_files else None

    if not p:
        st.info("No file found to preview.")
    else:
        df = read_table(p)
        st.caption(f"Showing: {p} — rows: {len(df):,}")
        if len(df) == 0:
            st.info("File exists but is empty.")
        else:
            # Your requested default columns:
            want = ["season","gw","player","team","opponent","fdr","venue","pred_minutes","p_return_any","xPts"]
            # Map fallbacks by availability
            fallback_map = {
                "player": ["player","player_name","player_id"],
                "team": ["team","team_name","team_id"],
                "opponent": ["opponent","opponent_name","opponent_id","opp","opp_id"]
            }
            cols: list[str] = []
            for c in want:
                if c in df.columns:
                    cols.append(c)
                elif c in fallback_map:
                    found = next((alt for alt in fallback_map[c] if alt in df.columns), None)
                    if found: cols.append(found)
            # ensure gw/season present if missing
            for base in ["season","gw"]:
                if base not in cols and base in df.columns: cols.insert(0, base)
            st.dataframe(df[cols] if cols else df.head(50), use_container_width=True)

# 2) Plan
with tabs[1]:
    st.header("Current plan preview")
    plan_mode = "Single-GW" if mode == "Single-GW" else "Multi-GW"
    p = find_latest_plan(season, plan_mode)
    if not p:
        st.info("No plan.json found.")
    else:
        plan = _load_json(p, {})
        st.caption(f"Plan file: {p}")
        if not plan:
            st.warning("Empty or unreadable plan.json")
        else:
            obj = plan.get("objective", {})
            meta = plan.get("meta", {})
            st.subheader("Objective")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("EV", obj.get("ev", 0))
            c2.metric("Hit cost", obj.get("hit_cost", 0))
            c3.metric("Risk penalty", obj.get("risk_penalty", 0))
            c4.metric("Total", obj.get("total", 0))

            st.subheader("Meta")
            st.json({k: meta.get(k) for k in [
                "season","gw","snapshot_gw","formation","free_transfers_before","free_transfers_after",
                "free_transfers_next","bank_before","bank_after","transfers_used","extra_transfers","chip_used"
            ]})

            if "transfers" in plan:
                st.subheader("Transfers")
                st.table(plan["transfers"] if isinstance(plan["transfers"], list) else [plan["transfers"]])
            if "xi" in plan:
                st.subheader("Starting XI")
                st.table(plan["xi"])
            if "bench" in plan:
                st.subheader("Bench")
                st.table(plan["bench"])

            st.markdown("---")
            # Apply -> team_state
            if st.button("Apply plan → team_state.json", type="primary"):
                try:
                    state0 = _load_json(TEAM_STATE_PATH, {})
                    if not state0:
                        st.error("team_state.json not found or empty. Create it in the Team State tab first.")
                    else:
                        new_state = apply_plan_to_state(plan, state0)
                        _backup_file(TEAM_STATE_PATH)
                        _save_json(TEAM_STATE_PATH, new_state)
                        st.success("Applied plan to team_state.json (bank/FT/chip/history).")
                except Exception as e:
                    st.error(f"Failed to apply plan: {e}")

# 3) Team State (with validation)
with tabs[2]:
    st.header("Edit team_state.json (validated)")
    teams_map = load_teams_map()
    state = _load_json(TEAM_STATE_PATH, fallback={
        "season": season,
        "gw": 1,
        "bank": 0.0,
        "free_transfers": 1,
        "chips": {"WC": True, "FH": True, "TC": True, "BB": True},
        "squad": [],   # list of dicts {player_id, pos, team_id, buy_price?, sell_price?}
        "captain": None,
        "vice_captain": None,
        "history": []
    })

    col1, col2, col3 = st.columns(3)
    with col1:
        state["season"] = st.text_input("Season", value=state.get("season", season))
        state["gw"] = st.number_input("Current GW", value=int(state.get("gw", 1)), step=1, min_value=1, max_value=60)
        state["bank"] = float(st.number_input("Bank", value=float(state.get("bank", 0.0)), step=0.1))
    with col2:
        state["free_transfers"] = int(st.number_input("Free Transfers", value=int(state.get("free_transfers", 1)), step=1, min_value=0, max_value=5))
        chips = state.get("chips", {"WC": True, "FH": True, "TC": True, "BB": True})
        st.markdown("**Chips available**")
        chips["WC"] = st.checkbox("Wildcard", value=bool(chips.get("WC", True)))
        chips["FH"] = st.checkbox("Free Hit", value=bool(chips.get("FH", True)))
        chips["TC"] = st.checkbox("Triple Captain", value=bool(chips.get("TC", True)))
        chips["BB"] = st.checkbox("Bench Boost", value=bool(chips.get("BB", True)))
        state["chips"] = chips
    with col3:
        state["captain"] = st.text_input("Captain (player_id)", value=str(state.get("captain") or ""))
        state["vice_captain"] = st.text_input("Vice (player_id)", value=str(state.get("vice_captain") or ""))

    st.markdown("**Squad (JSON list — requires `pos` and `team_id` for validation)**")
    squad_str = st.text_area("Entries like: {\"player_id\": 123, \"pos\":\"DEF\", \"team_id\": 14}", value=json.dumps(state.get("squad", []), indent=2), height=240)
    valid_squad_json = True
    try:
        squad_val = json.loads(squad_str) if squad_str.strip() else []
        state["squad"] = squad_val
    except Exception as e:
        valid_squad_json = False
        st.error(f"Invalid squad JSON: {e}")

    # Validation
    ok, errs = (False, ["Invalid squad JSON"]) if not valid_squad_json else validate_squad_rules(state.get("squad", []), teams_map)
    if ok:
        st.success("Squad is valid ✅ (2 GK, 5 DEF, 5 MID, 3 FWD; ≤3 per club).")
    else:
        for e in errs:
            st.error(e)

    save_col1, save_col2 = st.columns(2)
    with save_col1:
        if st.button("Save team_state.json", type="primary", disabled=not (valid_squad_json and ok)):
            try:
                _backup_file(TEAM_STATE_PATH)
                _save_json(TEAM_STATE_PATH, state)
                st.success(f"Saved → {TEAM_STATE_PATH}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    with save_col2:
        if st.button("Reload from disk"):
            st.cache_data.clear()
            st.rerun()

# 4) Runners (live logs)
with tabs[3]:
    st.header("Run forecasters / optimizers (on demand)")
    if not exec_mode:
        st.info("Read-only mode is ON. Toggle 'Enable command execution' in the sidebar to run scripts.")
    else:
        live_area = st.container()

        st.subheader("Forecasters")
        f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
        if f_col1.button("Minutes"):
            cmd = [sys.executable, "-m", "scripts.models.minutes_forecast",
                   "--future-season", season, "--as-of", "now", "--as-of-tz", "Africa/Lagos",
                   "--as-of-gw", str(as_of_gw), "--n-future", str(n_future),
                   "--out-format", "both", "--confirm"]
            if profile_path.strip(): cmd += ["--profile", profile_path]
            rc = run_cli_stream(cmd, live_area, "Running minutes_forecast...")
            st.toast("Minutes run complete." if rc == 0 else "Minutes run failed.", icon="✅" if rc==0 else "❌")

        if f_col2.button("Goals & Assists"):
            cmd = [sys.executable, "-m", "scripts.models.goals_assists_forecast",
                   "--future-season", season, "--as-of", "now", "--as-of-tz", "Africa/Lagos",
                   "--as-of-gw", str(as_of_gw), "--n-future", str(n_future),
                   "--out-format", "both", "--confirm", "--apply-calibration", "--skip-gk"]
            if profile_path.strip(): cmd += ["--profile", profile_path]
            rc = run_cli_stream(cmd, live_area, "Running goals_assists_forecast...")
            st.toast("GA run complete." if rc == 0 else "GA run failed.", icon="✅" if rc==0 else "❌")

        if f_col3.button("Defense"):
            cmd = [sys.executable, "-m", "scripts.models.defense_forecast",
                   "--future-season", season, "--as-of", "now", "--as-of-tz", "Africa/Lagos",
                   "--as-of-gw", str(as_of_gw), "--n-future", str(n_future),
                   "--out-format", "both", "--confirm"]
            if profile_path.strip(): cmd += ["--profile", profile_path]
            rc = run_cli_stream(cmd, live_area, "Running defense_forecast...")
            st.toast("DEF run complete." if rc == 0 else "DEF run failed.", icon="✅" if rc==0 else "❌")

        if f_col4.button("Saves"):
            cmd = [sys.executable, "-m", "scripts.models.saves_forecast",
                   "--future-season", season, "--as-of", "now", "--as-of-tz", "Africa/Lagos",
                   "--as-of-gw", str(as_of_gw), "--n-future", str(n_future),
                   "--out-format", "both", "--confirm"]
            if profile_path.strip(): cmd += ["--profile", profile_path]
            rc = run_cli_stream(cmd, live_area, "Running saves_forecast...")
            st.toast("SAV run complete." if rc == 0 else "SAV run failed.", icon="✅" if rc==0 else "❌")

        if f_col5.button("Expected Points"):
            cmd = [sys.executable, "-m", "scripts.models.points_forecast",
                   "--future-season", season, "--as-of", "now", "--as-of-tz", "Africa/Lagos",
                   "--as-of-gw", str(as_of_gw), "--n-future", str(n_future),
                   "--out-format", "both", "--confirm"]
            if profile_path.strip(): cmd += ["--profile", profile_path]
            rc = run_cli_stream(cmd, live_area, "Running points_forecast...")
            st.toast("Points run complete." if rc == 0 else "Points run failed.", icon="✅" if rc==0 else "❌")

        st.divider()
        st.subheader("Optimizers")
        if mode == "Single-GW":
            gw = st.number_input("GW to optimize", value=as_of_gw, step=1, min_value=1, max_value=60)
            if st.button("Run Single-GW plan"):
                cmd = [sys.executable, "-m", "scripts.optimizers.single_gw",
                       "--gw", str(gw), "--season", season, "--confirm"]
                if profile_path.strip(): cmd += ["--profile", profile_path]
                rc = run_cli_stream(cmd, live_area, "Running single_gw...")
                st.toast("Single-GW done." if rc == 0 else "Single-GW failed.", icon="✅" if rc==0 else "❌")
        else:
            gw_from = st.number_input("GW from", value=as_of_gw, step=1, min_value=1, max_value=60)
            gw_to   = st.number_input("GW to", value=as_of_gw+2, step=1, min_value=1, max_value=60)
            k_sweep = st.selectbox("K sweep", ["K1","K3","K5","K6"], index=0)
            if st.button("Run Multi-GW (hold)"):
                cmd = [sys.executable, "-m", "scripts.optimizers.multi_gw_hold",
                       "--gw-from", str(gw_from), "--gw-to", str(gw_to),
                       "--season", season, "--k", k_sweep, "--confirm"]
                if profile_path.strip(): cmd += ["--profile", profile_path]
                rc = run_cli_stream(cmd, live_area, "Running multi_gw_hold...")
                st.toast("Multi-GW done." if rc == 0 else "Multi-GW failed.", icon="✅" if rc==0 else "❌")

        st.divider()
        if st.button("Refresh page data (no runs)"):
            st.cache_data.clear()
            st.rerun()
