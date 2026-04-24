from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from fpl_assistant.platform.paths import ARTIFACTS_ROOT, DATA_ROOT, data_path

DEFAULT_LEAGUE = "ENG-Premier League"
PRED_ROOT = data_path("predictions") if data_path("predictions").exists() else ARTIFACTS_ROOT / "predictions"
PLANS_ROOT = data_path("plans") if data_path("plans").exists() else ARTIFACTS_ROOT / "plans"
REGISTRY_ROOT = data_path("processed", "registry")
TEAM_STATE_PATH = data_path("state", "team_state.json")
TEAMS_JSON = REGISTRY_ROOT / "master_teams.json"


def load_json(path: Path, fallback: dict | None = None) -> dict:
    if not path.exists():
        return fallback or {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def backup_file(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_suffix(".bak"))


def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def detect_latest_season() -> Optional[str]:
    season_pat = re.compile(r"^\d{4}-\d{4}$")
    candidates: set[str] = set()
    if PLANS_ROOT.exists():
        for directory in PLANS_ROOT.iterdir():
            if not directory.is_dir():
                continue
            if directory.name == "single":
                for season_dir in directory.iterdir():
                    if season_dir.is_dir() and season_pat.match(season_dir.name):
                        candidates.add(season_dir.name)
            elif season_pat.match(directory.name):
                candidates.add(directory.name)
    if PRED_ROOT.exists():
        for subdir in PRED_ROOT.iterdir():
            if not subdir.is_dir():
                continue
            for season_dir in subdir.iterdir():
                if season_dir.is_dir() and season_pat.match(season_dir.name):
                    candidates.add(season_dir.name)
    if not candidates:
        return None
    return sorted(candidates)[-1]


def read_table(path_csv_or_parquet: Path) -> pd.DataFrame:
    if not path_csv_or_parquet.exists():
        return pd.DataFrame()
    if path_csv_or_parquet.suffix.lower() == ".parquet":
        return pd.read_parquet(path_csv_or_parquet)
    return pd.read_csv(path_csv_or_parquet)


def find_latest_pred_path(season: str, subdir: str, stem: str) -> Optional[Path]:
    candidates: list[Path] = []
    season_dir = PRED_ROOT / subdir / season
    if season_dir.exists():
        for ext in (".parquet", ".csv"):
            candidates.append(season_dir / f"{stem}{ext}")
        gw_files = sorted(season_dir.glob("GW*.parquet")) + sorted(season_dir.glob("GW*.csv"))
        if gw_files:
            candidates.insert(0, gw_files[-1])
    for ext in (".parquet", ".csv"):
        candidates.append(PRED_ROOT / subdir / f"{stem}{ext}")
    return first_existing(candidates)


def find_latest_plan(season: str, mode: str) -> Optional[Path]:
    if mode == "Single-GW":
        base = PLANS_ROOT / "single" / season
        if not base.exists():
            return None
        gws = sorted(
            [path for path in base.iterdir() if path.is_dir() and path.name.isdigit()],
            key=lambda path: int(path.name),
        )
        for gw_dir in reversed(gws):
            plan_path = gw_dir / "plan.json"
            if plan_path.exists():
                return plan_path
            chips_dir = gw_dir / "chips"
            if chips_dir.exists():
                for chip_dir in sorted(chips_dir.iterdir(), reverse=True):
                    plan_path = chip_dir / "plan.json"
                    if plan_path.exists():
                        return plan_path
        return None

    base = PLANS_ROOT / season / "multi" / "hold"
    if not base.exists():
        return None
    windows = sorted([path for path in base.iterdir() if path.is_dir() and "gw" in path.name], key=lambda path: path.name)
    for window_dir in reversed(windows):
        chips_root = window_dir / "chips"
        if chips_root.exists():
            for chip_dir in sorted(chips_root.iterdir(), reverse=True):
                for k_dir in sorted(chip_dir.iterdir(), reverse=True):
                    plan_path = k_dir / "plan.json"
                    if plan_path.exists():
                        return plan_path
        base_root = window_dir / "base"
        if base_root.exists():
            for k_dir in sorted(base_root.iterdir(), reverse=True):
                plan_path = k_dir / "plan.json"
                if plan_path.exists():
                    return plan_path
    return None


def pretty_money(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def run_cli_stream(argv: List[str], live_area, spinner_label: str = "Running...") -> int:
    live_area.expander("Live logs (click to expand)", expanded=True).write(" ".join(argv))
    output = live_area.expander("Output", expanded=True)
    lines: list[str] = []
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        output.error(f"[launcher-error] {exc}")
        return 1

    with st.spinner(spinner_label):
        for line in iter(proc.stdout.readline, ""):
            lines.append(line.rstrip("\n"))
            if len(lines) > 400:
                lines = lines[-400:]
            output.code("\n".join(lines))
        proc.stdout.close()
        return_code = proc.wait()
    if return_code == 0:
        output.success("Completed.")
    else:
        output.error(f"Process exited with code {return_code}")
    return return_code


def load_teams_map() -> dict[str, str]:
    teams = load_json(TEAMS_JSON, {})
    if isinstance(teams, dict):
        if "teams" in teams and isinstance(teams["teams"], list):
            result: dict[str, str] = {}
            for item in teams["teams"]:
                team_id = str(item.get("team_id") or item.get("id") or item.get("code") or item.get("hex_id") or "")
                short = item.get("short") or item.get("short_name") or item.get("name") or ""
                if team_id:
                    result[team_id] = short
            return result
        result = {}
        for key, value in teams.items():
            if isinstance(value, dict):
                short = value.get("short") or value.get("short_name") or value.get("name") or ""
                result[str(key)] = short
        return result
    return {}


def normalize_pos(value: str | None) -> Optional[str]:
    if not value:
        return None
    norm = str(value).upper()
    aliases = {
        "GKP": "GK",
        "GOALKEEPER": "GK",
        "DEFENDER": "DEF",
        "MIDFIELDER": "MID",
        "FORWARD": "FWD",
        "STRIKER": "FWD",
    }
    return aliases.get(norm, norm) if norm in {"GK", "DEF", "MID", "FWD"} else aliases.get(norm)


def validate_squad_rules(squad: list, teams_map: dict[str, str]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(squad, list) or len(squad) != 15:
        return False, ["Squad must be a list of exactly 15 entries."]

    pos_counts = Counter()
    team_counts = Counter()
    missing_pos: list[int] = []
    missing_team: list[int] = []

    for index, item in enumerate(squad, start=1):
        if isinstance(item, dict):
            pos = normalize_pos(item.get("pos"))
            team_id = item.get("team_id")
        else:
            pos = None
            team_id = None
        if not pos:
            missing_pos.append(index)
        else:
            pos_counts[pos] += 1
        if team_id is None:
            missing_team.append(index)
        else:
            team_counts[str(team_id)] += 1

    for pos, required in {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}.items():
        if pos_counts.get(pos, 0) != required:
            errors.append(f"Position count invalid: {pos}={pos_counts.get(pos, 0)} (must be {required}).")
    for team_id, count in team_counts.items():
        if count > 3:
            short = teams_map.get(str(team_id), str(team_id))
            errors.append(f"Team cap exceeded: {short} has {count} players (max 3).")
    if missing_pos:
        errors.append(f"Missing 'pos' for entries at indexes: {missing_pos}. Add pos for all players.")
    if missing_team:
        errors.append(f"Missing 'team_id' for entries at indexes: {missing_team}. Add team_id for all players.")
    return len(errors) == 0, errors


def apply_plan_to_state(plan: dict, state: dict) -> dict:
    updated = json.loads(json.dumps(state))
    meta = plan.get("meta", {})
    if "season" in meta:
        updated["season"] = meta["season"]
    if "gw" in meta:
        updated["gw"] = meta["gw"]
    if "bank_after" in meta:
        updated["bank"] = float(meta["bank_after"])
    elif "budget" in meta and isinstance(meta["budget"], dict):
        try:
            bank_before = float(meta.get("bank_before", updated.get("bank", 0.0)))
            net_spend = float(meta["budget"].get("net_spend", 0.0))
            updated["bank"] = bank_before - net_spend
        except Exception:
            pass
    if "free_transfers_after" in meta:
        updated["free_transfers"] = int(meta["free_transfers_after"])
    elif "transfers_used" in meta:
        try:
            before = int(meta.get("free_transfers_before", updated.get("free_transfers", 1)))
            used = int(meta.get("transfers_used", 0))
            updated["free_transfers"] = max(0, before - used)
        except Exception:
            pass
    if "chip_used" in meta and meta["chip_used"]:
        chip = str(meta["chip_used"]).upper()
        chips = updated.get("chips", {"WC": True, "FH": True, "TC": True, "BB": True})
        if chip in chips and chips[chip] is True:
            chips[chip] = False
        updated["chips"] = chips
    history = updated.get("history", [])
    history.append(
        {
            "gw": meta.get("gw"),
            "season": meta.get("season"),
            "objective": plan.get("objective", {}),
            "transfers": plan.get("transfers", []),
            "chip_used": meta.get("chip_used"),
            "bank_before": meta.get("bank_before"),
            "bank_after": updated.get("bank"),
            "free_transfers_before": meta.get("free_transfers_before"),
            "free_transfers_after": updated.get("free_transfers"),
        }
    )
    updated["history"] = history
    return updated


def default_team_state(season: str) -> dict:
    return {
        "season": season,
        "gw": 1,
        "bank": 0.0,
        "free_transfers": 1,
        "chips": {"WC": True, "FH": True, "TC": True, "BB": True},
        "squad": [],
        "captain": None,
        "vice_captain": None,
        "history": [],
    }

