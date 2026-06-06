#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_builder.py — one-shot trainer for Minutes, Goals/Assists, Defense, Saves

Key features
------------
• One CLI to run any subset: --do minutes,ga,defense,saves or --do all
• Global flags (--seasons, --first-test-gw, --version-tag, --bump-version) that cascade
• Per-task overrides (e.g., --ga.seasons, --def.version_tag, --saves.bump_version) take precedence
• Optional --profile (JSON/TOML) to load presets
• Strict error handling (fail fast unless --continue-on-error)
• Dry-run to print exact commands without executing

Precedence (strongest → weakest)
--------------------------------
per-task CLI  >  global CLI  >  profile  >  task defaults
"""

from __future__ import annotations
import argparse, shlex, subprocess, sys, json, os
from pathlib import Path
from typing import List, Dict, Tuple, Any

# ----------------------------- helpers ----------------------------------------

def _run(cmd: List[str], dry_run: bool, env: Dict[str, str], title: str) -> Tuple[int, str]:
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n[run] {title} → {printable}")
    if dry_run:
        return 0, "[dry-run]"
    proc = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    return proc.returncode, printable

def _merge(overrides: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defaults)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out

def _load_profile(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--profile not found: {path}")
    txt = p.read_text(encoding="utf-8").strip()
    # Try JSON first, then TOML
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        try:
            import tomllib  # Py3.11+
        except Exception:
            import tomli as tomllib  # fallback if available
        return tomllib.loads(txt)

# ----------------------------- defaults ---------------------------------------

def minutes_defaults():
    return dict(
        seasons="2021-2022,2022-2023,2023-2024,2024-2025",
        first_test_gw="26",
        fix_root="data/processed/registry/fixtures",
        model_out="data/models/minutes",
        use_fdr=True,
        form_root="data/processed/registry/features",
        form_version="v2",
        form_source="team",
        add_team_rotation=True,
        use_taper=True,
        taper_lo="0.40", taper_hi="0.70", taper_min_scale="0.80",
        use_pos_bench_caps=True,
        use_calibration=True, use_p60_calibration=True,
        p60_mode="direct",
        t_lo="0.25", t_hi="0.65",
        gate_blend="0.25",
        disable_pstart_caps=True,
        version_tag="v1",
        bump_version=False,            # added for global control
    )

def ga_defaults():
    return dict(
        seasons="2020-2021,2021-2022,2022-2023,2023-2024,2024-2025",
        first_test_gw="30",
        features_root="data/processed/registry/features",
        form_version="v1",
        minutes_preds="data/models/minutes/versions/v1/expected_minutes.csv",
        require_pred_minutes=True,
        use_z=True, poisson_heads=True, calibrate_poisson=True,
        ewm_halflife="3",
        ewm_halflife_pos="GK:4,DEF:4,MID:3,FWD:2",
        model_out="data/models/goals_assists",
        bump_version=True,             # GA default behavior you used
        version_tag="v2",
        skip_gk=True, dump_lambdas=True,
    )

def defense_defaults():
    return dict(
        seasons="2020-2021,2021-2022,2022-2023,2023-2024,2024-2025",
        first_test_gw="26",
        features_root="data/processed/registry/features",
        form_version="v2",
        use_z=True, na_thresh="0.70",
        minutes_preds="data/models/minutes/versions/v1/expected_minutes.csv",
        fallback_p60_from_minutes=True,
        calibrate_team_cs=True, reliability_bins="10", monotone_cs=True,
        min_dcp_minutes="30",
        dcp_k90="DEF:10;MID:12;FWD:12",
        skip_gk=True,
        model_out="data/models/defense",
        version_tag="v1",
        log_level="INFO",
        bump_version=False,            # added for global control
    )

def saves_defaults():
    return dict(
        seasons="2020-2021,2021-2022,2022-2023,2023-2024,2024-2025",
        first_test_gw="26",
        features_root="data/processed/registry/features",
        form_version="v2",
        minutes_preds="data/models/minutes/versions/v1/expected_minutes.csv",
        require_pred_minutes=True,
        poisson_head=True,
        model_out="data/models/saves",
        bump_version=True,             # your current behavior
        version_tag="v1",
    )

# ----------------------------- CLI --------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--do", default="all",
                    help="Comma list: minutes,ga,defense,saves,all")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default=sys.executable, help="Python binary to use")

    # Global flags applied to all tasks unless overridden per-task
    ap.add_argument("--seasons", default=None,
                    help="Global seasons for all tasks (comma list). Overridden by --minutes.seasons, --ga.seasons, etc.")
    ap.add_argument("--first-test-gw", dest="first_test_gw", default=None,
                    help="Global first test GW for all tasks. Overridden by task-specific flags.")
    ap.add_argument("--version-tag", dest="version_tag", default=None,
                    help="Global version tag for all tasks. Overridden by task-specific flags like --ga.version_tag.")

    # Tri-state global bump-version: True / False / None (unset)
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument("--bump-version", dest="bump_version", action="store_true",
                    help="Globally enable bump-version for all tasks (can be overridden per task).")
    mx.add_argument("--no-bump-version", dest="bump_version", action="store_false",
                    help="Globally disable bump-version for all tasks (can be overridden per task).")
    ap.set_defaults(bump_version=None)

    ap.add_argument("--profile", default=None,
                    help="Optional TOML/JSON profile with keys: common, minutes, ga, def, saves")

    # Per-task override groups (all default to None so overrides are explicit)
    grp_m = ap.add_argument_group("minutes")
    for k in minutes_defaults().keys():
        grp_m.add_argument(f"--minutes.{k}", default=None)

    grp_g = ap.add_argument_group("ga")
    for k in ga_defaults().keys():
        grp_g.add_argument(f"--ga.{k}", default=None)

    grp_d = ap.add_argument_group("def")
    for k in defense_defaults().keys():
        grp_d.add_argument(f"--def.{k}", default=None)

    grp_s = ap.add_argument_group("saves")
    for k in saves_defaults().keys():
        grp_s.add_argument(f"--saves.{k}", default=None)

    return ap

# ----------------------------- command builders --------------------------------

def _cmd_minutes(cfg: Dict[str, Any], pybin: str) -> List[str]:
    cmd = [
        pybin, "-m", "scripts.models.minutes_model_builder",
        "--seasons", str(cfg["seasons"]),
        "--first-test-gw", str(cfg["first_test_gw"]),
        "--fix-root", str(cfg["fix_root"]),
        "--model-out", str(cfg["model_out"]),
        "--use-fdr",
        "--form-root", str(cfg["form_root"]),
        "--form-version", str(cfg["form_version"]),
        "--form-source", str(cfg["form_source"]),
    ]
    if cfg["add_team_rotation"]: cmd += ["--add-team-rotation"]
    if cfg["use_taper"]:
        cmd += ["--use-taper", "--taper-lo", str(cfg["taper_lo"]),
                "--taper-hi", str(cfg["taper_hi"]), "--taper-min-scale", str(cfg["taper_min_scale"])]
    if cfg["use_pos_bench_caps"]: cmd += ["--use-pos-bench-caps"]
    if cfg["use_calibration"]: cmd += ["--use-calibration"]
    if cfg["use_p60_calibration"]: cmd += ["--use-p60-calibration"]
    cmd += ["--p60-mode", str(cfg["p60_mode"]),
            "--t-lo", str(cfg["t_lo"]), "--t-hi", str(cfg["t_hi"]),
            "--gate-blend", str(cfg["gate_blend"]),
            "--version-tag", str(cfg["version_tag"])]
    if cfg["disable_pstart_caps"]: cmd += ["--disable-pstart-caps"]
    if cfg["bump_version"]: cmd += ["--bump-version"]
    return cmd

def _cmd_ga(cfg: Dict[str, Any], pybin: str) -> List[str]:
    cmd = [
        pybin, "-m", "scripts.models.goals_assists_model_builder",
        "--seasons", str(cfg["seasons"]),
        "--first-test-gw", str(cfg["first_test_gw"]),
        "--features-root", str(cfg["features_root"]),
        "--form-version", str(cfg["form_version"]),
        "--minutes-preds", str(cfg["minutes_preds"]),
    ]
    if cfg["require_pred_minutes"]: cmd += ["--require-pred-minutes"]
    if cfg["use_z"]: cmd += ["--use-z"]
    if cfg["poisson_heads"]: cmd += ["--poisson-heads"]
    if cfg["calibrate_poisson"]: cmd += ["--calibrate-poisson"]
    cmd += ["--ewm-halflife", str(cfg["ewm_halflife"]),
            "--ewm-halflife-pos", str(cfg["ewm_halflife_pos"]),
            "--model-out", str(cfg["model_out"])]
    if cfg["bump_version"]: cmd += ["--bump-version"]
    if cfg["version_tag"]: cmd += ["--version-tag", str(cfg["version_tag"])]
    if cfg["skip_gk"]: cmd += ["--skip-gk"]
    if cfg["dump_lambdas"]: cmd += ["--dump-lambdas"]
    return cmd

def _cmd_defense(cfg: Dict[str, Any], pybin: str) -> List[str]:
    cmd = [
        pybin, "-m", "scripts.models.defense_model_builder",
        "--seasons", str(cfg["seasons"]),
        "--first-test-gw", str(cfg["first_test_gw"]),
        "--features-root", str(cfg["features_root"]),
        "--form-version", str(cfg["form_version"]),
    ]
    if cfg["use_z"]: cmd += ["--use-z"]
    cmd += ["--na-thresh", str(cfg["na_thresh"]),
            "--minutes-preds", str(cfg["minutes_preds"])]
    if cfg["fallback_p60_from_minutes"]: cmd += ["--fallback-p60-from-minutes"]
    if cfg["calibrate_team_cs"]: cmd += ["--calibrate-team-cs"]
    cmd += ["--reliability-bins", str(cfg["reliability_bins"])]
    if cfg["monotone_cs"]: cmd += ["--monotone-cs"]
    cmd += ["--min-dcp-minutes", str(cfg["min_dcp_minutes"]),
            "--dcp-k90", str(cfg["dcp_k90"])]
    if cfg["skip_gk"]: cmd += ["--skip-gk"]
    cmd += ["--model-out", str(cfg["model_out"]),
            "--version-tag", str(cfg["version_tag"]),
            "--log-level", str(cfg["log_level"])]
    if cfg["bump_version"]: cmd += ["--bump-version"]
    return cmd

def _cmd_saves(cfg: Dict[str, Any], pybin: str) -> List[str]:
    cmd = [
        pybin, "-m", "scripts.models.saves_model_builder",
        "--seasons", str(cfg["seasons"]),
        "--first-test-gw", str(cfg["first_test_gw"]),
        "--features-root", str(cfg["features_root"]),
        "--form-version", str(cfg["form_version"]),
        "--minutes-preds", str(cfg["minutes_preds"]),
    ]
    if cfg["require_pred_minutes"]: cmd += ["--require-pred-minutes"]
    if cfg["poisson_head"]: cmd += ["--poisson-head"]
    cmd += ["--model-out", str(cfg["model_out"])]
    if cfg["bump_version"]: cmd += ["--bump-version"]
    if cfg["version_tag"]: cmd += ["--version-tag", str(cfg["version_tag"])]
    return cmd

# ----------------------------- main -------------------------------------------

def main():
    ap = build_parser()
    args = ap.parse_args()
    pybin = args.python

    # Load profile (optional)
    prof = _load_profile(args.profile)
    prof_common = prof.get("common", {})
    prof_minutes = prof.get("minutes", {})
    prof_ga = prof.get("ga", {})
    prof_def = prof.get("def", {})
    prof_saves = prof.get("saves", {})

    # Which tasks?
    want = [w.strip().lower() for w in args.do.split(",") if w.strip()]
    if "all" in want:
        want = ["minutes", "ga", "defense", "saves"]

    # Common scope (CLI wins over profile)
    common = {
        "seasons": args.seasons or prof_common.get("seasons"),
        "first_test_gw": args.first_test_gw or prof_common.get("first_test_gw"),
        "version_tag": args.version_tag or prof_common.get("version_tag"),
        "bump_version": args.bump_version if args.bump_version is not None else prof_common.get("bump_version", None),
    }

    def _apply_common(defaults: dict, prof_task: dict, cli_task: dict, common: dict) -> dict:
        """Precedence: per-task CLI > global CLI > profile > defaults."""
        cfg = dict(defaults)

        # 1) Global CLI/profile overrides land early (overwrite defaults).
        for f in ("seasons", "first_test_gw", "version_tag"):
            if common.get(f) not in (None, ""):
                cfg[f] = common[f]
        # bump_version is tri-state: only override if not None
        if common.get("bump_version") is not None:
            cfg["bump_version"] = bool(common["bump_version"])

        # 2) Profile task-level overrides
        for k, v in prof_task.items():
            if v is not None:
                cfg[k] = v

        # 3) Per-task CLI overrides (strongest)
        for k, v in cli_task.items():
            if v is not None:
                cfg[k] = v

        return cfg

    # Build task configs with correct precedence
    cli_minutes = {k.split(".",1)[1]: getattr(args, k) for k in vars(args) if k.startswith("minutes.")}
    cli_ga      = {k.split(".",1)[1]: getattr(args, k) for k in vars(args) if k.startswith("ga.")}
    cli_def     = {k.split(".",1)[1]: getattr(args, k) for k in vars(args) if k.startswith("def.")}
    cli_saves   = {k.split(".",1)[1]: getattr(args, k) for k in vars(args) if k.startswith("saves.")}

    cfg_m = _apply_common(minutes_defaults(), prof_minutes, cli_minutes, common)
    cfg_g = _apply_common(ga_defaults(),       prof_ga,      cli_ga,      common)
    cfg_d = _apply_common(defense_defaults(),  prof_def,     cli_def,     common)
    cfg_s = _apply_common(saves_defaults(),    prof_saves,   cli_saves,   common)

    # Plan/runners
    runners = {
        "minutes": ("Minutes", lambda: _cmd_minutes(cfg_m, pybin)),
        "ga":      ("Goals/Assists", lambda: _cmd_ga(cfg_g, pybin)),
        "defense": ("Defense", lambda: _cmd_defense(cfg_d, pybin)),
        "saves":   ("Saves", lambda: _cmd_saves(cfg_s, pybin)),
    }

    env = {"PYTHONWARNINGS": "ignore", "LOG_LEVEL": args.log_level}
    results = []
    for key in want:
        if key not in runners:
            raise SystemExit(f"Unknown task in --do: {key}")
        title, fn = runners[key]
        cmd = fn()
        rc, printed = _run(cmd, dry_run=args.dry_run, env=env, title=title)
        results.append((title, rc, printed))
        if rc != 0 and not args.continue_on_error:
            print(f"\n[error] {title} failed with exit code {rc}. Aborting (use --continue-on-error to proceed).")
            break

    # Summary
    summary = [{"task": t, "exit_code": rc, "cmd": c} for (t, rc, c) in results]
    print("\n=== model_builder summary ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
