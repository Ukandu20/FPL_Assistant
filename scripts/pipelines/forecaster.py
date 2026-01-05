#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot forecaster pipeline (Windows-safe argv lists)
Runs: minutes → goals_assists → defense → saves → points (hard-fail on any error)
"""
from __future__ import annotations
import argparse, subprocess, sys, logging, time
from pathlib import Path
from typing import List

# ───────────────────────────── logging ─────────────────────────────
def setup_logging(level: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)]
    )
    logging.info("Pipeline log → %s", log_path)

def run_stage(stage: str, argv: List[str]) -> None:
    logging.info("▶ [%s] RUN: %s", stage, " ".join(argv))
    t0 = time.time()
    proc = subprocess.run([a for a in argv if a != ""], capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        logging.error("✖ [%s] FAILED in %.1fs", stage, dt)
        if proc.stdout:
            logging.error("─── STDOUT (%s) ───\n%s\n──────────────", stage, proc.stdout.strip())
        if proc.stderr:
            logging.error("─── STDERR (%s) ───\n%s\n──────────────", stage, proc.stderr.strip())
        raise SystemExit(f"[{stage}] failed (exit={proc.returncode}). See log above.")
    logging.info("✔ [%s] DONE in %.1fs", stage, dt)

def require_non_empty_dir(stage: str, path: Path) -> None:
    if not path.exists() or not any(path.iterdir()):
        raise SystemExit(f"[{stage}] produced no artifacts: {path} is empty or missing")

# ───────────────────────────── args ─────────────────────────────
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="One-shot forecaster (minutes→GA→DEF→SAV→points)")
    # windowing
    p.add_argument("--history-seasons", required=True)
    p.add_argument("--future-season", required=True)
    p.add_argument("--as-of", required=True)          # "now" or timestamp
    p.add_argument("--as-of-tz", required=True)
    p.add_argument("--as-of-gw", type=int, required=True)
    p.add_argument("--n-future", type=int, required=True)

    # commons
    p.add_argument("--features-root", required=True)
    p.add_argument("--form-version", required=True)
    p.add_argument("--fix-root", required=True)
    p.add_argument("--team-fixtures-filename", default="fixture_calendar.csv")

    # models
    p.add_argument("--minutes-model-dir", required=True)
    p.add_argument("--ga-model-dir", required=True)
    p.add_argument("--defense-model-dir", required=True)
    p.add_argument("--saves-model-dir", required=True)

    # registry / filter
    p.add_argument("--teams-json", required=True)
    p.add_argument("--league-filter", required=True)

    # toggles (defaults mirror your current runs)
    p.add_argument("--use-fdr", action="store_true", default=True)
    p.add_argument("--use-taper", action="store_true", default=True)
    p.add_argument("--taper-lo", type=float, default=0.40)
    p.add_argument("--taper-hi", type=float, default=0.70)
    p.add_argument("--taper-min-scale", type=float, default=0.80)
    p.add_argument("--use-pos-bench-caps", action="store_true", default=True)
    p.add_argument("--no-mix-gk", action="store_true", default=True)
    p.add_argument("--apply-calibration", action="store_true", default=True)
    p.add_argument("--skip-gk", action="store_true", default=True)
    p.add_argument("--require-pred-minutes", action="store_true", default=True)

    # outputs
    p.add_argument("--pred-root", default="data/predictions")
    p.add_argument("--out-format", default="both", choices=["csv","parquet","both"])

    # logging
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--log-file", default="logs/pipelines/one_shot_forecaster.log")
    return p

# ───────────────────────────── main ─────────────────────────────
def main(argv=None) -> None:
    args = parser().parse_args(argv)
    setup_logging(args.log_level, Path(args.log_file))

    py = sys.executable  # use the current Python
    pred_root     = Path(args.pred_root)
    minutes_root  = pred_root / "minutes"
    ga_root       = pred_root / "goals_assists"
    defense_root  = pred_root / "defense"
    saves_root    = pred_root / "saves"
    points_outdir = pred_root / "expected_points"

    for d in (minutes_root, ga_root, defense_root, saves_root, points_outdir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) MINUTES
    minutes_argv = [
        py, "-m", "scripts.models.minutes_forecast",
        "--history-seasons", args.history_seasons,
        "--future-season", args.future_season,
        "--as-of", args.as_of, "--as-of-tz", args.as_of_tz, "--as-of-gw", str(args.as_of_gw),
        "--n-future", str(args.n_future),
        "--fix-root", args.fix_root,
        "--team-fixtures-filename", args.team_fixtures_filename,
        "--model-dir", args.minutes_model_dir,
        "--form-root", args.features_root, "--form-version", args.form_version, "--form-source", "team",
        "--taper-lo", str(args.taper_lo), "--taper-hi", str(args.taper_hi), "--taper-min-scale", str(args.taper_min_scale),
        "--out-dir", str(minutes_root),
        "--out-format", args.out_format,
    ]
    if args.use-fdr if hasattr(args, "use-fdr") else args.use_fdr: minutes_argv.append("--use-fdr")
    if args.use_taper: minutes_argv.append("--use-taper")
    if args.use_pos_bench_caps: minutes_argv.append("--use-pos-bench-caps")
    if args.no_mix_gk: minutes_argv.append("--no-mix-gk")

    run_stage("minutes", minutes_argv)
    require_non_empty_dir("minutes", minutes_root)

    # 2) GOALS & ASSISTS
    ga_argv = [
        py, "-m", "scripts.models.goals_assists_forecast",
        "--history-seasons", args.history_seasons,
        "--future-season", args.future_season,
        "--as-of", args.as_of, "--as-of-tz", args.as_of_tz, "--as-of-gw", str(args.as_of_gw),
        "--n-future", str(args.n_future),
        "--features-root", args.features_root, "--form-version", args.form_version,
        "--fix-root", args.fix_root, "--team-fixtures-filename", args.team_fixtures_filename,
        "--model-dir", args.ga_model_dir,
        "--out-dir", str(ga_root),
        "--out-format", args.out_format,
        "--teams-json", args.teams_json,
        "--league-filter", args.league_filter,
    ]
    if args.apply_calibration: ga_argv.append("--apply-calibration")
    if args.skip_gk: ga_argv.append("--skip-gk")

    run_stage("goals_assists", ga_argv)
    require_non_empty_dir("goals_assists", ga_root)

    # 3) DEFENSE
    defense_argv = [
        py, "-m", "scripts.models.defense_forecast",
        "--history-seasons", args.history_seasons,
        "--future-season", args.future_season,
        "--as-of", args.as_of, "--as-of-tz", args.as_of_tz, "--as-of-gw", str(args.as_of_gw),
        "--n-future", str(args.n_future),
        "--features-root", args.features_root, "--form-version", args.form_version,
        "--fix-root", args.fix_root, "--team-fixtures-filename", args.team_fixtures_filename,
        "--model-dir", args.defense_model_dir,
        "--out-dir", str(defense_root),
        "--out-format", args.out_format,
        "--teams-json", args.teams_json,
        "--league-filter", args.league_filter,
    ]
    run_stage("defense", defense_argv)
    require_non_empty_dir("defense", defense_root)

    # 4) SAVES
    saves_argv = [
        py, "-m", "scripts.models.saves_forecast",
        "--history-seasons", args.history_seasons,
        "--future-season", args.future_season,
        "--as-of", args.as_of, "--as-of-tz", args.as_of_tz, "--as-of-gw", str(args.as_of_gw),
        "--n-future", str(args.n_future),
        "--features-root", args.features_root, "--form-version", args.form_version,
        "--minutes-root", str(minutes_root),
        "--model-dir", args.saves_model_dir,
        "--teams-json", args.teams_json,
        "--league-filter", args.league_filter,
        "--out-dir", str(saves_root),
        "--out-format", args.out_format,
    ]
    if args.require_pred_minutes: saves_argv.append("--require-pred-minutes")

    run_stage("saves", saves_argv)
    require_non_empty_dir("saves", saves_root)

    # 5) POINTS (your exact CLI)
    points_argv = [
        py, "-m", "scripts.models.points_forecast",
        "--out-dir", str(points_outdir),
        "--future-season", args.future_season,
        "--as-of-gw", str(args.as_of_gw), "--n-future", str(args.n_future),
        "--minutes-root", str(minutes_root),
        "--ga-root",      str(ga_root),
        "--defense-root", str(defense_root),
        "--saves-root",   str(saves_root),
        "--teams-json", args.teams_json,
        "--league-filter", args.league_filter,
        "--require-on-roster",
    ]
    run_stage("points", points_argv)
    require_non_empty_dir("points", points_outdir)

    logging.info("🎯 Pipeline complete. Artifacts under %s", pred_root)

if __name__ == "__main__":
    main()
