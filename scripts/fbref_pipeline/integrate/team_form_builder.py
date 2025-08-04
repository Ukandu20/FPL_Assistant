#!/usr/bin/env python3
r"""team_form_builder.py  ▸ schema v1.3   (def + att support)

Builds rolling-form features for every team-match and writes them to

    data/processed/features/<version>/<SEASON>/<file_name>

where  <file_name> is

    • team_def_form.csv   ( --metric def )
    • team_att_form.csv   ( --metric att )

Key points
──────────
* One script, parameterised by `--metric`:
      def  → GA / xGA   (flip sign so higher = better defence)
      att  → GF / xG    (natural polarity: higher = better attack)
* Rolling 5-match mean with early-season decay (weights 0.8,0.6,0.4,0.2).
* Venue splits with empirical shrinkage λ = n/(n+τ).
* Per-GW percentile and z-score (sign-flipped if requested).
* Promoted-club prior = mean of relegated clubs’ GA/xGA (or GF/xG) from
  previous season.
* Break flag when *every* club has ≥13-day gap since its last match.

Run examples
────────────
```bash
# Defensive form (GA/xGA)
python -m scripts.fbref_pipeline.integrate.team_form_builder \
       --metric def --fixtures-root data/processed/fixtures

# Attacking form (GF/xG)
python -m scripts.fbref_pipeline.integrate.team_form_builder \
       --metric att --fixtures-root data/processed/fixtures
"""
from __future__ import annotations
import argparse, json, logging, statistics, datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCHEMA_VERSION = "v1.3"

# ───────────────────── metric configuration ────────────────────────────────
METRIC_CFG = {
    "def": {  # defensive
        "raw":       ("ga", "xga"),          # columns in fixture_calendar
        "forms":     ("ga90_roll5", "xga90_roll5"),
        "flip_sign": True,                   # higher z = stronger defence
        "file":      "team_def_form.csv",
    },
    "att": {  # attacking
        "raw":       ("gf", "xg"),
        "forms":     ("gf90_roll5", "xg90_roll5"),
        "flip_sign": False,
        "file":      "team_att_form.csv",
    },
}

# ────────────────────────── helpers I/O ────────────────────────────────────
def load_json(p: Path) -> dict:
    return json.loads(p.read_text("utf-8")) if p.is_file() else {}

def save_json(obj: dict, p: Path) -> None:
    p.write_text(json.dumps(obj, indent=2))

# ───────────────────── rolling-window calculator ───────────────────────────
def rolling_form(
    team_df: pd.DataFrame,
    val_col: str,
    window: int,
    decay: Tuple[float],
    tau: float,
    promoted_prior: float | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return overall, home, away rolling means (per-90) for one value column."""
    v = team_df[val_col].to_numpy()
    venues = team_df["venue"].to_numpy()  # "Home"/"Away"
    m = len(v)

    roll, roll_h, roll_a = np.empty(m), np.empty(m), np.empty(m)

    for i in range(m):
        lo = max(0, i - window + 1)
        recent = v[lo : i + 1]
        mean = recent.mean()

        # early-season decay toward prior
        if i < len(decay):
            w = decay[i]
            prior = promoted_prior if promoted_prior is not None else mean
            mean = (1 - w) * mean + w * prior

        roll[i] = mean

        # venue splits with shrinkage
        recent_h = recent[venues[lo : i + 1] == "Home"]
        recent_a = recent[venues[lo : i + 1] == "Away"]
        nh, na = len(recent_h), len(recent_a)
        lam_h, lam_a = nh / (nh + tau), na / (na + tau)

        roll_h[i] = lam_h * (recent_h.mean() if nh else mean) + (1 - lam_h) * mean
        roll_a[i] = lam_a * (recent_a.mean() if na else mean) + (1 - lam_a) * mean

    return roll, roll_h, roll_a

# ───────────────────── break-week flag ─────────────────────────────────────
def add_break_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["team_id", "date_played"]).copy()
    df["gap"] = df.groupby("team_id")["date_played"].diff().dt.days
    league_gap = df.groupby("gw_orig")["date_played"].min().diff().dt.days
    df["league_gap"] = df["gw_orig"].map(league_gap)
    df["is_break"] = np.where((df["gap"] >= 8) & (df["league_gap"] >= 13), 1, 0)
    return df.drop(columns=["gap", "league_gap"])

# ───────────────────── per-season builder ──────────────────────────────────
def build_team_form(
    season: str,
    fixtures_root: Path,
    out_version: str,
    out_dir: Path,
    decay: Tuple[float],
    tau: float,
    metric: str,
    force: bool,
) -> bool:
    cfg = METRIC_CFG[metric]
    cal_fp = fixtures_root / season / "fixture_calendar.csv"
    dst_dir = out_dir / out_version / season
    dst_csv = dst_dir / cfg["file"]
    meta_fp = dst_dir / "team_form.meta.json"

    if dst_csv.exists() and not force:
        logging.info("%s • %s exists – skip (use --force)", season, cfg["file"])
        return False
    if not cal_fp.is_file():
        logging.warning("%s • fixture_calendar missing – skipped", season)
        return False

    cal = pd.read_csv(cal_fp, parse_dates=["date_played"])

    # --- column sanity -----------------------------------------------------
    raw_g, raw_xg = cfg["raw"]
    required = {
        "team_id",
        "team",
        "venue",
        "gw_orig",
        "date_played",
        raw_g,
        raw_xg,
    }
    if not required.issubset(cal.columns):
        raise KeyError(f"{season}: fixture_calendar missing {required - set(cal.columns)}")

    # --- promoted-club priors ---------------------------------------------
    season_start = int(season.split("-")[0])
    prev_fp = fixtures_root / f"{season_start-1}-{season_start}" / "fixture_calendar.csv"
    promoted_lookup: Dict[int, Tuple[float, float]] = {}
    if prev_fp.is_file():
        prev_cal = pd.read_csv(prev_fp, usecols=["team_id", raw_g, raw_xg])
        relegated = set(prev_cal.team_id.unique()) - set(cal.team_id.unique())
        promoted = set(cal.team_id.unique()) - set(prev_cal.team_id.unique())
        if relegated:
            rel_mean_g = prev_cal.loc[prev_cal.team_id.isin(relegated), raw_g].mean()
            rel_mean_x = prev_cal.loc[prev_cal.team_id.isin(relegated), raw_xg].mean()
            promoted_lookup = {tid: (rel_mean_g, rel_mean_x) for tid in promoted}

    # --- rolling calculations ---------------------------------------------
    form_rows: List[pd.DataFrame] = []
    form_g, form_x = cfg["forms"]
    for tid, g in cal.groupby("team_id"):
        pri_g, pri_x = promoted_lookup.get(tid, (None, None))
        roll_g, roll_g_h, roll_g_a = rolling_form(g, raw_g, 5, decay, tau, pri_g)
        roll_x, _, _ = rolling_form(g, raw_xg, 5, decay, tau, pri_x)

        tmp = g.copy()
        tmp[form_g] = roll_g
        tmp[form_x] = roll_x
        tmp[f"{form_g[:-5]}home_5"] = roll_g_h
        tmp[f"{form_g[:-5]}away_5"] = roll_g_a
        form_rows.append(tmp)

    df = pd.concat(form_rows, ignore_index=True)
    df = add_break_flag(df)

    # --- league percentiles & z-scores ------------------------------------
    base_cols = [
        form_g,
        form_x,
        f"{form_g[:-5]}home_5",
        f"{form_g[:-5]}away_5",
    ]
    for col in base_cols:
        df[col + "_pct"] = df.groupby("gw_orig")[col].rank(pct=True)
        z = (df[col] - df.groupby("gw_orig")[col].transform("mean")) / df.groupby(
            "gw_orig"
        )[col].transform("std")
        df[col + "_z"] = -z if cfg["flip_sign"] else z

    # --- write -------------------------------------------------------------
    dst_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_csv, index=False)

    meta = {
        "schema": SCHEMA_VERSION,
        "metric": metric,
        "build_ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "decay": decay,
        "tau": tau,
    }
    save_json(meta, meta_fp)
    logging.info(
        "%s • %s (%d rows) written", season, cfg["file"], len(df)
    )
    return True

# ───────────────────────── batch driver ────────────────────────────────────
def run_batch(
    seasons: List[str],
    fixtures_root: Path,
    out_version: str,
    out_dir: Path,
    decay: Tuple[float],
    tau: float,
    metric: str,
    force: bool,
):
    for season in seasons:
        try:
            build_team_form(
                season,
                fixtures_root,
                out_version,
                out_dir,
                decay,
                tau,
                metric,
                force,
            )
        except Exception:
            logging.exception("%s • unhandled error – skipped", season)

# ──────────────────────────── CLI entry ────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", help="e.g. 2024-2025; omit for batch")
    ap.add_argument("--fixtures-root", type=Path, default=Path("data/processed/fixtures"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--version", default="v1")
    ap.add_argument(
        "--metric",
        choices=["def", "att"],
        default="def",
        help="def = GA/xGA,  att = GF/xG",
    )
    ap.add_argument("--decay", default="0.8,0.6,0.4,0.2")
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=args.log_level.upper())
    decay = tuple(float(x) for x in args.decay.split(","))

    seasons = [args.season] if args.season else sorted(
        d.name for d in args.fixtures_root.iterdir() if d.is_dir()
    )
    if not seasons:
        logging.error("No season folders in %s", args.fixtures_root)
        return

    logging.info("Processing seasons: %s", ", ".join(seasons))
    run_batch(
        seasons,
        args.fixtures_root,
        args.version,
        args.out_dir,
        decay,
        args.tau,
        args.metric,
        args.force,
    )

if __name__ == "__main__":
    main()
