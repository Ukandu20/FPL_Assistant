#!/usr/bin/env python3
"""
Assurance suite for provider-neutral FPL integration.

Validates, per season:
  • fixture_calendar.csv  (schema, types, venue logic, 2x rows per match, sched_missing)
  • player_fixture_calendar.csv (schema, identity, uniqueness, optional price/xP coverage)
  • Cross-file join: every pmc row maps to exactly one fixture_calendar row
  • Registry membership (team_id / player_id in your _id_lookup_*.json)
  • Random player sampling checks (no missing key cols)

Soft warnings:
  • is_starter==1 & minutes==0
  • starters per (match_id,team_id) outside [10..12]
  • optional FBref lineup cross-check mismatch rate (if lineups file exists)

Empty seasons (0 rows with correct header) are treated as OK.
Exit non-zero on any failure (CI-friendly).
"""

from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- Defaults (can be overridden via CLI) ----------
REQ_FIXTURE_COLS = {
    "fpl_id", "gw_orig", "date_sched", "date_played", "team", "team_id",
    "opponent_id", "is_home", "home", "away", "home_id", "away_id",
    "status", "sched_missing",
}

# Core PMC contract (now includes starter_source & clean_sheets)
REQ_PMC_COLS = {
    "player_id","player","pos","fpl_id","gw_orig",
    "date_played","team_id","team","minutes","days_since_last","is_active",
    "venue","was_home","gf","ga","fdr_home","fdr_away",
    "is_starter","starter_source","total_points","bonus","bps","clean_sheets",
}

# Coverage thresholds (CLI can override)
PRICE_COVERAGE_MIN_DEFAULT = 0.999   # ≥ 99.9%
XP_COVERAGE_MIN_DEFAULT    = 0.98    # ≥ 98% (tune if your xP is GW-level)

# Plausible BPS range (raw BPS can be negative due to deductions)
BPS_MIN_DEFAULT = -50
BPS_MAX_DEFAULT = 200

SAMPLE_PLAYERS  = 12
SAMPLE_ROWS     = 60

# Optional FBref lineup file candidates
LINEUPS_CANDIDATES = [
    "match_lineups/lineups.csv",
    "team_match/lineups.csv",
]

# ---------- Utils ----------
def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def floor_day(dt_series: pd.Series) -> pd.Series:
    s = pd.to_datetime(dt_series, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(None)
    return s.dt.floor("D")

def fail(msg: str):
    raise AssertionError(msg)

def warn(msg: str):
    print(f"WARNING: {msg}")


def _ensure_match_identity(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """Normalize the canonical match key while retaining the legacy alias."""
    out = df.copy()
    if "match_id" not in out.columns and "fbref_id" in out.columns:
        out["match_id"] = out["fbref_id"]
    if "match_id" not in out.columns:
        fail(f"{context}: missing canonical match_id (or legacy fbref_id alias)")
    if "fbref_id" not in out.columns:
        out["fbref_id"] = out["match_id"]
    return out

# ---------- Validators ----------
def validate_fixture_calendar(season_dir: Path) -> pd.DataFrame:
    fp = season_dir / "fixture_calendar.csv"
    if not fp.exists():
        fail(f"{season_dir.name}: fixture_calendar.csv missing")

    df = pd.read_csv(fp)

    missing_header = REQ_FIXTURE_COLS - set(df.columns)
    assert not missing_header, (
        f"{season_dir.name}: fixture header missing columns: {sorted(missing_header)}"
    )
    df = _ensure_match_identity(df, f"{season_dir.name}: fixture_calendar")

    # Empty season is allowed
    if df.empty:
        return df

    # Types & normalizations
    for c in ["team_id","home_id","away_id","fpl_id","match_id","fbref_id",
              "team","home","away","venue","status","result"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    df["date_played"]   = floor_day(df["date_played"])
    df["date_sched"]    = floor_day(df["date_sched"])
    df["is_home"]       = df["is_home"].astype("Int8")
    df["was_home"]      = df["is_home"]
    df["sched_missing"] = df["sched_missing"].astype(int)

    # Venue logic: was_home == (team_id == home_id)
    pred = (df["team_id"] == df["home_id"]).astype("Int8")
    bad = (df["is_home"] != pred)
    assert not bad.any(), (
        f"{season_dir.name}: was_home mismatch vs IDs "
        f"(e.g., {df.loc[bad, ['match_id','team','team_id','home_id','venue']].head(10).to_dict('records')})"
    )

    # Cardinality: 2 team-perspective rows per canonical match.
    n_matches = df["match_id"].nunique(dropna=True)
    assert len(df) == 2 * n_matches, f"{season_dir.name}: cardinality off: {len(df)} vs 2*{n_matches}"

    expected_sched_missing = df["match_id"].isna().astype(int)
    assert (df["sched_missing"].values == expected_sched_missing.values).all(), (
        f"{season_dir.name}: sched_missing not consistent with match_id nullity"
    )

    # Uniqueness of join index
    join_index = df[["match_id","team_id"]].copy()
    dups = join_index.duplicated().sum()
    assert dups == 0, f"{season_dir.name}: duplicate join keys in fixture_calendar index"

    return df

def validate_player_minutes_calendar(season_dir: Path,
                                     price_min: float,
                                     xp_min: float,
                                     bps_min: float,
                                     bps_max: float) -> pd.DataFrame:
    candidates = [
        season_dir / "player_fixture_calendar.csv",
        season_dir / "player_minutes_calendar.csv",
    ]
    fp = next((path for path in candidates if path.is_file()), None)
    if fp is None:
        fail(f"{season_dir.name}: player_fixture_calendar.csv missing")

    df = pd.read_csv(fp)

    # Header must include all required columns (even if empty)
    missing_header = REQ_PMC_COLS - set(df.columns)
    assert not missing_header, f"{season_dir.name}: header missing PMC cols: {missing_header}"
    df = _ensure_match_identity(df, f"{season_dir.name}: player fixture calendar")

    # Empty season is allowed
    if df.empty:
        return df

    # Parse/normalize types on non-empty
    df["date_played"] = floor_day(df["date_played"])
    if "date_sched" not in df.columns:
        # Legacy completed calendars predate the scheduled-date field.  They
        # remain auditable from date_played; pending calendars must provide
        # date_sched so their effective fixture date is still defined.
        df["date_sched"] = df["date_played"]
    else:
        df["date_sched"] = floor_day(df["date_sched"])
    df["_fixture_date"] = df["date_played"].fillna(df["date_sched"])
    pending = (
        df["observation_status"].astype(str).eq("fixture_pending")
        if "observation_status" in df.columns
        else pd.Series(False, index=df.index)
    )

    # Strings for IDs/labels
    for c in ["player_id","team_id","match_id","fbref_id","fpl_id","team","venue","starter_source"]:
        df[c] = df[c].astype(str)

    # Binarys / numerics
    df["was_home"]   = pd.to_numeric(df["was_home"], errors="coerce").astype("Int8")
    df["is_active"]  = pd.to_numeric(df["is_active"], errors="coerce").astype("Int8")
    df["is_starter"] = pd.to_numeric(df["is_starter"], errors="coerce").astype("Int8")

    # Range checks
    assert set(df["is_starter"].dropna().unique()).issubset({0,1}), f"{season_dir.name}: is_starter must be 0/1"
    assert set(df["was_home"].dropna().unique()).issubset({0,1}),    f"{season_dir.name}: was_home must be 0/1"

    # starter_source values
    allowed_src = {
        "fpl", "fallback", "imputed", "reconstructed_fpl_dnp", "pending"
    }
    bad_src = set(df["starter_source"].dropna().unique()) - allowed_src
    assert not bad_src, f"{season_dir.name}: unexpected starter_source values: {bad_src}"

    if "bonus" in df.columns:
        ok_bonus = pd.to_numeric(df["bonus"], errors="coerce")
        assert ok_bonus.dropna().isin([0,1,2,3]).all(), f"{season_dir.name}: bonus must be in {{0,1,2,3}}"

    if "bps" in df.columns:
        ok_bps = pd.to_numeric(df["bps"], errors="coerce")
        notna = ok_bps.dropna()
        band_ok = notna.between(bps_min, bps_max).all()
        int_ok  = (notna == notna.round()).all()
        assert band_ok and int_ok, (
            f"{season_dir.name}: bps out of plausible band [{bps_min},{bps_max}] or non-integer present"
        )

    # GK rule: GK with minutes>0 must be starter (we backfill in builder)
    if "pos" in df.columns:
        is_gk = df["pos"].astype(str).str.upper().str.contains("GK", na=False)
        bad_gk = is_gk & (pd.to_numeric(df["minutes"], errors="coerce") > 0) & (df["is_starter"] == 0)
        assert not bad_gk.any(), f"{season_dir.name}: GKs with minutes>0 must have is_starter=1"

    # Keys non-null
    keys = ["player_id","match_id","team_id","_fixture_date","was_home"]
    assert df[keys].notna().all().all(), f"{season_dir.name}: nulls in join keys"

    # Uniqueness: one row per player-match-team
    vc = df.groupby(["player_id","match_id","team_id"]).size().value_counts().to_dict()
    assert set(vc) == {1}, f"{season_dir.name}: duplicates on (player_id, match_id, team_id): {vc}"

    # Coverage (price/xP)
    completed_df = df.loc[~pending]
    coverage_df = completed_df
    if "row_source" in coverage_df.columns:
        # Provider-only appearance rows are retained precisely because the
        # archived FPL universe lacks that player-fixture record; FPL-owned
        # price/xP coverage is undefined for those conservative residuals.
        coverage_df = coverage_df[
            coverage_df["row_source"].astype(str).ne("provider_only_observation")
        ]
    if "price" in df.columns and not coverage_df.empty:
        price_cov = coverage_df["price"].notna().mean()
        assert price_cov >= price_min, (
            f"{season_dir.name}: price coverage {price_cov:.3%} < {price_min:.1%}"
        )
    if "xp" in df.columns and not coverage_df.empty:
        xp_cov = coverage_df["xp"].notna().mean()
        assert xp_cov >= xp_min, (
            f"{season_dir.name}: xp coverage {xp_cov:.3%} < {xp_min:.1%}"
        )

    # Soft warnings:
    # 1) starters flagged but 0 minutes
    m0 = df[(df["is_starter"]==1) & (pd.to_numeric(df["minutes"], errors="coerce")==0)]
    if not m0.empty:
        warn(f"{season_dir.name}: {len(m0)} rows where is_starter=1 but minutes=0 (soft)")

    # 2) per-team starter counts ~11
    cnt = (completed_df.groupby(["match_id","team_id"])["is_starter"].sum().reset_index(name="starters"))
    if not cnt.empty:
        too_low  = cnt["starters"] < 10
        too_high = cnt["starters"] > 12
        n_low, n_high = int(too_low.sum()), int(too_high.sum())
        if n_low or n_high:
            warn(f"{season_dir.name}: starter counts outside [10..12] for {n_low+n_high} team-games (soft)")

    return df

def cross_validate_fixture_vs_pmc(fix: pd.DataFrame,
                                  pmc: pd.DataFrame,
                                  season_name: str):
    # Skip if either side empty
    if fix.empty or pmc.empty:
        return

    # Ensure fixture join index has no duplicates
    right = fix[["match_id","team_id","date_played","date_sched","was_home"]].copy()
    right["_fixture_date"] = right["date_played"].fillna(right["date_sched"])
    right = right[["match_id", "team_id", "_fixture_date", "was_home"]]
    right_dups = right.duplicated().sum()
    assert right_dups == 0, f"{season_name}: duplicate keys in fixture_calendar join index"

    # Map player rows to provider-neutral fixture identities.
    left = pmc[["player_id","match_id","team_id","_fixture_date","was_home"]].copy()
    m = left.merge(right.drop_duplicates(),
                   on=["match_id","team_id","_fixture_date","was_home"],
                   how="left", indicator=True)
    miss = int((m["_merge"] == "left_only").sum())
    assert miss == 0, (
        f"{season_name}: {miss} pmc rows don't map to a fixture_calendar row "
        f"by (match_id,team_id,effective fixture date,was_home)"
    )

def try_fbref_lineups_crosscheck(fbref_league_dir: Path, season: str, pmc: pd.DataFrame):
    """Optional: If a lineup CSV exists, soft-check starters vs that source."""
    season_dir = fbref_league_dir / season
    target = None
    for rel in LINEUPS_CANDIDATES:
        cand = season_dir / rel
        if cand.exists():
            target = cand
            break
    if target is None or pmc.empty:
        return
    try:
        lu = pd.read_csv(target)
        # try to be flexible with column names
        col_map = {
            "game_id": "fbref_id",
            "team_id": "team_id",
            "player_id": "player_id",
            "starter": "starter",
            "is_starter": "starter",
            "started": "starter",
        }
        # normalize
        rename = {}
        for c in lu.columns:
            lc = c.lower()
            if lc in col_map:
                rename[c] = col_map[lc]
            elif c in col_map:
                rename[c] = col_map[c]
        lu = lu.rename(columns=rename)
        req = {"fbref_id","team_id","player_id","starter"}
        if not req.issubset(set(lu.columns)):
            warn(f"{season}: lineup file at {target} lacks required columns; skipping cross-check")
            return
        lu["fbref_id"] = lu["fbref_id"].astype(str)
        lu["team_id"]  = lu["team_id"].astype(str)
        lu["player_id"]= lu["player_id"].astype(str)
        lu["starter"]  = pd.to_numeric(lu["starter"], errors="coerce").fillna(0).astype(int)

        left = pmc[["player_id","fbref_id","team_id","is_starter"]].copy()
        left["player_id"] = left["player_id"].astype(str)
        left["fbref_id"]  = left["fbref_id"].astype(str)
        left["team_id"]   = left["team_id"].astype(str)

        j = left.merge(lu[["player_id","fbref_id","team_id","starter"]].drop_duplicates(),
                       on=["player_id","fbref_id","team_id"], how="inner")
        if j.empty:
            warn(f"{season}: lineup cross-check found no overlapping rows (soft)"); return

        mism = (j["is_starter"].astype(int) != j["starter"].astype(int)).sum()
        rate = mism / len(j)
        if mism:
            warn(f"{season}: lineup cross-check mismatches: {mism}/{len(j)} = {rate:.2%} (soft)")
    except Exception as e:
        warn(f"{season}: lineup cross-check error: {e} (soft)")

def validate_registry_membership(df_fix: pd.DataFrame,
                                 df_pmc: pd.DataFrame,
                                 teams_lookup: dict,
                                 players_lookup: dict,
                                 season_name: str):
    valid_team_ids = set(map(str, teams_lookup.values()))
    valid_player_ids = set(map(str, players_lookup.values()))

    bad_teams_fix = set(df_fix["team_id"].astype(str)) - valid_team_ids if not df_fix.empty else set()
    bad_teams_pmc = set(df_pmc["team_id"].astype(str)) - valid_team_ids if not df_pmc.empty else set()
    assert not bad_teams_fix, f"{season_name}: team_ids in fixture_calendar not in registry: {bad_teams_fix}"
    assert not bad_teams_pmc, f"{season_name}: team_ids in player fixture calendar not in registry: {bad_teams_pmc}"

    if not df_pmc.empty:
        bad_players = set(df_pmc["player_id"].astype(str)) - valid_player_ids
        assert not bad_players, f"{season_name}: player_ids in pmc not in registry: {bad_players}"

def random_sampling_probe(pmc_all: list[pd.DataFrame], seasons: list[str]):
    if not pmc_all:
        return
    df = pd.concat([d for d in pmc_all if not d.empty], ignore_index=True) if any(not d.empty for d in pmc_all) else pd.DataFrame()
    if "observation_status" in df.columns:
        df = df[df["observation_status"].astype(str).ne("fixture_pending")].copy()
    if df.empty:
        return
    players = df["player_id"].dropna().unique().tolist()
    if len(players) == 0:
        return

    sample_pids = random.sample(players, min(SAMPLE_PLAYERS, len(players)))
    sample = df[df["player_id"].isin(sample_pids)].sample(
        min(SAMPLE_ROWS, len(df)), replace=False, random_state=42
    )

    must_have = ["player_id","match_id","team_id","date_played","was_home","minutes"]
    nulls = {c: int(sample[c].isna().sum()) for c in must_have if c in sample.columns}
    assert all(v == 0 for v in nulls.values()), f"Random sample probe found nulls in {nulls}"

# ---------- Runner ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-root", type=Path, default=Path("data/processed/registry/fixtures"))
    ap.add_argument(
        "--fbref-league-dir",
        type=Path,
        default=Path("data/processed/fbref/ENG-Premier League"),
        help="Optional FBref root used only for a soft lineup cross-check",
    )
    ap.add_argument("--teams-lookup", type=Path, default=Path("data/processed/registry/_id_lookup_teams.json"))
    ap.add_argument("--players-lookup", type=Path, default=Path("data/processed/registry/_id_lookup_players.json"))
    ap.add_argument("--seasons", type=str, default="")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--price-min", type=float, default=PRICE_COVERAGE_MIN_DEFAULT)
    ap.add_argument("--xp-min", type=float, default=XP_COVERAGE_MIN_DEFAULT)
    ap.add_argument("--bps-min", type=float, default=BPS_MIN_DEFAULT)
    ap.add_argument("--bps-max", type=float, default=BPS_MAX_DEFAULT)
    args = ap.parse_args()

    price_min = float(args.price_min)
    xp_min    = float(args.xp_min)
    bps_min   = float(args.bps_min)
    bps_max   = float(args.bps_max)

    seasons = (
        [s.strip() for s in args.seasons.split(",") if s.strip()]
        if args.seasons
        else sorted(d.name for d in args.fixtures_root.iterdir() if d.is_dir())
    )

    teams_lookup   = load_json(args.teams_lookup)
    players_lookup = load_json(args.players_lookup)
    # A name lookup cannot contain two IDs for an ambiguous canonical name.
    # Canonical membership is therefore the union of lookup targets and master
    # registry keys, not lookup targets alone.
    master_players_path = args.players_lookup.parent / "master_players.json"
    if master_players_path.is_file():
        master_players = load_json(master_players_path)
        players_lookup.update(
            {f"__master_player__{player_id}": player_id for player_id in master_players}
        )

    failures: list[str] = []
    pmc_all: list[pd.DataFrame] = []

    for s in seasons:
        season_dir = args.fixtures_root / s
        try:
            fix = validate_fixture_calendar(season_dir)
            pmc = validate_player_minutes_calendar(season_dir, price_min, xp_min, bps_min, bps_max)
            cross_validate_fixture_vs_pmc(fix, pmc, s)
            validate_registry_membership(fix, pmc, teams_lookup, players_lookup, s)
            try_fbref_lineups_crosscheck(args.fbref_league_dir, s, pmc)
            pmc_all.append(pmc)
            print(f"OK: {s}: fixture_calendar + player calendar + joins + registry")
        except (AssertionError, KeyError, ValueError, FileNotFoundError) as e:
            failures.append(str(e))
            print(f"FAIL: {s}: {e}")

    # Cross-season probe
    try:
        random_sampling_probe(pmc_all, seasons)
        print("OK: Cross-season random sampling probe")
    except AssertionError as e:
        failures.append(str(e))
        print(f"FAIL: Cross-season probe: {e}")

    if failures:
        print("\n=== FAILURES ===")
        for f in failures:
            print("-", f)
        sys.exit(1)

    print("\nAll assurance checks passed.")

if __name__ == "__main__":
    main()
