# -*- coding: utf-8 -*-
"""
Data quality tests for optimizer inputs & model preds.

Run:
  pytest -q -k data_inputs

Env:
- OPT_INPUT: path to optimizer_input.parquet
- EXP_POINTS_CSV: path to expected points CSV (merged)
- GA_CSV: optional path to goals_assists forecast CSV (for prob calibration checks)
"""
import os, math
import pandas as pd
import numpy as np
import pytest

OPT_INPUT = os.getenv("OPT_INPUT", "data/aggregator/optimizer_input.parquet")
EXP_POINTS = os.getenv("EXP_POINTS_CSV", "data/predictions/expected_points/latest.csv")
GA_CSV = os.getenv("GA_CSV", "")

CRITICAL_COLS = [
    "season","gw_orig","player_id","team_id","pos",
    "xPts","p60","is_home","fdr"
]

@pytest.mark.skipif(not os.path.exists(OPT_INPUT), reason="optimizer input parquet missing")
def test_optimizer_input_no_nans_and_types():
    df = pd.read_parquet(OPT_INPUT)
    cols_present = [c for c in CRITICAL_COLS if c in df.columns]
    assert cols_present, f"None of critical cols present in {OPT_INPUT}"
    # No NaNs in present critical columns
    bad = {c:int(df[c].isna().sum()) for c in cols_present if df[c].isna().any()}
    assert not bad, f"NaNs in critical cols: {bad}"

    # Dtypes: is_home ∈ {0,1}; fdr integer-ish; pos categorical 1 of {GK,DEF,MID,FWD}
    if "is_home" in df:
        assert set(map(int, pd.unique(df["is_home"].astype("Int64").fillna(0)))) <= {0,1}
    if "fdr" in df:
        assert np.all(np.mod(df["fdr"].dropna().to_numpy(), 1) == 0), "fdr should be integer-valued"
    if "pos" in df:
        allowed = {"GK","DEF","MID","FWD"}
        assert set(map(str.upper, df["pos"].dropna().astype(str))) <= allowed

@pytest.mark.skipif(not (os.path.exists(OPT_INPUT) and os.path.exists(EXP_POINTS)), reason="inputs missing")
def test_dgw_aggregation_rules():
    """
    Rule: if optimizer_input has per-GW rows aggregated from per-fixture preds,
    check that per (season, gw_orig, player_id) xPts equals sum over fixtures.
    """
    agg = pd.read_parquet(OPT_INPUT)
    exp = pd.read_csv(EXP_POINTS)

    keys = ["season","gw_orig","player_id"]
    have_game = "game_id" in exp

    if not have_game:
        pytest.skip("Expected points CSV lacks game_id; cannot verify per-fixture summation.")

    exp_sum = (exp
               .groupby(keys, dropna=False, as_index=False)["xPts"]
               .sum()
               .rename(columns={"xPts":"xPts_sum"}))
    merged = agg.merge(exp_sum, on=keys, how="inner", validate="one_to_one")
    if merged.empty:
        pytest.skip("No overlapping keys between optimizer input and expected points CSV.")

    # allow small numerical tolerance
    tol = 1e-6
    diffs = (merged["xPts"] - merged["xPts_sum"]).abs()
    assert bool((diffs <= tol).all()), f"DGW aggregation mismatch; max abs diff={diffs.max()}"

@pytest.mark.skipif(not os.path.exists(EXP_POINTS), reason="expected points CSV missing")
def test_monotonic_gw_within_season():
    df = pd.read_csv(EXP_POINTS)
    k = ["season","player_id"]
    df = df.sort_values(k + ["gw_orig"])
    # Check gw_orig strictly increasing *over time buckets* (ties allowed if true DGW rows)
    grp = df.groupby(k)["gw_orig"].apply(lambda s: (s.diff().fillna(1) >= 0).all())
    assert bool(grp.all()), "gw_orig must be non-decreasing per (season, player)"
