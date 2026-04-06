import pandas as pd
import numpy as np
import pytest

from scripts.fbref_pipeline.integrate import team_form_builder as tfb


def _fixture_rows(
    season: str,
    fpl_id: int,
    gw_orig: int,
    date_played: str,
    home_id: str,
    away_id: str,
    home_att_z: float,
    home_def_z: float,
    away_att_z: float,
    away_def_z: float,
    home_gf: int,
    home_ga: int,
    away_gf: int,
    away_ga: int,
):
    dt = pd.to_datetime(date_played)
    rows = [
        {
            "season": season,
            "fpl_id": fpl_id,
            "gw_orig": gw_orig,
            "date_played": dt,
            "team_id": home_id,
            "home_id": home_id,
            "away_id": away_id,
            "is_home": 1,
            "att_xg_home_roll_z": home_att_z,
            "def_xga_home_roll_z": home_def_z,
            "att_xg_away_roll_z": np.nan,
            "def_xga_away_roll_z": np.nan,
            "gf": home_gf,
            "ga": home_ga,
        },
        {
            "season": season,
            "fpl_id": fpl_id,
            "gw_orig": gw_orig,
            "date_played": dt,
            "team_id": away_id,
            "home_id": home_id,
            "away_id": away_id,
            "is_home": 0,
            "att_xg_home_roll_z": np.nan,
            "def_xga_home_roll_z": np.nan,
            "att_xg_away_roll_z": away_att_z,
            "def_xga_away_roll_z": away_def_z,
            "gf": away_gf,
            "ga": away_ga,
        },
    ]
    return rows


def test_fdr_continuous_and_bucket_columns():
    rows = _fixture_rows(
        season="2024-2025",
        fpl_id=1001,
        gw_orig=8,
        date_played="2024-10-01",
        home_id="h1",
        away_id="a1",
        home_att_z=0.4,
        home_def_z=1.0,
        away_att_z=0.8,
        away_def_z=1.2,
        home_gf=2, home_ga=1,
        away_gf=1, away_ga=2,
    )
    df = pd.DataFrame(rows)

    fixtures, audit_df, home_adv = tfb._compute_fixture_fdr_difficulty(
        df,
        season="2024-2025",
        bucket_mode="global",
        hybrid_cutoff=6,
        home_adv_mode="none",
        shrink_matches=0,
        do_audit=False,
    )

    assert audit_df is None
    assert home_adv == 0.0

    # Continuous columns exist
    for c in tfb.FDR_CONT_COLS:
        assert c in fixtures.columns

    # Buckets exist
    for c in tfb.FDR_BUCKET_COLS:
        assert c in fixtures.columns

    # Check continuous values against expected math
    # home_att = -away_def_z, home_def = +away_att_z
    exp_home_att = -1.2
    exp_home_def = 0.8
    exp_home = 0.5 * exp_home_att + 0.5 * exp_home_def
    # away_att = -home_def_z, away_def = +home_att_z
    exp_away_att = -1.0
    exp_away_def = 0.4
    exp_away = 0.5 * exp_away_att + 0.5 * exp_away_def

    row = fixtures.iloc[0]
    assert row["fdr_att_home_cont"] == pytest.approx(exp_home_att, 1e-6)
    assert row["fdr_def_home_cont"] == pytest.approx(exp_home_def, 1e-6)
    assert row["fdr_home_cont"] == pytest.approx(exp_home, 1e-6)
    assert row["fdr_att_away_cont"] == pytest.approx(exp_away_att, 1e-6)
    assert row["fdr_def_away_cont"] == pytest.approx(exp_away_def, 1e-6)
    assert row["fdr_away_cont"] == pytest.approx(exp_away, 1e-6)


def test_fdr_shrinkage_early_gw():
    rows = _fixture_rows(
        season="2024-2025",
        fpl_id=2001,
        gw_orig=1,
        date_played="2024-08-10",
        home_id="h2",
        away_id="a2",
        home_att_z=0.9,
        home_def_z=0.7,
        away_att_z=-0.5,
        away_def_z=0.3,
        home_gf=1, home_ga=0,
        away_gf=0, away_ga=1,
    )
    df = pd.DataFrame(rows)

    fixtures, _, _ = tfb._compute_fixture_fdr_difficulty(
        df,
        season="2024-2025",
        bucket_mode="global",
        hybrid_cutoff=6,
        home_adv_mode="none",
        shrink_matches=6,
        do_audit=False,
    )

    # gw=1 => alpha=0 => all continuous scores shrink to 0
    for c in tfb.FDR_CONT_COLS:
        assert float(fixtures.iloc[0][c]) == pytest.approx(0.0, 1e-9)


def test_fdr_home_adv_constant_shift():
    rows = _fixture_rows(
        season="2024-2025",
        fpl_id=3001,
        gw_orig=10,
        date_played="2024-11-01",
        home_id="h3",
        away_id="a3",
        home_att_z=0.2,
        home_def_z=0.6,
        away_att_z=0.1,
        away_def_z=0.4,
        home_gf=1, home_ga=1,
        away_gf=1, away_ga=1,
    )
    df = pd.DataFrame(rows)

    base, _, _ = tfb._compute_fixture_fdr_difficulty(
        df,
        season="2024-2025",
        bucket_mode="global",
        hybrid_cutoff=6,
        home_adv_mode="none",
        shrink_matches=0,
        do_audit=False,
    )
    shifted, _, _ = tfb._compute_fixture_fdr_difficulty(
        df,
        season="2024-2025",
        bucket_mode="global",
        hybrid_cutoff=6,
        home_adv_mode="constant",
        home_adv_value=1.0,
        shrink_matches=0,
        do_audit=False,
    )

    # With w_att=w_def=0.5, home_adv=1.0 shifts each composite score by 0.5.
    dh = shifted.iloc[0]["fdr_home_cont"] - base.iloc[0]["fdr_home_cont"]
    da = shifted.iloc[0]["fdr_away_cont"] - base.iloc[0]["fdr_away_cont"]
    assert dh == pytest.approx(-0.5, 1e-6)
    assert da == pytest.approx(0.5, 1e-6)


def test_fdr_audit_emits_rows():
    rows = []
    for i in range(10):
        rows.extend(
            _fixture_rows(
                season="2024-2025",
                fpl_id=4000 + i,
                gw_orig=7 + i,
                date_played=f"2024-10-{10+i:02d}",
                home_id=f"h{i}",
                away_id=f"a{i}",
                home_att_z=0.1 * i,
                home_def_z=0.2 * i,
                away_att_z=-0.1 * i,
                away_def_z=0.15 * i,
                home_gf=2 if i % 2 == 0 else 1,
                home_ga=1 if i % 2 == 0 else 2,
                away_gf=1 if i % 2 == 0 else 2,
                away_ga=2 if i % 2 == 0 else 1,
            )
        )
    df = pd.DataFrame(rows)

    fixtures, audit_df, _ = tfb._compute_fixture_fdr_difficulty(
        df,
        season="2024-2025",
        bucket_mode="global",
        hybrid_cutoff=6,
        home_adv_mode="none",
        shrink_matches=0,
        do_audit=True,
    )

    assert fixtures.shape[0] == 10
    assert audit_df is not None
    assert not audit_df.empty
    assert set(audit_df["metric"]) == {
        "home_fdr_vs_home_ga",
        "home_fdr_vs_home_gf",
        "away_fdr_vs_away_ga",
        "away_fdr_vs_away_gf",
    }
