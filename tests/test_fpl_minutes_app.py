from pathlib import Path

import pandas as pd

from apps.fpl.minutes import add_fixture_minutes, forecast_files, load_minutes, minutes_table


def test_discovery_orders_gameweeks_and_excludes_other_artifacts(tmp_path: Path):
    folder = tmp_path / "2026-2027"
    folder.mkdir()
    for name in ("GW02.csv", "GW10.csv", "GW10_audit.csv", "GW10.csv.meta.json"):
        (folder / name).touch()
    assert [p.name for p in forecast_files("2026-2027", tmp_path)] == ["GW10.csv", "GW02.csv"]
    assert forecast_files("2025-2026", tmp_path) == []


def test_minutes_respects_overrides_zero_and_season(tmp_path: Path):
    path = tmp_path / "GW05.csv"
    pd.DataFrame({
        "season": ["2026-2027"] * 3 + ["2025-2026"],
        "pred_exp_minutes_final": [0, 80, None, 90],
        "pred_exp_minutes": [70, 70, 60, 90],
    }).to_csv(path, index=False)
    data, _ = load_minutes("2026-2027", path)
    assert data["pred_minutes"].tolist() == [0, 80, 60]


def test_double_gameweek_uses_fixture_and_player_identity():
    data = pd.DataFrame({
        "player_id": ["a", "a", "b"], "fpl_id": [41, 42, 41],
        "pred_minutes": [0, 80, 90], "p_cameo_state": [.1, .2, .3],
        "p_cameo_cal": [.5, .6, .7],
    })
    fixtures = pd.DataFrame({"fpl_id": [41, 42, 43], "pred_minutes": [70, 70, 60]})
    result = add_fixture_minutes(fixtures, data, "a")
    assert result["pred_minutes"].tolist() == [0, 80, 60]
    assert minutes_table(data)["Cameo %"].tolist() == [10, 20, 30]
    assert len(minutes_table(data)) == 3
