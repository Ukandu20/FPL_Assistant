import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.transfermarkt_pipeline.scrape.manager_history_scraper import (
    build_output_path,
    parse_manager_history_html,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "transfermarkt_mun_manager_history.html"
)


def test_parse_manager_history_html_extracts_expected_rows_and_types():
    html = FIXTURE_PATH.read_text(encoding="utf-8")

    rows = parse_manager_history_html(html, as_of_date=date(2026, 4, 6))

    assert len(rows) >= 1
    assert list(rows[0].keys()) == [
        "name",
        "nationality",
        "appointment_date",
        "end_date",
        "tenure",
        "matches",
    ]

    first = rows[0]
    assert first["name"] == "Michael Carrick"
    assert first["nationality"] == "England"
    assert first["appointment_date"] == "2026-01-13"
    assert first["end_date"] == ""
    assert isinstance(first["tenure"], int)
    assert first["tenure"] == 83
    assert isinstance(first["matches"], int)
    assert first["matches"] == 10


def test_parse_manager_history_html_computes_fallback_tenure_from_dates():
    html = FIXTURE_PATH.read_text(encoding="utf-8")

    rows = parse_manager_history_html(html, as_of_date=date(2026, 4, 6))

    amorim = next(row for row in rows if row["name"] == "Rúben Amorim")
    assert amorim["appointment_date"] == "2024-11-11"
    assert amorim["end_date"] == "2026-01-05"
    assert amorim["tenure"] == 420
    assert amorim["matches"] == 63


def test_build_output_path_uses_uppercase_team_code():
    out_path = build_output_path("mun", Path("data/raw/transfermarkt/premier_league/managers"))
    assert out_path == Path("data/raw/transfermarkt/premier_league/managers/MUN.csv")
