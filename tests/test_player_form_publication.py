import uuid
from pathlib import Path

import pandas as pd

from fpl_assistant.pipelines.integrate.calendar_builder import _derive_was_home
from fpl_assistant.pipelines.integrate.team_form_builder import (
    _resolve_version as resolve_team_form_version,
)
from fpl_assistant.pipelines.integrate.player_form_builder import (
    _coerce_columns,
    _copy_to_latest_dir,
    _resolve_version as resolve_player_form_version,
    resolve_season_selection,
)


def _case_dir(name: str) -> Path:
    path = Path(".tmp") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_player_form_season_selector_supports_all_latest_and_short_names():
    fixtures = _case_dir("player_form_seasons")
    for season in ("2024-2025", "2025-2026"):
        (fixtures / season).mkdir()

    assert resolve_season_selection(fixtures, season="all") == [
        "2024-2025",
        "2025-2026",
    ]
    assert resolve_season_selection(fixtures, season="latest") == ["2025-2026"]
    assert resolve_season_selection(fixtures, season="2025-26") == ["2025-2026"]
    assert resolve_season_selection(
        fixtures,
        season="all",
        seasons_csv="2024-25,2025-2026",
    ) == ["2024-2025", "2025-2026"]


def test_player_form_publication_copies_output_without_removing_team_form():
    root = _case_dir("player_form_publish")
    source = root / "v3" / "2025-2026"
    latest = root / "latest" / "2025-2026"
    source.mkdir(parents=True)
    latest.mkdir(parents=True)
    (source / "players_form.csv").write_text("player_id\np1\n", encoding="utf-8")
    (source / "player_form.meta.json").write_text("{}", encoding="utf-8")
    (latest / "team_form.csv").write_text("team_id\nt1\n", encoding="utf-8")

    _copy_to_latest_dir(root, "v3", "2025-2026")

    assert (latest / "players_form.csv").is_file()
    assert (latest / "player_form.meta.json").is_file()
    assert (latest / "team_form.csv").is_file()


def test_form_builders_publish_to_latest_by_default_and_version_only_on_request():
    root = _case_dir("form_versions")
    (root / "v2").mkdir()
    (root / "v7").mkdir()

    assert resolve_team_form_version(root, requested=None, auto=False) == "latest"
    assert resolve_player_form_version(root, requested=None, auto=False) == "latest"
    assert resolve_team_form_version(root, requested=None, auto=True) == "v8"
    assert resolve_player_form_version(root, requested="auto", auto=False) == "v8"


def test_stadium_venue_is_preserved_and_home_flag_comes_from_team_identity():
    frame = pd.DataFrame(
        {
            "team_id": ["home-team", "away-team"],
            "home_id": ["home-team", "home-team"],
            "venue": ["Old Trafford", "Old Trafford"],
        }
    )

    was_home = _derive_was_home(frame)

    assert was_home.tolist() == [1, 0]
    assert frame["venue"].tolist() == ["Old Trafford", "Old Trafford"]


def test_player_form_uses_was_home_without_rewriting_stadium_or_dropping_gkp_stats():
    frame = pd.DataFrame(
        {
            "season": ["2025-2026", "2025-2026"],
            "date_played": ["2025-08-16", "2025-08-23"],
            "gw_orig": [1, 2],
            "player_id": ["gk-1", "gk-1"],
            "team_id": ["team-1", "team-1"],
            "opponent_id": ["team-2", "team-3"],
            "player": ["Keeper", "Keeper"],
            "pos": ["GKP", "GKP"],
            "venue": ["Old Trafford", "Emirates Stadium"],
            "was_home": [1, 0],
            "minutes": [90, 90],
            "saves": [4, 2],
            "sot_against": [5, 3],
            "save_pct": [80.0, 66.7],
        }
    )

    coerced, _ = _coerce_columns(frame, fill_missing_fdr=None)

    assert coerced["venue"].tolist() == ["Old Trafford", "Emirates Stadium"]
    assert coerced["was_home"].tolist() == [1, 0]
    assert coerced["pos"].tolist() == ["GKP", "GKP"]
    assert coerced["saves"].tolist() == [4, 2]
