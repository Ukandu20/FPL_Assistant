"""Keep production code independent of the retired script tree."""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pandas as pd

from fpl_assistant.platform.scrape_runs import (
    ScrapeJobId, get_last_run, record_last_run, should_run,
)
from fpl_assistant.providers.whoscored.scrape import whoscored_match_stats_scraper_soccerdata


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "fpl_assistant"


def test_package_has_no_legacy_imports_or_launch_targets():
    violations = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names = [node.value]
            if any(name.startswith(("scripts.", "src.fpl_assistant.")) for name in names):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, violations


def test_production_implementations_are_not_mirrored():
    implementations = {}
    for path in PACKAGE.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        if not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body):
            continue
        digest = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
        assert digest not in implementations, (path, implementations.get(digest))
        implementations[digest] = path
    assert sorted(path.relative_to(ROOT / "scripts").as_posix()
                  for path in (ROOT / "scripts").rglob("*.py")) == ["generate_archetype_v1_sample.py"]


def test_scrape_scheduler_reads_writer_records(tmp_path):
    path = tmp_path / "runs.json"
    job = ScrapeJobId("match", "ENG-Premier League", "2026-2027", "both")
    now = datetime.now(timezone.utc)
    record_last_run(path, job, {"scrape_ts": now.isoformat(), "mode": "manual"})
    assert get_last_run(path, job) == now
    assert not should_run(path, job, timedelta(days=1))
    record_last_run(path, job, {"scrape_ts": (now - timedelta(days=2)).isoformat()})
    assert should_run(path, job, timedelta(days=1))


@pytest.mark.parametrize("timestamp", ["2026-09-01T12:00:00Z", "2026-09-01T12:00:00+0000", "2026-09-01T12:00:00"])
def test_scrape_scheduler_retains_timestamp_only_history(tmp_path, timestamp):
    path = tmp_path / "runs.json"
    job = ScrapeJobId("match", "ENG-Premier League", "2026-2027", "both")
    path.write_text(json.dumps({"match:ENG-Premier League:2026-2027:both": timestamp}))
    assert get_last_run(path, job) == datetime(2026, 9, 1, 12, tzinfo=timezone.utc)


def test_soccerdata_entrypoint_preserves_command_line(monkeypatch):
    forwarded = []
    module = whoscored_match_stats_scraper_soccerdata
    monkeypatch.setattr(module, "whoscored_main", forwarded.append)
    monkeypatch.setattr(module.sys, "argv", ["scraper", "--league", "ENG-Premier League", "--seasons", "2026-2027"])
    module.main()
    assert forwarded == [["--backend", "soccerdata", "--league", "ENG-Premier League", "--seasons", "2026-2027"]]


@pytest.mark.parametrize("entity", ["player", "team"])
def test_table_cleaner_preserves_both_entity_formats(tmp_path, entity):
    from fpl_assistant.providers.fbref.clean.table_cleaner import clean_csv

    source = tmp_path / "raw.csv"
    source.write_text("player,pos,pl\nplayer,pos,pl\nTest Player,MF-FW,22\n")
    output = tmp_path / "clean"
    output.mkdir()
    clean_csv(source, output, [], entity=entity)
    frame = pd.read_csv(output / "raw.csv")
    if entity == "team":
        assert frame.loc[0, "no_of_players_used"] == 22
        assert "pos_primary" not in frame.columns
        assert "first_name" not in frame.columns
    else:
        assert frame.loc[0, "pos_primary"] == "MF"
        assert frame.loc[0, "pos_alt"] == "FW"
        assert frame.loc[0, "first_name"] == "Test"
        assert frame.loc[0, "last_name"] == "Player"
        assert frame.loc[0, "pl"] == 22


@pytest.mark.parametrize("players_only", [False, True])
def test_bulk_export_retains_all_and_player_only_modes(monkeypatch, tmp_path, players_only):
    from types import SimpleNamespace
    from fpl_assistant.tools.legacy import fbref_bulk_export as bulk

    calls = []
    reader = SimpleNamespace(
        read_team_season_stats=lambda: None,
        read_team_match_stats=lambda: None,
        read_player_season_stats=lambda: None,
        read_schedule=lambda: pd.DataFrame(index=pd.Index([], name="game_id")),
    )
    monkeypatch.setattr(bulk.sd, "FBref", lambda **kwargs: reader)
    monkeypatch.setattr(bulk, "loop_stats", lambda reader, dest, level, *args: calls.append(level))
    bulk.scrape_season("ENG-Premier League", "2026-2027", tmp_path, players_only=players_only)
    assert calls == (["player_season"] if players_only else ["team_season", "team_match", "player_season"])


def test_optimizer_imports_do_not_require_historical_strategy_engine(monkeypatch):
    from fpl_assistant import optimizers
    from fpl_assistant.optimizers import strategy, single_gw, mc

    assert callable(single_gw.main)
    assert callable(mc.simulate_gw)
    assert optimizers.team_state.__name__ == "fpl_assistant.domain.team_state"

    def unavailable(name):
        raise ModuleNotFoundError("missing historical engine", name=name)

    monkeypatch.setattr(strategy, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="external mc_sim_v01 engine"):
        strategy._load_mc_engine()
