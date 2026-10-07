import json
from uuid import uuid4

import pandas as pd
import pytest

from fpl_assistant.pipelines.integrate.fixtures_meta_builder import build_fixture_calendar
from fpl_assistant.testing.paths import get_test_run_dir


def _inputs():
    root = get_test_run_dir("fixture_date_reconciliation") / uuid4().hex
    root.mkdir()
    pd.DataFrame([dict(id=45, event=5, kickoff_time="2026-09-20T13:00:00Z",
                       team_h=1, team_a=2, finished=True)]).to_csv(root / "fixtures.csv", index=False)
    pd.DataFrame([dict(id=1, name="Manchester City"), dict(id=2, name="Sunderland")]).to_csv(
        root / "teams.csv", index=False
    )
    (root / "ids.json").write_text(json.dumps({"Manchester City": "city", "Sunderland": "sun"}))
    (root / "codes.json").write_text(json.dumps({"Manchester City": "MCI", "Sunderland": "SUN"}))
    ws, und = [], []
    for home in (True, False):
        team, opp = ("city", "sun") if home else ("sun", "city")
        ws.append(dict(match_id="match1", team_id=team, opponent_id=opp,
                       team="MCI" if home else "SUN", home="MCI", away="SUN",
                       game_date="2026-09-19", venue="Home" if home else "Away", is_home=home))
        und.append(dict(match_id="match1", team_id=team, opp_id=opp,
                        game_date="2026-09-20", team_goals=5 if home else 3,
                        opp_goals=3 if home else 5, team_xg=2.0, opp_xg=1.0,
                        result="W" if home else "L", is_result=True))
    pd.DataFrame(ws).to_csv(root / "ws.csv", index=False)
    pd.DataFrame(und).to_csv(root / "und.csv", index=False)
    return dict(season="2026-2027", fpl_csv=root / "fixtures.csv",
                teams_csv=root / "teams.csv", fb_csv=root / "fbref/team_match/schedule.csv",
                ws_csv=root / "ws.csv", und_csv=root / "und.csv",
                team_map_fp=root / "ids.json", short_map_fp=root / "codes.json",
                out_dir=root / "output", attach_fdr=None, features_root=root / "features",
                views_subdir="views", force=True)


def test_reconciles_both_team_rows_preserves_sources_and_refreshes_audit():
    args = _inputs()
    originals = {key: args[key].read_bytes() for key in ("ws_csv", "und_csv", "fpl_csv")}
    assert build_fixture_calendar(**args)
    dst = args["out_dir"] / args["season"]
    calendar = pd.read_csv(dst / "fixture_calendar.csv")
    audit = pd.read_csv(dst / "_date_reconciliation_audit.csv")
    assert len(calendar) == len(audit) == 2
    assert calendar.date_played.eq("2026-09-20").all()
    assert calendar.match_id.eq("match1").all()
    assert set(calendar.team_id) == {"city", "sun"}
    assert audit.whoscored_date.eq("2026-09-19").all()
    assert audit.understat_date.eq("2026-09-20").all()
    assert audit.fpl_date.eq("2026-09-20").all()
    assert audit.fpl_id.eq(45).all()
    for key, content in originals.items():
        assert args[key].read_bytes() == content
    ws = pd.read_csv(args["ws_csv"])
    ws["game_date"] = "2026-09-20"
    ws.to_csv(args["ws_csv"], index=False)
    assert build_fixture_calendar(**args)
    assert pd.read_csv(dst / "_date_reconciliation_audit.csv").empty


@pytest.mark.parametrize("case", ["unfinished", "different_date", "missing_date", "reversed", "duplicate", "missing"])
def test_rejects_uncorroborated_conflicts_without_overwriting_outputs(case):
    args = _inputs()
    fpl = pd.read_csv(args["fpl_csv"])
    if case == "unfinished":
        fpl["finished"] = False
    elif case == "different_date":
        fpl["kickoff_time"] = "2026-09-21T13:00:00Z"
    elif case == "missing_date":
        fpl["kickoff_time"] = None
    elif case == "reversed":
        fpl["team_h"], fpl["team_a"] = 2, 1
    elif case == "duplicate":
        fpl = pd.concat([fpl, fpl.assign(id=46, finished=False)], ignore_index=True)
    else:
        fpl = fpl.iloc[:0]
    fpl.to_csv(args["fpl_csv"], index=False)
    dst = args["out_dir"] / args["season"]
    dst.mkdir(parents=True)
    for name in ("fixture_calendar.csv", "_date_reconciliation_audit.csv"):
        (dst / name).write_text("previous output")
    with pytest.raises(ValueError, match="conflict on 2 rows without unique finished"):
        build_fixture_calendar(**args)
    for name in ("fixture_calendar.csv", "_date_reconciliation_audit.csv"):
        assert (dst / name).read_text() == "previous output"
