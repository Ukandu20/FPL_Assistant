import pandas as pd

from scripts.understat_pipeline.clean.clean_understat_raw import (
    build_fpl_mode_maps,
    build_teams_config_maps,
    map_league_value,
    map_player_match_pos,
    normalize_player_key,
    process_player_match,
    process_player_season,
    season_long_to_short,
    season_to_long,
    transform_schedule,
)


def test_league_and_season_mapping_helpers():
    assert map_league_value("EPL") == "ENG-Premier League"
    assert map_league_value("La liga") == "ESP-La Liga"
    assert season_to_long("2025") == "2025-2026"
    assert season_to_long("2025-26") == "2025-2026"
    assert season_long_to_short("2025-2026") == "2025-26"


def test_normalize_player_key_basic():
    assert normalize_player_key("Matt O&#039;Riley") == "matt o'riley"
    assert normalize_player_key("Ibrahima Konaté") == "ibrahima konate"
    assert normalize_player_key("Emile Smith-Rowe") == "emile smith rowe"


def test_player_match_position_mapping():
    assert map_player_match_pos("GK") == "GK"
    assert map_player_match_pos("DR") == "DEF"
    assert map_player_match_pos("MC") == "MID"
    assert map_player_match_pos("ML") == "MID"
    assert map_player_match_pos("FW") == "FWD"
    assert map_player_match_pos("Sub") == "SUB"
    assert map_player_match_pos("XYZ") == "UNK"


def test_schedule_transform_doubles_rows_and_rounds_and_result_blanks():
    df = pd.DataFrame(
        [
            {
                "league": "EPL",
                "season": 2025,
                "game": "2025-08-15_Liverpool_vs_Bournemouth",
                "game_id": 1,
                "date": "2025-08-15 19:00:00",
                "home_team_id": 87,
                "away_team_id": 73,
                "home_team": "Liverpool",
                "away_team": "Bournemouth",
                "home_team_code": "LIV",
                "away_team_code": "BOU",
                "home_goals": 2,
                "away_goals": 1,
                "is_result": True,
            },
            {
                "league": "EPL",
                "season": 2025,
                "game": "2025-08-22_Bournemouth_vs_Liverpool",
                "game_id": 2,
                "date": "2025-08-22 19:00:00",
                "home_team_id": 73,
                "away_team_id": 87,
                "home_team": "Bournemouth",
                "away_team": "Liverpool",
                "home_team_code": "BOU",
                "away_team_code": "LIV",
                "home_goals": None,
                "away_goals": None,
                "is_result": False,
            },
        ]
    )
    team_lookup = {"liv": "259f237e", "bou": "56f0abca"}
    out, team_missing = transform_schedule(
        df,
        league_std="ENG-Premier League",
        season_long="2025-2026",
        team_lookup=team_lookup,
    )

    assert len(out) == 4
    assert len(team_missing) == 0
    assert set(out["venue"].unique()) == {"H", "A"}
    assert out["game"].str.contains(" - ").all()
    assert set(out["game"].unique()) == {"LIV - BOU", "BOU - LIV"}
    assert (out["game_time"] == "19:00:00").all()

    liv = out[out["team"] == "LIV"].sort_values("round")
    assert liv["round"].tolist() == [1, 2]
    assert liv.iloc[0]["result"] == "W"
    assert liv.iloc[1]["result"] == ""


def test_schedule_transform_uses_teams_config_for_codes_and_names():
    df = pd.DataFrame(
        [
            {
                "league": "EPL",
                "season": 2025,
                "game": "2025-08-15_Brighton_vs_Fulham",
                "game_id": 1,
                "date": "2025-08-15 19:00:00",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team": "Brighton",
                "away_team": "Fulham",
                "home_team_code": "BRI",
                "away_team_code": "FLH",
                "home_goals": 2,
                "away_goals": 1,
                "is_result": True,
            }
        ]
    )
    teams_name_to_code, teams_name_to_display = build_teams_config_maps(
        {
            "Brighton": "BHA",
            "Brighton & Hove Albion": "BHA",
            "Fulham": "FUL",
        }
    )
    team_lookup = {"bha": "5dbeea62", "ful": "049e06ec"}
    out, team_missing = transform_schedule(
        df,
        league_std="ENG-Premier League",
        season_long="2025-2026",
        team_lookup=team_lookup,
        teams_name_to_code=teams_name_to_code,
        teams_name_to_display=teams_name_to_display,
    )

    assert len(team_missing) == 0
    assert set(out["team"].unique()) == {"BHA", "FUL"}
    assert out.loc[out["team"] == "BHA", "team_id"].eq("5dbeea62").all()
    assert out.loc[out["team"] == "FUL", "team_id"].eq("049e06ec").all()
    assert "team_id_missing" not in out.columns
    assert "opp_id_missing" not in out.columns


def test_schedule_transform_name_canonicalization_does_not_cross_code_collisions():
    df = pd.DataFrame(
        [
            {
                "league": "Bundesliga",
                "season": 2025,
                "game": "2025-08-15_Leverkusen_vs_Dortmund",
                "game_id": 1,
                "date": "2025-08-15 19:00:00",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team": "Leverkusen",
                "away_team": "Dortmund",
                "home_team_code": "BAY",
                "away_team_code": "DOR",
                "home_goals": 1,
                "away_goals": 1,
                "is_result": True,
            }
        ]
    )
    teams_name_to_code, teams_name_to_display = build_teams_config_maps(
        {
            "Leverkusen": "LEV",
            "Levante": "LEV",
            "Dortmund": "BVB",
        }
    )
    team_lookup = {"lev": "01f4c6e3", "bvb": "f8eb621b"}
    out, team_missing = transform_schedule(
        df,
        league_std="GER-Bundesliga",
        season_long="2025-2026",
        team_lookup=team_lookup,
        teams_name_to_code=teams_name_to_code,
        teams_name_to_display=teams_name_to_display,
    )

    assert len(team_missing) == 0
    assert set(out["team"].unique()) == {"LEV", "BVB"}


def test_player_id_cleaning_and_player_season_mode_with_name_fallback():
    player_lookup = {
        normalize_player_key("Alex Scott"): "11111111",
        normalize_player_key("John Doe"): "22222222",
    }
    master_fpl = {
        "11111111": {"career": {"2025-26": {"fpl_position": "MID"}}},
        "22222222": {"career": {"2025-26": {"fpl_position": "DEF"}}},
    }

    pm = pd.DataFrame(
        [
            {
                "league": "EPL",
                "season": 2025,
                "player": "Alex Scott",
                "player_id": 999,
                "position": "MC",
                "minutes": 90,
            },
            {
                "league": "EPL",
                "season": 2025,
                "player": "Alex Scott",
                "player_id": 999,
                "position": "AMC",
                "minutes": 30,
            },
            {
                "league": "EPL",
                "season": 2025,
                "player": "Mystery Name",
                "player_id": 555,
                "position": "FW",
                "minutes": 50,
            },
        ]
    )
    pm_clean, unknown_pos, pm_mism = process_player_match(
        pm,
        league_std="ENG-Premier League",
        season_long="2025-2026",
        player_lookup=player_lookup,
        ambiguous_lookup_keys=set(),
        override_lookup={},
        master_fpl=master_fpl,
        teams_name_to_code={},
    )
    assert len(unknown_pos) == 0
    assert len(pm_mism) == 0
    assert "understat_player_id" in pm_clean.columns
    assert pm_clean.loc[pm_clean["player"] == "Alex Scott", "player_id"].eq("11111111").all()
    assert pm_clean.loc[pm_clean["player"] == "Mystery Name", "player_id_missing"].all()

    by_id, by_name = build_fpl_mode_maps(pm_clean)
    assert by_id[("ENG-Premier League", "2025-2026", "11111111")] == "MID"
    assert by_name[("ENG-Premier League", "2025-2026", normalize_player_key("Mystery Name"))] == "FWD"

    ps = pd.DataFrame(
        [
            {"league": "EPL", "season": 2025, "player": "Alex Scott", "player_id": 999, "position": "M S"},
            {"league": "EPL", "season": 2025, "player": "Mystery Name", "player_id": 123, "position": "F S"},
        ]
    )
    ps_clean, ps_mism = process_player_season(
        ps,
        league_std="ENG-Premier League",
        season_long="2025-2026",
        player_lookup=player_lookup,
        ambiguous_lookup_keys=set(),
        override_lookup={},
        master_fpl=master_fpl,
        mode_by_id=by_id,
        mode_by_name=by_name,
        teams_name_to_code={},
    )
    assert len(ps_mism) == 0
    alex = ps_clean.loc[ps_clean["player"] == "Alex Scott"].iloc[0]
    mystery = ps_clean.loc[ps_clean["player"] == "Mystery Name"].iloc[0]
    assert alex["player_id"] == "11111111"
    assert alex["fpl_pos"] == "MID"
    assert mystery["fpl_pos"] == "FWD"
