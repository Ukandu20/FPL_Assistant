import pandas as pd

from scripts.understat_pipeline.clean.clean_understat_raw import build_player_lookup
from scripts.understat_pipeline.clean.propose_player_aliases import (
    build_lookup_bucket,
    build_review_df,
    propose_for_player,
)


def _lookup(raw_map):
    unique, ambiguous = build_player_lookup(raw_map)
    bucket = build_lookup_bucket(raw_map)
    return unique, ambiguous, bucket


def test_firstname_alias_same_last_safe():
    unique, ambiguous, bucket = _lookup({"nayef aguerd": "df2c8c45"})
    p = propose_for_player(
        raw_player="Naif Aguerd",
        fullname_alias_map={},
        firstname_alias_map={"naif": ["nayef"]},
        lookup_bucket=bucket,
        unique_lookup=unique,
        ambiguous_lookup_keys=ambiguous,
    )
    assert p.confidence == "safe"
    assert p.rule_name == "rule_firstname_alias_same_last"
    assert p.proposed_lookup_key == "nayef aguerd"
    assert p.proposed_player_id == "df2c8c45"


def test_fullname_alias_map_safe_for_cherki():
    unique, ambiguous, bucket = _lookup({"rayan cherki": "463a677e"})
    p = propose_for_player(
        raw_player="Mathis Cherki",
        fullname_alias_map={"mathis cherki": "rayan cherki"},
        firstname_alias_map={},
        lookup_bucket=bucket,
        unique_lookup=unique,
        ambiguous_lookup_keys=ambiguous,
    )
    assert p.confidence == "safe"
    assert p.rule_name == "rule_fullname_alias_map"
    assert p.proposed_player_id == "463a677e"


def test_first_two_tokens_exact_safe_for_added_surname():
    unique, ambiguous, bucket = _lookup({"pierre kalulu": "69466c15"})
    p = propose_for_player(
        raw_player="Pierre Kalulu Kyatengwa",
        fullname_alias_map={},
        firstname_alias_map={},
        lookup_bucket=bucket,
        unique_lookup=unique,
        ambiguous_lookup_keys=ambiguous,
    )
    assert p.confidence == "safe"
    assert p.rule_name == "rule_first_two_tokens_exact"
    assert p.proposed_lookup_key == "pierre kalulu"
    assert p.proposed_player_id == "69466c15"


def test_hyphen_variant_safe_when_other_rules_do_not_hit():
    unique, ambiguous, bucket = _lookup({"philippe mateta": "aaaa1111"})
    p = propose_for_player(
        raw_player="Jean-Philippe Mateta",
        fullname_alias_map={},
        firstname_alias_map={},
        lookup_bucket=bucket,
        unique_lookup=unique,
        ambiguous_lookup_keys=ambiguous,
    )
    assert p.confidence == "safe"
    assert p.rule_name == "rule_hyphen_token_variant"
    assert p.proposed_lookup_key == "philippe mateta"
    assert p.proposed_player_id == "aaaa1111"


def test_mononym_blocked_without_explicit_fullname_alias():
    unique, ambiguous, bucket = _lookup({"kepa": "f0730001"})
    p = propose_for_player(
        raw_player="Kepa",
        fullname_alias_map={},
        firstname_alias_map={},
        lookup_bucket=bucket,
        unique_lookup=unique,
        ambiguous_lookup_keys=ambiguous,
    )
    assert p.confidence == "blocked"
    assert p.rule_name == "rule_block_mononym"
    assert p.block_reason == "single_token_manual_only"


def test_ambiguous_collision_blocked_even_with_fullname_alias():
    raw_lookup = {
        "ederson": "3ae7f0de",
        "éderson": "84709e6a",
    }
    unique, ambiguous, bucket = _lookup(raw_lookup)
    p = propose_for_player(
        raw_player="Ederson",
        fullname_alias_map={"ederson": "ederson"},
        firstname_alias_map={},
        lookup_bucket=bucket,
        unique_lookup=unique,
        ambiguous_lookup_keys=ambiguous,
    )
    assert p.confidence == "blocked"
    assert p.rule_name == "rule_fullname_alias_map"
    assert p.block_reason == "ambiguous_lookup_key_collision"


def test_safety_no_safe_for_multi_id_collision():
    raw_lookup = {
        "pablo martinez": "1faa8961",
        "pablo martínez": "0cf5a17b",
    }
    unique, ambiguous, bucket = _lookup(raw_lookup)
    miss = pd.DataFrame(
        [
            {
                "league": "ESP-La Liga",
                "season": "2025-2026",
                "file": "player_match.csv",
                "player": "Pablo Martínez",
                "missing_rows": 23,
            }
        ]
    )
    out = build_review_df(
        missing_df=miss,
        fullname_alias_map={"pablo martinez": "pablo martinez"},
        firstname_alias_map={},
        lookup_bucket=bucket,
        unique_lookup=unique,
        ambiguous_lookup_keys=ambiguous,
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["confidence"] != "safe"
    assert row["block_reason"] == "ambiguous_lookup_key_collision"
