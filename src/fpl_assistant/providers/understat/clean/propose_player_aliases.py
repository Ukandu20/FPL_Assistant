#!/usr/bin/env python3
"""
Generate conservative alias-match proposals for unresolved Understat player IDs.

This script is intentionally non-mutating for registry files:
- Does NOT write to overrides.json
- Does NOT write to _id_lookup_players.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from html import unescape
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from scripts.understat_pipeline.clean.clean_understat_raw import build_player_lookup, normalize_player_key
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.understat_pipeline.clean.clean_understat_raw import build_player_lookup, normalize_player_key

LOG = logging.getLogger("understat_alias_proposal")


@dataclass
class Proposal:
    rule_name: str
    proposed_lookup_key: str
    proposed_player_id: str
    confidence: str
    block_reason: str
    evidence: str


def init_logger(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_alias_config(path: Path) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    if not path.exists():
        return {}, {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    full = {
        normalize_player_key(k): normalize_player_key(v)
        for k, v in (raw.get("fullname_alias_map") or {}).items()
        if normalize_player_key(k) and normalize_player_key(v)
    }
    first = {}
    for k, vals in (raw.get("firstname_alias_map") or {}).items():
        nk = normalize_player_key(k)
        nvals = [normalize_player_key(v) for v in (vals or []) if normalize_player_key(v)]
        if nk and nvals:
            # stable unique order
            seen = set()
            out = []
            for v in nvals:
                if v not in seen:
                    seen.add(v)
                    out.append(v)
            first[nk] = out
    return full, first


def build_lookup_bucket(player_lookup_raw: Dict[str, str]) -> Dict[str, set[str]]:
    bucket: Dict[str, set[str]] = {}
    for raw_name, pid in player_lookup_raw.items():
        key = normalize_player_key(raw_name)
        if not key:
            continue
        bucket.setdefault(key, set()).add(str(pid).strip().lower())
    return bucket


def _expand_hyphen_variants(raw_player_name: str) -> List[str]:
    raw = unescape(str(raw_player_name)).replace("\u2019", "'").replace("`", "'").strip().lower()
    if "-" not in raw:
        return []
    toks = [t for t in raw.split() if t]
    option_sets: List[List[str]] = []
    for tok in toks:
        if "-" not in tok:
            option_sets.append([tok])
            continue
        parts = [p for p in tok.split("-") if p]
        opts = [tok] + parts
        # unique preserve
        seen = set()
        uniq = []
        for x in opts:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        option_sets.append(uniq)

    out = set()
    for seq in product(*option_sets):
        out.add(normalize_player_key(" ".join(seq)))
    return sorted([x for x in out if x])


def _evaluate_candidate_keys(
    *,
    candidate_keys: Sequence[str],
    rule_name: str,
    evidence: str,
    lookup_bucket: Dict[str, set[str]],
    ambiguous_lookup_keys: set[str],
) -> Optional[Proposal]:
    keys = sorted({normalize_player_key(k) for k in candidate_keys if normalize_player_key(k)})
    if not keys:
        return None

    hit_keys = [k for k in keys if k in lookup_bucket]
    if not hit_keys:
        return None

    ambiguous_hits = sorted([k for k in hit_keys if k in ambiguous_lookup_keys])
    if ambiguous_hits:
        ids = sorted({pid for k in ambiguous_hits for pid in lookup_bucket.get(k, set())})
        return Proposal(
            rule_name=rule_name,
            proposed_lookup_key="; ".join(ambiguous_hits),
            proposed_player_id="; ".join(ids),
            confidence="blocked",
            block_reason="ambiguous_lookup_key_collision",
            evidence=evidence,
        )

    unique_hits = [(k, next(iter(lookup_bucket[k]))) for k in hit_keys if len(lookup_bucket[k]) == 1]
    unique_ids = sorted({pid for _, pid in unique_hits})
    if len(unique_ids) == 1 and unique_hits:
        # deterministic best key
        best_key = sorted([k for k, pid in unique_hits if pid == unique_ids[0]])[0]
        return Proposal(
            rule_name=rule_name,
            proposed_lookup_key=best_key,
            proposed_player_id=unique_ids[0],
            confidence="safe",
            block_reason="",
            evidence=evidence,
        )

    if len(unique_ids) > 1:
        return Proposal(
            rule_name=rule_name,
            proposed_lookup_key="; ".join(sorted(k for k, _ in unique_hits)),
            proposed_player_id="; ".join(unique_ids),
            confidence="needs_review",
            block_reason="multiple_candidate_ids",
            evidence=evidence,
        )

    return None


def propose_for_player(
    *,
    raw_player: str,
    fullname_alias_map: Dict[str, str],
    firstname_alias_map: Dict[str, List[str]],
    lookup_bucket: Dict[str, set[str]],
    unique_lookup: Dict[str, str],
    ambiguous_lookup_keys: set[str],
) -> Proposal:
    norm_player = normalize_player_key(raw_player)
    toks = norm_player.split()

    # hard block: mononyms are manual-only unless explicit fullname alias is provided
    if len(toks) == 1 and norm_player not in fullname_alias_map:
        return Proposal(
            rule_name="rule_block_mononym",
            proposed_lookup_key="",
            proposed_player_id="",
            confidence="blocked",
            block_reason="single_token_manual_only",
            evidence="single-token name without explicit fullname alias",
        )

    # rule 1: explicit fullname alias map
    if norm_player in fullname_alias_map:
        target_key = normalize_player_key(fullname_alias_map[norm_player])
        if not target_key:
            return Proposal(
                rule_name="rule_fullname_alias_map",
                proposed_lookup_key="",
                proposed_player_id="",
                confidence="needs_review",
                block_reason="configured_alias_empty_target",
                evidence=f"alias configured for '{norm_player}' but target is empty",
            )
        res = _evaluate_candidate_keys(
            candidate_keys=[target_key],
            rule_name="rule_fullname_alias_map",
            evidence=f"configured fullname alias: {norm_player} -> {target_key}",
            lookup_bucket=lookup_bucket,
            ambiguous_lookup_keys=ambiguous_lookup_keys,
        )
        if res is not None:
            return res
        return Proposal(
            rule_name="rule_fullname_alias_map",
            proposed_lookup_key=target_key,
            proposed_player_id="",
            confidence="needs_review",
            block_reason="alias_target_not_found_in_lookup",
            evidence=f"configured fullname alias target not present: {target_key}",
        )

    # rule 2: explicit first-name alias with same remainder
    if len(toks) >= 2:
        first = toks[0]
        rest = toks[1:]
        aliases = firstname_alias_map.get(first, [])
        if aliases:
            cand_keys = [normalize_player_key(" ".join([a] + rest)) for a in aliases]
            res = _evaluate_candidate_keys(
                candidate_keys=cand_keys,
                rule_name="rule_firstname_alias_same_last",
                evidence=f"first-name alias candidates from '{first}': {', '.join(aliases)}",
                lookup_bucket=lookup_bucket,
                ambiguous_lookup_keys=ambiguous_lookup_keys,
            )
            if res is not None:
                return res
            return Proposal(
                rule_name="rule_firstname_alias_same_last",
                proposed_lookup_key="; ".join(sorted(set(cand_keys))),
                proposed_player_id="",
                confidence="needs_review",
                block_reason="firstname_alias_candidates_not_found",
                evidence=f"explicit first-name alias tried but no lookup hit for '{raw_player}'",
            )

    # rule 3: first two tokens exact
    if len(toks) >= 3:
        first_two = " ".join(toks[:2])
        res = _evaluate_candidate_keys(
            candidate_keys=[first_two],
            rule_name="rule_first_two_tokens_exact",
            evidence=f"first two tokens from '{norm_player}' => '{first_two}'",
            lookup_bucket=lookup_bucket,
            ambiguous_lookup_keys=ambiguous_lookup_keys,
        )
        if res is not None:
            return res

    # rule 4: hyphen token expansion
    hyphen_variants = _expand_hyphen_variants(raw_player)
    if hyphen_variants:
        res = _evaluate_candidate_keys(
            candidate_keys=hyphen_variants,
            rule_name="rule_hyphen_token_variant",
            evidence="generated from hyphen token expansion",
            lookup_bucket=lookup_bucket,
            ambiguous_lookup_keys=ambiguous_lookup_keys,
        )
        if res is not None:
            return res

    return Proposal(
        rule_name="rule_none",
        proposed_lookup_key="",
        proposed_player_id="",
        confidence="blocked",
        block_reason="no_conservative_rule_matched",
        evidence="no deterministic conservative match found",
    )


def build_review_df(
    *,
    missing_df: pd.DataFrame,
    fullname_alias_map: Dict[str, str],
    firstname_alias_map: Dict[str, List[str]],
    lookup_bucket: Dict[str, set[str]],
    unique_lookup: Dict[str, str],
    ambiguous_lookup_keys: set[str],
) -> pd.DataFrame:
    if missing_df.empty:
        return pd.DataFrame(
            columns=[
                "league",
                "season",
                "file",
                "player",
                "missing_rows",
                "normalized_player",
                "rule_name",
                "proposed_lookup_key",
                "proposed_player_id",
                "confidence",
                "block_reason",
                "evidence",
            ]
        )

    by_player: Dict[str, Proposal] = {}
    for raw_player in sorted(set(missing_df["player"].dropna().astype(str))):
        by_player[raw_player] = propose_for_player(
            raw_player=raw_player,
            fullname_alias_map=fullname_alias_map,
            firstname_alias_map=firstname_alias_map,
            lookup_bucket=lookup_bucket,
            unique_lookup=unique_lookup,
            ambiguous_lookup_keys=ambiguous_lookup_keys,
        )

    out = missing_df.copy()
    out["normalized_player"] = out["player"].apply(normalize_player_key)
    out["rule_name"] = out["player"].map(lambda p: by_player[str(p)].rule_name)
    out["proposed_lookup_key"] = out["player"].map(lambda p: by_player[str(p)].proposed_lookup_key)
    out["proposed_player_id"] = out["player"].map(lambda p: by_player[str(p)].proposed_player_id)
    out["confidence"] = out["player"].map(lambda p: by_player[str(p)].confidence)
    out["block_reason"] = out["player"].map(lambda p: by_player[str(p)].block_reason)
    out["evidence"] = out["player"].map(lambda p: by_player[str(p)].evidence)

    order = {"safe": 0, "needs_review": 1, "blocked": 2}
    out["__confidence_order"] = out["confidence"].map(lambda c: order.get(str(c), 9))
    out = out.sort_values(
        ["__confidence_order", "missing_rows", "player"],
        ascending=[True, False, True],
    ).drop(columns=["__confidence_order"])
    return out.reset_index(drop=True)


def run(
    *,
    missing_audit: Path,
    player_lookup: Path,
    alias_config: Path,
    out_csv: Path,
) -> Dict[str, Any]:
    miss = pd.read_csv(missing_audit)
    req = {"league", "season", "file", "player", "missing_rows"}
    missing_cols = sorted(req - set(miss.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns in {missing_audit}: {missing_cols}")

    player_lookup_raw = json.loads(player_lookup.read_text(encoding="utf-8"))
    unique_lookup, ambiguous_keys = build_player_lookup(player_lookup_raw)
    lookup_bucket = build_lookup_bucket(player_lookup_raw)
    fullname_alias_map, firstname_alias_map = load_alias_config(alias_config)

    review = build_review_df(
        missing_df=miss[list(req)].copy(),
        fullname_alias_map=fullname_alias_map,
        firstname_alias_map=firstname_alias_map,
        lookup_bucket=lookup_bucket,
        unique_lookup=unique_lookup,
        ambiguous_lookup_keys=ambiguous_keys,
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(out_csv, index=False)

    summary = {
        "rows": int(len(review)),
        "safe_rows": int((review["confidence"] == "safe").sum()),
        "needs_review_rows": int((review["confidence"] == "needs_review").sum()),
        "blocked_rows": int((review["confidence"] == "blocked").sum()),
        "safe_unique_players": int(review.loc[review["confidence"] == "safe", "player"].nunique()),
        "needs_review_unique_players": int(review.loc[review["confidence"] == "needs_review", "player"].nunique()),
        "blocked_unique_players": int(review.loc[review["confidence"] == "blocked", "player"].nunique()),
        "output_csv": str(out_csv),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser("Propose conservative alias matches for unresolved Understat players")
    parser.add_argument(
        "--missing-audit",
        type=Path,
        default=Path("data/processed/understat/_audit/player_lookup_missing.csv"),
    )
    parser.add_argument(
        "--player-lookup",
        type=Path,
        default=Path("data/processed/registry/_id_lookup_players.json"),
    )
    parser.add_argument(
        "--alias-config",
        type=Path,
        default=Path("config/player_spelling_aliases.json"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/processed/understat/_audit/player_alias_candidates_review.csv"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    init_logger(verbose=args.verbose)
    summary = run(
        missing_audit=args.missing_audit,
        player_lookup=args.player_lookup,
        alias_config=args.alias_config,
        out_csv=args.out_csv,
    )
    LOG.info("Alias proposal summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
