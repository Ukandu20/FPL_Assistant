# -*- coding: utf-8 -*-
"""
Golden regression tests.

Create a tiny deterministic fixture + plan (no randomness),
store under goldens/<case>/expected_plan_gwX.json.
The test compares current output with the golden (tolerant on floats).
"""
import json, os, math
from pathlib import Path

import numpy as np
import pytest

GOLDENS_DIR = Path(os.getenv("GOLDENS_DIR", "tests/goldens"))

def _load_json(fp: Path):
    return json.loads(fp.read_text(encoding="utf-8"))

def _norm(obj):
    """Normalize plan JSON for stable diffs: sort lists by id, round floats."""
    if isinstance(obj, dict):
        return {k:_norm(v) for k,v in sorted(obj.items())}
    if isinstance(obj, list):
        # stable sort by player id if available, else repr
        def key(x):
            if isinstance(x, dict) and ("id" in x or "player_id" in x):
                return str(x.get("id") or x.get("player_id"))
            return repr(x)
        return [_norm(v) for v in sorted(obj, key=key)]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj

def _eq(a, b):
    return _norm(a) == _norm(b)

@pytest.mark.skipif(not GOLDENS_DIR.exists(), reason="no goldens folder")
def test_goldens():
    for case_dir in GOLDENS_DIR.glob("*"):
        if not case_dir.is_dir(): continue
        exp = case_dir / "expected_plan.json"
        cur = case_dir / "current_plan.json"
        if not exp.exists() or not cur.exists():
            continue
        exp_j = _load_json(exp)
        cur_j = _load_json(cur)
        assert _eq(cur_j, exp_j), f"Regression in {case_dir.name}: current_plan.json != expected_plan.json"
