# scripts/utils/validate.py
from __future__ import annotations
import re
from typing import Dict, Any, Iterable, List, Optional
import numpy as np
import pandas as pd

_INT_TYPES   = ("Int8", "Int64")     # pandas nullable Int64 accepted
_FLOAT_TYPES = ("float",)
_STR_TYPES   = ("str","string","object")
_BOOL_TYPES  = ("bool",)
_DTYPES = set(_INT_TYPES + _FLOAT_TYPES + _STR_TYPES + _BOOL_TYPES + ("datetime64[ns]",))

def _is_dtype(df: pd.DataFrame, col: str, want: str) -> bool:
    s = df[col]
    if want in ("Int8","Int64"):
        return pd.api.types.is_integer_dtype(s.dtype)
    if want in _FLOAT_TYPES:
        return pd.api.types.is_float_dtype(s.dtype)
    if want in _STR_TYPES:
        return (pd.api.types.is_string_dtype(s.dtype) or s.dtype == "object")
    if want in _BOOL_TYPES:
        return pd.api.types.is_bool_dtype(s.dtype)
    if want == "datetime64[ns]":
        return pd.api.types.is_datetime64_ns_dtype(s.dtype)
    return False

def _fmt_list(vals: Iterable[Any], max_items=8) -> str:
    vals = list(vals)
    if len(vals) <= max_items: return ", ".join(map(str, vals))
    return ", ".join(map(str, vals[:max_items])) + f", … (+{len(vals)-max_items} more)"

def _preview_rows(df: pd.DataFrame,
                  mask: pd.Series,
                  show_cols: List[str],
                  max_rows: int) -> str:
    """Return a compact, single-line-per-row preview: idx + selected column values."""
    try:
        bad = df.loc[mask, show_cols].copy()
    except Exception:
        # Fallback if any column missing; show whatever exists
        existing = [c for c in show_cols if c in df.columns]
        bad = df.loc[mask, existing].copy() if existing else df.loc[mask].copy()

    if bad.empty:
        return ""
    if len(bad) > max_rows:
        bad = bad.iloc[:max_rows]

    parts = []
    for idx, row in bad.iterrows():
        kv = ", ".join(f"{c}={row[c]!r}" for c in bad.columns)
        parts.append(f"[idx {idx}] {kv}")
    extra = ""
    if mask.sum() > max_rows:
        extra = f" (+{int(mask.sum())-max_rows} more)"
    return "  → Offending rows:\n    " + "\n    ".join(parts) + extra

def validate_df(df: pd.DataFrame,
                schema: Dict[str, Any],
                name: str = "output",
                *,
                show_rows: bool = True,
                max_rows: int = 12) -> None:
    """
    schema = {
      "required": ["colA","colB",...],
      "dtypes":   {"colA":"Int64","colB":"float","colC":"datetime64[ns]", ...},
      "na":       {"colA": False, "colB": True, ...},  # False => no NA allowed
      "ranges":   {"colA":{"min":0,"max":10}, "p":{"min":0.0,"max":1.0}},
      "choices":  {"pos":{"in":["GK","DEF","MID","FWD"]}},
      "unique":   ["season","gw_orig","team_id","player_id"],
      "date_rules": {"normalize":["date_sched"]},  # must be date-only (no time-of-day)
      "logic":    [("venue_bin in {0,1}", ["venue_bin"])],
      "allow_extra": True,                         # optional, default True
      "id_cols": ["season","gw_orig","team_id","player_id"]  # shown in row previews
    }
    Raises ValueError with a compact diff if any check fails.
    """
    errors = []

    # Decide which columns to show in per-row previews
    id_cols: List[str] = list(schema.get("id_cols", []))
    # always include index-mapping columns if present
    for c in ["season","gw_orig","gw","team_id","player_id","date_played","date_sched"]:
        if c in df.columns and c not in id_cols:
            id_cols.append(c)

    def _err_with_rows(msg: str,
                       mask: Optional[pd.Series] = None,
                       extra_cols: Optional[List[str]] = None) -> str:
        if not show_rows or mask is None or mask.sum() == 0:
            return msg
        show = id_cols.copy()
        if extra_cols:
            for c in extra_cols:
                if c not in show:
                    show.append(c)
        msg_rows = _preview_rows(df, mask, show_cols=show, max_rows=max_rows)
        return msg + ("\n" + msg_rows if msg_rows else "")

    # 1) required columns
    req = set(schema.get("required", []))
    missing = [c for c in req if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {_fmt_list(missing)}")

    # 2) unexpected columns (optional)
    allow_extra = bool(schema.get("allow_extra", True))
    if not allow_extra:
        extra = [c for c in df.columns if c not in req]
        if extra:
            errors.append(f"Unexpected columns present: {_fmt_list(extra)}")

    # 3) dtypes
    dmap = schema.get("dtypes", {})
    for col, want in dmap.items():
        if col not in df.columns:
            continue
        if want not in _DTYPES:
            errors.append(f"dtypes[{col}] has unknown target '{want}'. Allowed: {_fmt_list(sorted(_DTYPES))}")
            continue
        if not _is_dtype(df, col, want):
            # dtype mismatch is column-wide; rows aren't meaningful here
            errors.append(f"dtype mismatch for '{col}': got {df[col].dtype}, expected {want}")

    # 4) NA policy
    na = schema.get("na", {})
    for col, no_na in na.items():
        if col in df.columns and no_na is False:
            mask = df[col].isna()
            if mask.any():
                msg = f"NA not allowed in '{col}' — {int(mask.sum())} NA(s) found"
                errors.append(_err_with_rows(msg, mask=mask, extra_cols=[col]))

    # 5) ranges
    ranges = schema.get("ranges", {})
    for col, rr in ranges.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if "min" in rr:
            mask = (s < rr["min"])
            if mask.fillna(False).any():
                n = int(mask.fillna(False).sum())
                msg = f"Range violation: '{col}' < {rr['min']} for {n} row(s)"
                errors.append(_err_with_rows(msg, mask=mask.fillna(False), extra_cols=[col]))
        if "max" in rr:
            mask = (s > rr["max"])
            if mask.fillna(False).any():
                n = int(mask.fillna(False).sum())
                msg = f"Range violation: '{col}' > {rr['max']} for {n} row(s)"
                errors.append(_err_with_rows(msg, mask=mask.fillna(False), extra_cols=[col]))

    # 6) choices
    choices = schema.get("choices", {})
    for col, spec in choices.items():
        if col not in df.columns:
            continue
        allowed = set(spec.get("in", []))
        mask = ~df[col].isin(allowed)
        if mask.dropna().any():
            bad_vals = df.loc[mask.fillna(False), col].dropna().unique()
            base = f"Illegal values for '{col}': {_fmt_list(bad_vals)}; allowed: {_fmt_list(sorted(allowed))}"
            errors.append(_err_with_rows(base, mask=mask.fillna(False), extra_cols=[col]))

    # 7) date rules (normalize => date-only)
    dr = schema.get("date_rules", {})
    for col in dr.get("normalize", []):
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            mask = ~(df[col].dt.normalize().equals(df[col]))
            # .equals returns scalar; build row mask:
            row_mask = (df[col].dt.normalize() != df[col])
            if row_mask.any():
                msg = f"'{col}' must be date-only (normalized); found time components in {int(row_mask.sum())} row(s)"
                errors.append(_err_with_rows(msg, mask=row_mask, extra_cols=[col]))

    # 8) simple logic guardrails
    logic = schema.get("logic", [])
    for expr, cols in logic:
        # support "c in {0,1,2}" checks on ints
        m = re.match(r"^([a-zA-Z0-9_]+)\s+in\s+\{([0-9,\s]+)\}$", expr.strip())
        if m:
            c = m.group(1); allowed = set(int(x) for x in m.group(2).split(","))
            if c in df.columns:
                mask = ~df[c].isin(allowed)
                if mask.dropna().any():
                    base = f"Logic: {c} must be in {sorted(allowed)}; got {_fmt_list(df.loc[mask.fillna(False), c].dropna().unique())}"
                    errors.append(_err_with_rows(base, mask=mask.fillna(False), extra_cols=[c]))
        # (extend here if you add more logic patterns)

    # 9) uniqueness
    uniq = schema.get("unique")
    if uniq:
        dup_mask = df.duplicated(uniq, keep=False)
        if dup_mask.any():
            dups = int(df.duplicated(uniq).sum())
            # Build a compact preview grouped by the duplicate key(s)
            dupe_keys = (
                df.loc[dup_mask, uniq]
                  .astype(str)
                  .agg("|".join, axis=1)
                  .value_counts()
            )
            top_keys = dupe_keys.head(max_rows)
            key_lines = [f"{k} ×{v}" for k, v in top_keys.items()]
            extra = ""
            if len(dupe_keys) > max_rows:
                extra = f" (+{len(dupe_keys)-max_rows} more key groups)"
            base = f"Unique key violation on {uniq}: {dups} duplicate row(s)\n  → Duplicate key groups:\n    " + "\n    ".join(key_lines) + extra
            # Also show sample rows for context
            errors.append(_err_with_rows(base, mask=dup_mask, extra_cols=list(set(uniq + (schema.get('id_cols') or [])))))

    if errors:
        raise ValueError(
            f"[Schema validation failed for {name}]\n" + "\n\n".join(errors)
        )
