# scripts/common/cli.py
from __future__ import annotations
import argparse, sys, os, logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Any
import json

try:
    import tomllib  # py >=3.11
except Exception:
    import tomli as tomllib  # py311-

LOG_FMT = "%(levelname)s | %(asctime)s | %(name)s | %(message)s"
DATE_FMT = "%H:%M:%S"

@dataclass
class UX:
    app: str
    logger: logging.Logger

def add_standard_io_flags(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out-format", choices=["csv","parquet","both"], default="both")
    ap.add_argument("--dry-run", action="store_true", help="Execute all steps but skip writes")
    ap.add_argument("--confirm", action="store_true", help="Required to overwrite existing outputs")
    ap.add_argument("--log-level", default="INFO")

def add_asof_window_flags(ap: argparse.ArgumentParser, require_as_of_gw: bool = True) -> None:
    ap.add_argument("--as-of", default="now", help='ISO ts or "now"')
    ap.add_argument("--as-of-tz", default="Africa/Lagos")
    if require_as_of_gw:
        ap.add_argument("--as-of-gw", type=int, required=True)
    ap.add_argument("--n-future", type=int, default=3)
    ap.add_argument("--gw-from", type=int, default=None)
    ap.add_argument("--gw-to", type=int, default=None)
    ap.add_argument("--strict-n-future", action="store_true")

def add_profile_flag(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--profile", type=Path, help="TOML file with default flags")

def load_profile_args(profile: Optional[Path]) -> Dict[str, Any]:
    if not profile: return {}
    if not profile.exists():
        raise FileNotFoundError(f"--profile not found: {profile}")
    with profile.open("rb") as f:
        cfg = tomllib.load(f)
    # flat dict of defaults; later explicit CLI flags still win
    return {k.replace("-", "_"): v for k,v in (cfg.get("defaults") or {}).items()}

def init_logger(name: str, level: str) -> logging.Logger:
    logging.basicConfig(level=level.upper(), format=LOG_FMT, datefmt=DATE_FMT)
    lg = logging.getLogger(name)
    for noisy in ("lightgbm","urllib3","botocore"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLogger(noisy).level))
    return lg

def require_confirm(confirm: bool, paths: Iterable[Path]) -> None:
    to_over = [p for p in paths if p.exists()]
    if to_over and not confirm:
        raise RuntimeError(f"{len(to_over)} output(s) exist and would be overwritten. "
                           f"Re-run with --confirm or remove:\n" + "\n".join(map(str,to_over)))

def dump_run_meta(path: Path, args: argparse.Namespace, extras: Dict[str, Any] | None = None) -> None:
    meta = {"args": vars(args), "extras": extras or {}}
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
