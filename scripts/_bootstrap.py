"""
Common import path and data locations for `scripts/`.

Data are not part of this repository. Point the scripts to local copies with
environment variables (defaults in brackets, relative to the repository root):

    PLQLD_POLAND_DIR      national SEWIK data (merged tables, derived event level) [data/poland]
    PLQLD_QUEENSLAND_DIR  Queensland TMR data (raw CSV, derived event level)        [data/queensland]
    PLQLD_DATA_DIR        harmonized panels (outputs)                               [data/harmonized]

Usage:

    from _bootstrap import install, PL_QLD_ROOT, POLAND, QUEENSLAND, TABLES_MD, TABLES_CSV
    PL_QLD_ROOT = install()
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parent
PL_QLD_ROOT = SCRIPTS_ROOT.parent


def _env_dir(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


POLAND = _env_dir("PLQLD_POLAND_DIR", PL_QLD_ROOT / "data" / "poland")
QUEENSLAND = _env_dir("PLQLD_QUEENSLAND_DIR", PL_QLD_ROOT / "data" / "queensland")
TABLES_MD = PL_QLD_ROOT / "tables_md"
TABLES_CSV = PL_QLD_ROOT / "tables_csv"
FIGURES = PL_QLD_ROOT / "figures"
HARMONIZED_EVENT_LEVEL_DIR = _env_dir("PLQLD_DATA_DIR", PL_QLD_ROOT / "data" / "harmonized") / "event_level"


def display_path(path: Path) -> str:
    """Path relative to the repository root when inside it, else the absolute path."""
    try:
        return path.resolve().relative_to(PL_QLD_ROOT).as_posix()
    except ValueError:
        return str(path)


def install() -> Path:
    """Add `scripts/` to sys.path; return the repository root."""
    root = str(SCRIPTS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    for out_dir in (TABLES_MD, TABLES_CSV, FIGURES):
        out_dir.mkdir(parents=True, exist_ok=True)
    return PL_QLD_ROOT
