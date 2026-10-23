"""
Builds `<PLQLD_QUEENSLAND_DIR>/data/derived/event_level/event_level_qld_2015_2024.parquet`.

Source: Queensland TMR "Crash data from Queensland roads: Road crash locations"
(`_1_crash_locations.csv`, one row per crash, key `Crash_Ref_Number`), available at
https://www.data.qld.gov.au/dataset/crash-data-from-queensland-roads

Filters (same window as the Polish data):
  - `crash_year` in [2015, 2024]
  - exclude `Crash_Severity == "Property damage only"`

Derived columns (cross-border):
  - `morti_count` <- `Count_Casualty_Fatality`
  - `feriti_count` <- hospitalised + medically treated + minor injury
  - `y_morti` <- `morti_count > 0`
  - `country` = `AU_QLD`

Run from the repository root:

  python scripts/national/queensland/build_derived_event_level_qld_2015_2024.py

Default input: `<PLQLD_QUEENSLAND_DIR>/data/raw/tmr/_1_crash_locations.csv`.
Set `QLD_TMR_RAW_CSV` to use another CSV file.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))
from _bootstrap import QUEENSLAND, display_path, install  # noqa: E402

install()

QUEENSLAND_ROOT = QUEENSLAND
RAW_TMR = QUEENSLAND / "data" / "raw" / "tmr"
DERIVED_EVENT_LEVEL = QUEENSLAND / "data" / "derived" / "event_level"

import polars as pl  # noqa: E402

YEAR_FROM = 2015
YEAR_TO = 2024
OUT_PATH = DERIVED_EVENT_LEVEL / "event_level_qld_2015_2024.parquet"
META_PATH = DERIVED_EVENT_LEVEL / "build_meta.json"

DEFAULT_RAW = RAW_TMR / "_1_crash_locations.csv"
PD_ONLY = "Property damage only"

_COL_YEAR = "Crash_Year"
_COL_SEVERITY = "Crash_Severity"
_COL_ID = "Crash_Ref_Number"
_COL_FATAL = "Count_Casualty_Fatality"
_COL_HOSP = "Count_Casualty_Hospitalised"
_COL_MED = "Count_Casualty_MedicallyTreated"
_COL_MINOR = "Count_Casualty_MinorInjury"


def _raw_csv_path() -> Path:
    env = os.environ.get("QLD_TMR_RAW_CSV", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_RAW


def _parse_year() -> pl.Expr:
    return pl.col(_COL_YEAR).cast(pl.Int64, strict=False)


def _injured_expr() -> pl.Expr:
    parts = [
        pl.col(c).cast(pl.Int64, strict=False).fill_null(0)
        for c in (_COL_HOSP, _COL_MED, _COL_MINOR)
        if c
    ]
    if not parts:
        return pl.lit(0).alias("feriti_count")
    return pl.sum_horizontal(parts).alias("feriti_count")


def main() -> int:
    raw = _raw_csv_path()
    if not raw.is_file():
        print(f"[err] CSV file not found: {raw}", file=sys.stderr)
        print("  Download it from data.qld.gov.au to <PLQLD_QUEENSLAND_DIR>/data/raw/tmr/_1_crash_locations.csv", file=sys.stderr)
        print("  or set QLD_TMR_RAW_CSV", file=sys.stderr)
        return 1

    lf = pl.scan_csv(
        raw,
        infer_schema_length=100_000,
        null_values=["UNK", ""],
    )
    schema = lf.collect_schema().names()
    for c in (_COL_ID, _COL_YEAR, _COL_SEVERITY, _COL_FATAL):
        if c not in schema:
            print(f"[err] Missing column {c!r} in {raw.name}", file=sys.stderr)
            return 1

    n_raw = lf.select(pl.len()).collect().item()

    lf = lf.with_columns(
        _parse_year().alias("crash_year"),
        pl.col(_COL_FATAL).cast(pl.Int64, strict=False).fill_null(0).alias("morti_count"),
        _injured_expr(),
        pl.lit("AU_QLD").alias("country"),
    ).with_columns((pl.col("morti_count") > 0).alias("y_morti"))

    lf = lf.filter(
        (pl.col("crash_year") >= YEAR_FROM) & (pl.col("crash_year") <= YEAR_TO)
    )
    if _COL_SEVERITY in schema:
        lf = lf.filter(pl.col(_COL_SEVERITY).cast(pl.Utf8) != PD_ONLY)

    out = lf.collect()
    DERIVED_EVENT_LEVEL.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_PATH, compression="snappy")

    n_fatal = int(out.filter(pl.col("y_morti")).height)
    prev = n_fatal / out.height if out.height else 0.0
    n_injured_rows = int(out.filter(pl.col("feriti_count") > 0).height)

    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(raw.relative_to(QUEENSLAND_ROOT)) if raw.is_relative_to(QUEENSLAND_ROOT) else str(raw),
        "year_range": [YEAR_FROM, YEAR_TO],
        "exclude_severity": PD_ONLY,
        "counts": {
            "rows_raw_csv": n_raw,
            "rows_after_filters": out.height,
            "rows_y_morti": n_fatal,
            "prevalence_y_morti": round(prev, 6),
            "rows_feriti_count_gt_0": n_injured_rows,
        },
        "output": str(OUT_PATH.relative_to(QUEENSLAND_ROOT)),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"QLD event_level {YEAR_FROM}-{YEAR_TO}:")
    print(f"  raw CSV rows:     {n_raw:,}")
    print(f"  after filters:    {out.height:,}")
    print(f"  y_morti=1:        {n_fatal:,} ({100 * prev:.2f}%)")
    print(f"  feriti_count>0:   {n_injured_rows:,}")
    print(f"  -> {display_path(OUT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
