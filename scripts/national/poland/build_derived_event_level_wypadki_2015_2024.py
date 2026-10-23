"""
Builds `<PLQLD_POLAND_DIR>/data/derived/event_level/zdarzenia_wypadki_2015_2024.parquet`.

Subset of SEWIK events with at least one injured or killed participant. Excluded:
events in which every participant has `poziom_obrazen` == nieranny (property damage
only collisions in the SEWIK sense).

Inputs (merged SEWIK tables, not part of this repository):
  - `<PLQLD_POLAND_DIR>/data/merged/2015_2025/zdarzenia/zdarzenia_2015_2025.parquet`
  - `<PLQLD_POLAND_DIR>/data/merged/2015_2025/uczestnik/uczestnik_2015_2025.parquet`

Year filter: 2015-2024.

Criterion (after de-duplicating participants within an event):
  at least one participant with `poziom_obrazen` in
  {lekko_ranny, ciezko_ranny, zmarly_na_miejscu, zmarly_w_ciagu_30_dni}.

Output columns: all columns of `zdarzenia` plus participant aggregates:
  ilosc_uczestnikow, ilosc_lekko_rannych, ilosc_ciezko_rannych, ilosc_zmarlych, ilosc_rannych_lacznie

Run from the repository root:
  python scripts/national/poland/build_derived_event_level_wypadki_2015_2024.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))
from _bootstrap import POLAND, display_path, install  # noqa: E402

install()

import polars as pl  # noqa: E402

from sewik_codes import (  # noqa: E402
    COL_POZIOM,
    POZIOM_CIEZKO,
    POZIOM_LEKKO,
    POZIOM_ZM_30,
    POZIOM_ZM_MIEJSCE,
)

ZDARZENIA = POLAND / "data" / "merged" / "2015_2025" / "zdarzenia" / "zdarzenia_2015_2025.parquet"
UCZESTNIK = POLAND / "data" / "merged" / "2015_2025" / "uczestnik" / "uczestnik_2015_2025.parquet"
OUT_DIR = POLAND / "data" / "derived" / "event_level"
OUT_PATH = OUT_DIR / "zdarzenia_wypadki_2015_2024.parquet"

COL_EVT = "ID zdarzenia"
COL_UCZ = "ID uczestnika"
COL_ROK = "rok"

YEAR_FROM = 2015
YEAR_TO = 2024

INJURED_OR_DEAD = (POZIOM_LEKKO, POZIOM_CIEZKO, POZIOM_ZM_MIEJSCE, POZIOM_ZM_30)
DEAD_POZIOM = (POZIOM_ZM_MIEJSCE, POZIOM_ZM_30)


def _deduped_uczestnik_lf() -> pl.LazyFrame:
    lf = pl.scan_parquet(UCZESTNIK)
    schema = lf.collect_schema()
    for c in (COL_EVT, COL_UCZ, COL_POZIOM):
        if c not in schema.names():
            raise SystemExit(f"Missing column {c!r} in {UCZESTNIK}")

    sort_cols = [COL_EVT, COL_UCZ]
    if COL_ROK in schema.names():
        sort_cols.append(COL_ROK)

    po = pl.col(COL_POZIOM).cast(pl.String).str.strip_chars()
    return (
        lf.with_columns(
            pl.col(COL_EVT).cast(pl.String).str.strip_chars(),
            pl.col(COL_UCZ).cast(pl.String).str.strip_chars(),
            po.alias(COL_POZIOM),
        )
        .sort(sort_cols)
        .unique(subset=[COL_EVT, COL_UCZ], keep="first")
    )


def _event_agg(lf: pl.LazyFrame) -> pl.LazyFrame:
    po = pl.col(COL_POZIOM)
    return lf.group_by(COL_EVT).agg(
        pl.col(COL_UCZ).n_unique().alias("ilosc_uczestnikow"),
        pl.col(COL_UCZ).filter(po == pl.lit(POZIOM_LEKKO)).n_unique().alias("ilosc_lekko_rannych"),
        pl.col(COL_UCZ).filter(po == pl.lit(POZIOM_CIEZKO)).n_unique().alias("ilosc_ciezko_rannych"),
        pl.col(COL_UCZ).filter(po.is_in(list(DEAD_POZIOM))).n_unique().alias("ilosc_zmarlych"),
        pl.col(COL_UCZ).filter(po.is_in(list(INJURED_OR_DEAD))).n_unique().alias("ilosc_rannych_lacznie"),
    )


def _injury_event_ids(lf: pl.LazyFrame) -> pl.LazyFrame:
    po = pl.col(COL_POZIOM)
    return (
        lf.filter(po.is_in(list(INJURED_OR_DEAD)))
        .select(COL_EVT)
        .unique()
    )


def main() -> None:
    if not ZDARZENIA.is_file():
        raise SystemExit(f"Missing file: {ZDARZENIA}")
    if not UCZESTNIK.is_file():
        raise SystemExit(f"Missing file: {UCZESTNIK}")

    ucz = _deduped_uczestnik_lf()
    injury_ids = _injury_event_ids(ucz)
    agg = _event_agg(ucz)

    zdarzenia = pl.scan_parquet(ZDARZENIA).with_columns(
        pl.col(COL_EVT).cast(pl.String).str.strip_chars()
    )
    z_schema = zdarzenia.collect_schema()
    if COL_ROK not in z_schema.names():
        raise SystemExit(f"Missing column {COL_ROK!r} in {ZDARZENIA}; the merged table needs a year column.")

    in_range = zdarzenia.filter(pl.col(COL_ROK).is_between(YEAR_FROM, YEAR_TO))
    n_all = in_range.select(pl.len()).collect().item()

    out = (
        in_range.join(injury_ids, on=COL_EVT, how="inner")
        .join(agg, on=COL_EVT, how="left")
        .with_columns(
            pl.col("ilosc_uczestnikow").fill_null(0).cast(pl.UInt32),
            pl.col("ilosc_lekko_rannych").fill_null(0).cast(pl.UInt32),
            pl.col("ilosc_ciezko_rannych").fill_null(0).cast(pl.UInt32),
            pl.col("ilosc_zmarlych").fill_null(0).cast(pl.UInt32),
            pl.col("ilosc_rannych_lacznie").fill_null(0).cast(pl.UInt32),
        )
        .collect()
    )

    if out.height == 0:
        raise SystemExit("Empty result: check the merged tables and the poziom_obrazen criterion.")

    bad = out.filter(pl.col("ilosc_rannych_lacznie") == 0).height
    if bad:
        raise SystemExit(f"Consistency check failed: {bad} rows without injured or killed participants after filtering.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_PATH, compression="snappy")

    print(
        f"Events {YEAR_FROM}-{YEAR_TO}: {n_all:,} in merged -> "
        f"{out.height:,} injury crashes (at least one injured or killed) -> {display_path(OUT_PATH)}"
    )
    print(f"Excluded (no injured participant): {n_all - out.height:,}")


if __name__ == "__main__":
    main()
