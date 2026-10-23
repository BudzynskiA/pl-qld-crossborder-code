"""
Event-level harmonization of the outcome Y and of injuries (PL SEWIK + QLD TMR).

  morti_count   PL `ilosc_zmarlych`; QLD `morti_count` / `Count_Casualty_Fatality`
  feriti_count  PL `ilosc_feriti`; QLD `feriti_count` (derived)
  y_morti       1 when morti_count > 0
"""
from __future__ import annotations

import polars as pl

Y_MORTI = "y_morti"
MORTI_COUNT = "morti_count"
FERITI_COUNT = "feriti_count"

PL_COUNT_COL = "ilosc_zmarlych"
PL_FERITI_COL = "ilosc_feriti"
QLD_MORTI_COL = "morti_count"
QLD_FERITI_COL = "feriti_count"


def pl_morti_exprs(
    count_col: str = PL_COUNT_COL,
    feriti_col: str = PL_FERITI_COL,
) -> list[pl.Expr]:
    count = pl.col(count_col).cast(pl.Int64, strict=False).fill_null(0)
    feriti = pl.col(feriti_col).cast(pl.Int64, strict=False).fill_null(0)
    return [
        count.alias(MORTI_COUNT),
        feriti.alias(FERITI_COUNT),
        (count > 0).cast(pl.UInt8).alias(Y_MORTI),
    ]


def qld_morti_exprs(
    morti_col: str = QLD_MORTI_COL,
    feriti_col: str = QLD_FERITI_COL,
) -> list[pl.Expr]:
    count = pl.col(morti_col).cast(pl.Int64, strict=False).fill_null(0)
    feriti = pl.col(feriti_col).cast(pl.Int64, strict=False).fill_null(0)
    return [
        count.alias(MORTI_COUNT),
        feriti.alias(FERITI_COUNT),
        (count > 0).cast(pl.UInt8).alias(Y_MORTI),
    ]


def assert_outcome_morti(df: pl.DataFrame, label: str) -> None:
    for c in (MORTI_COUNT, FERITI_COUNT, Y_MORTI):
        if c not in df.columns:
            raise ValueError(f"{label}: missing column {c}")

    bad_y = df.filter(~pl.col(Y_MORTI).is_in([0, 1])).height
    if bad_y:
        raise ValueError(f"{label}: {Y_MORTI} outside {{0,1}}: {bad_y} rows")

    bad_neg = df.filter((pl.col(MORTI_COUNT) < 0) | (pl.col(FERITI_COUNT) < 0)).height
    if bad_neg:
        raise ValueError(f"{label}: negative counts: {bad_neg} rows")

    mismatch = df.filter((pl.col(Y_MORTI) == 1) != (pl.col(MORTI_COUNT) > 0)).height
    if mismatch:
        raise ValueError(f"{label}: inconsistent {Y_MORTI} vs {MORTI_COUNT}: {mismatch} rows")
