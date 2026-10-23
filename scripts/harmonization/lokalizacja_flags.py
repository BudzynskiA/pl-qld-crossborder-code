"""
Shared 0/1 location flags (PL SEWIK + QLD TMR), event level.

PL: same definitions as the Poland-Italy study (`obszar`, `rodzaj_drogi`).
QLD (proxy v1):
  lok_autostrada   - `State_Road_Name` contains motorway/freeway OR
                     State-controlled + `Crash_Speed_Limit` = 100-110 km/h
  lok_miejski      - `Loc_ABS_Remoteness` = Major Cities (not motorway)
  lok_poza_miastem - all others (Inner/Outer Regional, Remote, Very Remote, Unknown)
"""
from __future__ import annotations

import polars as pl

PL_OBSZAR_COL = "obszar"
PL_RODZAJ_COL = "rodzaj_drogi"
QLD_REMOTENESS_COL = "Loc_ABS_Remoteness"
QLD_STATE_ROAD_COL = "State_Road_Name"
QLD_AUTH_COL = "Crash_Controlling_Authority"
QLD_SPEED_COL = "Crash_Speed_Limit"

LOKALIZACJA_FLAGS: tuple[str, ...] = (
    "lok_autostrada",
    "lok_miejski",
    "lok_poza_miastem",
)


def _pl_obszar_expr(col: str = PL_OBSZAR_COL) -> pl.Expr:
    return pl.col(col).fill_null("").cast(pl.Utf8).str.strip_chars().str.to_lowercase()


def _pl_rodzaj_expr(col: str = PL_RODZAJ_COL) -> pl.Expr:
    return pl.col(col).fill_null("").cast(pl.Utf8).str.strip_chars().str.to_lowercase()


def pl_lokalizacja_flag_exprs(
    obszar_col: str = PL_OBSZAR_COL,
    rodzaj_col: str = PL_RODZAJ_COL,
) -> list[pl.Expr]:
    obszar = _pl_obszar_expr(obszar_col)
    rodzaj = _pl_rodzaj_expr(rodzaj_col)
    autostrada = rodzaj == "autostrada"
    ekspresowa = rodzaj == "ekspresowa"
    miejski = (obszar == "obszar_zabudowany") & ~autostrada & ~ekspresowa
    poza = ekspresowa | ((obszar == "obszar_niezabudowany") & ~autostrada)
    return [
        autostrada.cast(pl.UInt8).alias("lok_autostrada"),
        miejski.cast(pl.UInt8).alias("lok_miejski"),
        poza.cast(pl.UInt8).alias("lok_poza_miastem"),
    ]


def qld_lokalizacja_flag_exprs(
    remoteness_col: str = QLD_REMOTENESS_COL,
    state_road_col: str = QLD_STATE_ROAD_COL,
    auth_col: str = QLD_AUTH_COL,
    speed_col: str = QLD_SPEED_COL,
) -> list[pl.Expr]:
    srn = pl.col(state_road_col).fill_null("").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    motorway_name = srn.str.contains("motorway") | srn.str.contains("freeway")
    speed = pl.col(speed_col).fill_null("").cast(pl.Utf8).str.strip_chars()
    auth = pl.col(auth_col).fill_null("").cast(pl.Utf8).str.strip_chars()
    rem = pl.col(remoteness_col).fill_null("").cast(pl.Utf8).str.strip_chars()

    autostrada = motorway_name | ((auth == "State-controlled") & (speed == "100 - 110 km/h"))
    miejski = (rem == "Major Cities") & ~autostrada
    poza = ~autostrada & ~miejski

    return [
        autostrada.cast(pl.UInt8).alias("lok_autostrada"),
        miejski.cast(pl.UInt8).alias("lok_miejski"),
        poza.cast(pl.UInt8).alias("lok_poza_miastem"),
    ]


def assert_lokalizacja_flags(df: pl.DataFrame, label: str, *, allow_unassigned: bool = False) -> None:
    missing = [c for c in LOKALIZACJA_FLAGS if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")

    for c in LOKALIZACJA_FLAGS:
        bad = df.filter(~pl.col(c).is_in([0, 1])).height
        if bad:
            raise ValueError(f"{label}: {c} has values outside {{0,1}}: {bad} rows")

    n_flags = pl.sum_horizontal([pl.col(c) for c in LOKALIZACJA_FLAGS])
    if allow_unassigned:
        bad = df.filter(~n_flags.is_in([0, 1])).height
        msg = "sum of lok_* outside {0,1}"
    else:
        bad = df.filter(n_flags != 1).height
        msg = "sum of lok_* != 1 (exactly 1 expected)"
    if bad:
        raise ValueError(f"{label}: {bad} rows - {msg}")
