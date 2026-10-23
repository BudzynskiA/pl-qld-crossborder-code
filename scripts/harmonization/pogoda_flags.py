"""
Shared 0/1 weather flags (PL SEWIK + QLD TMR), event level.

PL: same definitions as the Poland-Italy study (`warunki_atmosferyczne`, `;`-separated tokens).
QLD: `Crash_Atmospheric_Condition` (Clear, Raining, Fog, Smoke/Dust).

TMR codes have no snow/hail or wind → `pogoda_snieg_grad`, `pogoda_wiatr` = 0
(expected for Queensland; the leakage audit may report FAIL vs PL).
"""
from __future__ import annotations

import polars as pl

POGODA_FLAGS: tuple[str, ...] = (
    "pogoda_bez_opadu",
    "pogoda_deszcz",
    "pogoda_mgla",
    "pogoda_snieg_grad",
    "pogoda_wiatr",
    "pogoda_inne",
)

PL_COL = "warunki_atmosferyczne"
QLD_ATM_COL = "Crash_Atmospheric_Condition"


def _pl_tokens_expr(col: str = PL_COL) -> pl.Expr:
    return (
        pl.col(col)
        .fill_null("")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_lowercase()
        .str.split(";")
        .list.eval(pl.element().cast(pl.Utf8).str.strip_chars())
    )


def _pl_has_token(token: str, col: str = PL_COL) -> pl.Expr:
    return _pl_tokens_expr(col).list.contains(pl.lit(token))


def pl_pogoda_flag_exprs(col: str = PL_COL) -> list[pl.Expr]:
    deszcz = _pl_has_token("opady_deszczu", col)
    snieg = _pl_has_token("opady_sniegu_gradu", col)
    mgla = _pl_has_token("mgla_dym", col)
    wiatr = _pl_has_token("silny_wiatr", col)
    oslon = _pl_has_token("oslepiajace_slonce", col)
    nieznane = _pl_has_token("nieznane", col)
    dobre = _pl_has_token("dobre", col)
    pochmurno = _pl_has_token("pochmurno", col)

    adverse = deszcz | snieg | mgla | wiatr
    inne_src = oslon | nieznane
    raw = pl.col(col).fill_null("").cast(pl.Utf8).str.strip_chars()
    empty = raw.str.len_chars() == 0

    bez_opadu = (dobre | pochmurno) & ~adverse & ~inne_src & ~empty
    inne = inne_src | empty

    flag_sum = (
        deszcz.cast(pl.UInt8)
        + snieg.cast(pl.UInt8)
        + mgla.cast(pl.UInt8)
        + wiatr.cast(pl.UInt8)
        + bez_opadu.cast(pl.UInt8)
        + inne.cast(pl.UInt8)
    )
    inne = inne | (flag_sum == 0)

    return [
        bez_opadu.cast(pl.UInt8).alias("pogoda_bez_opadu"),
        deszcz.cast(pl.UInt8).alias("pogoda_deszcz"),
        mgla.cast(pl.UInt8).alias("pogoda_mgla"),
        snieg.cast(pl.UInt8).alias("pogoda_snieg_grad"),
        wiatr.cast(pl.UInt8).alias("pogoda_wiatr"),
        inne.cast(pl.UInt8).alias("pogoda_inne"),
    ]


def _qld_atm_expr(col: str = QLD_ATM_COL) -> pl.Expr:
    return pl.col(col).fill_null("").cast(pl.Utf8).str.strip_chars()


def qld_pogoda_flag_exprs(col: str = QLD_ATM_COL) -> list[pl.Expr]:
    atm = _qld_atm_expr(col)
    deszcz = atm == "Raining"
    mgla = atm == "Fog"
    inne = atm == "Smoke/Dust"
    bez_opadu = atm == "Clear"
    snieg = pl.lit(False)
    wiatr = pl.lit(False)

    flag_sum = (
        deszcz.cast(pl.UInt8)
        + mgla.cast(pl.UInt8)
        + inne.cast(pl.UInt8)
        + bez_opadu.cast(pl.UInt8)
    )
    inne = inne | (flag_sum == 0)

    return [
        bez_opadu.cast(pl.UInt8).alias("pogoda_bez_opadu"),
        deszcz.cast(pl.UInt8).alias("pogoda_deszcz"),
        mgla.cast(pl.UInt8).alias("pogoda_mgla"),
        snieg.cast(pl.UInt8).alias("pogoda_snieg_grad"),
        wiatr.cast(pl.UInt8).alias("pogoda_wiatr"),
        inne.cast(pl.UInt8).alias("pogoda_inne"),
    ]


def assert_pogoda_flags(df: pl.DataFrame, label: str) -> None:
    missing = [c for c in POGODA_FLAGS if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")

    for c in POGODA_FLAGS:
        bad = df.filter(~pl.col(c).is_in([0, 1])).height
        if bad:
            raise ValueError(f"{label}: {c} has values outside {{0,1}}: {bad} rows")

    none_active = df.filter(pl.sum_horizontal([pl.col(c) for c in POGODA_FLAGS]) == 0).height
    if none_active:
        raise ValueError(f"{label}: {none_active} rows without any flag = 1")
