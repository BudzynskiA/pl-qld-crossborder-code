"""
Shared 0/1 road surface flags (PL SEWIK + QLD TMR), event level.

PL: same definitions as the Poland-Italy study (`nawierzchnia_drogi`, `stan_nawierzchni`).
QLD: `Crash_Road_Surface_Condition` (Sealed/Unsealed × Dry/Wet).

QLD does not distinguish ice, snow, contamination or damage → the corresponding flags = 0.
"""
from __future__ import annotations

import polars as pl

PL_NAW_COL = "nawierzchnia_drogi"
PL_STAN_COL = "stan_nawierzchni"
QLD_SURFACE_COL = "Crash_Road_Surface_Condition"

NAW_TYP_FLAGS: tuple[str, ...] = (
    "naw_twarda",
    "naw_gruntowa",
)

NAW_STAN_FLAGS: tuple[str, ...] = (
    "naw_sucha",
    "naw_mokra",
    "naw_sdrucciolevole",
    "naw_lod",
    "naw_snieg",
)

NAW_USZKODZONA_FLAG = "naw_uszkodzona"

NAWIERZCHNIA_FLAGS: tuple[str, ...] = NAW_TYP_FLAGS + NAW_STAN_FLAGS + (NAW_USZKODZONA_FLAG,)


def _pl_stan_tokens_expr(col: str = PL_STAN_COL) -> pl.Expr:
    return (
        pl.col(col)
        .fill_null("")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_lowercase()
        .str.split(";")
        .list.eval(pl.element().cast(pl.Utf8).str.strip_chars())
    )


def _pl_has_stan_token(token: str, col: str = PL_STAN_COL) -> pl.Expr:
    return _pl_stan_tokens_expr(col).list.contains(pl.lit(token))


def _pl_naw_expr(col: str = PL_NAW_COL) -> pl.Expr:
    return pl.col(col).fill_null("").cast(pl.Utf8).str.strip_chars().str.to_lowercase()


def pl_nawierzchnia_flag_exprs(
    naw_col: str = PL_NAW_COL,
    stan_col: str = PL_STAN_COL,
) -> list[pl.Expr]:
    naw = _pl_naw_expr(naw_col)
    mokra = _pl_has_stan_token("mokra", stan_col) | _pl_has_stan_token("kaluze_rozlewiska", stan_col)
    sdrucc = _pl_has_stan_token("zanieczyszczona", stan_col)
    lod = _pl_has_stan_token("oblodzenie_zasniezenie", stan_col)
    snieg = _pl_has_stan_token("oblodzona_zasniezona", stan_col)
    uszk = _pl_has_stan_token("dziury_wyboje", stan_col) | _pl_has_stan_token("koleiny_garby", stan_col)
    sucha = _pl_has_stan_token("sucha", stan_col) | (~mokra & ~sdrucc & ~lod & ~snieg)

    return [
        (naw == "twarda").cast(pl.UInt8).alias("naw_twarda"),
        (naw == "gruntowa").cast(pl.UInt8).alias("naw_gruntowa"),
        sucha.cast(pl.UInt8).alias("naw_sucha"),
        mokra.cast(pl.UInt8).alias("naw_mokra"),
        sdrucc.cast(pl.UInt8).alias("naw_sdrucciolevole"),
        lod.cast(pl.UInt8).alias("naw_lod"),
        snieg.cast(pl.UInt8).alias("naw_snieg"),
        uszk.cast(pl.UInt8).alias("naw_uszkodzona"),
    ]


def _qld_surface_expr(col: str = QLD_SURFACE_COL) -> pl.Expr:
    return pl.col(col).fill_null("").cast(pl.Utf8).str.strip_chars()


def qld_nawierzchnia_flag_exprs(col: str = QLD_SURFACE_COL) -> list[pl.Expr]:
    surf = _qld_surface_expr(col)
    sealed = surf.str.starts_with("Sealed")
    unsealed = surf.str.starts_with("Unsealed")
    dry = surf.str.ends_with("Dry")
    wet = surf.str.ends_with("Wet")
    mokra = wet
    sucha = dry
    return [
        sealed.cast(pl.UInt8).alias("naw_twarda"),
        unsealed.cast(pl.UInt8).alias("naw_gruntowa"),
        sucha.cast(pl.UInt8).alias("naw_sucha"),
        mokra.cast(pl.UInt8).alias("naw_mokra"),
        pl.lit(0).cast(pl.UInt8).alias("naw_sdrucciolevole"),
        pl.lit(0).cast(pl.UInt8).alias("naw_lod"),
        pl.lit(0).cast(pl.UInt8).alias("naw_snieg"),
        pl.lit(0).cast(pl.UInt8).alias("naw_uszkodzona"),
    ]


def assert_nawierzchnia_flags(df: pl.DataFrame, label: str) -> None:
    missing = [c for c in NAWIERZCHNIA_FLAGS if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")

    for c in NAWIERZCHNIA_FLAGS:
        bad = df.filter(~pl.col(c).is_in([0, 1])).height
        if bad:
            raise ValueError(f"{label}: {c} has values outside {{0,1}}: {bad} rows")

    typ_sum = pl.sum_horizontal([pl.col(c) for c in NAW_TYP_FLAGS])
    bad_typ = df.filter(typ_sum != 1).height
    if bad_typ:
        raise ValueError(f"{label}: {bad_typ} rows with naw_twarda+naw_gruntowa != 1")

    stan_sum = pl.sum_horizontal([pl.col(c) for c in NAW_STAN_FLAGS])
    no_stan = df.filter(stan_sum == 0).height
    if no_stan:
        raise ValueError(f"{label}: {no_stan} rows without any condition flag (=1)")
