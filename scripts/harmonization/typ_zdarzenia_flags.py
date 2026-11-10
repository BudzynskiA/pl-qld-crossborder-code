"""
Shared 0/1 crash type flags (PL SEWIK + QLD TMR), event level.

PL: `rodzaj_zdarzenia` (SEWIK field definitions).
QLD: `Crash_Nature` (preferred), with `Crash_Type` as fallback.
"""
from __future__ import annotations

import polars as pl

PL_COL = "rodzaj_zdarzenia"
QLD_NATURE_COL = "Crash_Nature"
QLD_TYPE_COL = "Crash_Type"

TYP_ZDARZENIA_FLAGS: tuple[str, ...] = (
    "typ_czolowe",
    "typ_boczne",
    "typ_tylne",
    "typ_pieszy",
    "typ_wywrocenie",
    "typ_ostacol",
    "typ_pojazd_stojacy",
    "typ_kolej",
    "typ_upadek_pasazer",
    "typ_inne",
)

_PL_TO_FLAG: dict[str, str] = {
    "zderzenie_czolowe": "typ_czolowe",
    "zderzenie_boczne": "typ_boczne",
    "zderzenie_tylne": "typ_tylne",
    "najechanie_na_pieszego": "typ_pieszy",
    "wywrocenie_pojazdu": "typ_wywrocenie",
    "najechanie_na_drzewo": "typ_ostacol",
    "najechanie_na_slup_znak": "typ_ostacol",
    "najechanie_na_bariere": "typ_ostacol",
    "najechanie_na_zwierze": "typ_ostacol",
    "najechanie_na_dziure_wyboi_garb": "typ_ostacol",
    "najechanie_na_pojazd_unieruchomiony": "typ_pojazd_stojacy",
    "najechanie_na_zapore_kolejowa": "typ_kolej",
    "zdarzenie_z_pasazerem": "typ_upadek_pasazer",
    "zdarzenie_z_osoba_uwr": "typ_inne",
    "inne": "typ_inne",
}

_QLD_NATURE_TO_FLAG: dict[str, str] = {
    "Head-on": "typ_czolowe",
    "Rear-end": "typ_tylne",
    "Angle": "typ_boczne",
    "Sideswipe": "typ_boczne",
    "Hit pedestrian": "typ_pieszy",
    "Overturned": "typ_wywrocenie",
    "Hit object": "typ_ostacol",
    "Hit animal": "typ_ostacol",
    "Hit parked vehicle": "typ_pojazd_stojacy",
    "Fall from vehicle": "typ_upadek_pasazer",
    "Struck by external load": "typ_inne",
    "Struck by internal load": "typ_inne",
}

_QLD_TYPE_TO_FLAG: dict[str, str] = {
    "Hit pedestrian": "typ_pieszy",
    "Other": "typ_inne",
}


def _pl_rodzaj_expr(col: str = PL_COL) -> pl.Expr:
    return pl.col(col).fill_null("").cast(pl.Utf8).str.strip_chars().str.to_lowercase()


def _qld_text_expr(col: str) -> pl.Expr:
    return pl.col(col).fill_null("").cast(pl.Utf8).str.strip_chars()


def _flag_from_map(value_expr: pl.Expr, mapping: dict[str, str], default: str) -> pl.Expr:
    expr = pl.lit(default)
    for key, flag in mapping.items():
        expr = pl.when(value_expr == pl.lit(key)).then(pl.lit(flag)).otherwise(expr)
    return expr


def pl_typ_zdarzenia_flag_exprs(col: str = PL_COL) -> list[pl.Expr]:
    rodzaj = _pl_rodzaj_expr(col)
    assigned = _flag_from_map(rodzaj, _PL_TO_FLAG, "typ_inne")
    return [(assigned == pl.lit(f)).cast(pl.UInt8).alias(f) for f in TYP_ZDARZENIA_FLAGS]


def qld_typ_zdarzenia_flag_exprs(
    nature_col: str = QLD_NATURE_COL,
    type_col: str = QLD_TYPE_COL,
) -> list[pl.Expr]:
    nature = _qld_text_expr(nature_col)
    crash_type = _qld_text_expr(type_col)
    from_nature = _flag_from_map(nature, _QLD_NATURE_TO_FLAG, "")
    from_type = _flag_from_map(crash_type, _QLD_TYPE_TO_FLAG, "typ_inne")
    assigned = pl.when(from_nature != pl.lit("")).then(from_nature).otherwise(from_type)
    return [(assigned == pl.lit(f)).cast(pl.UInt8).alias(f) for f in TYP_ZDARZENIA_FLAGS]


def assert_typ_zdarzenia_flags(df: pl.DataFrame, label: str) -> None:
    missing = [c for c in TYP_ZDARZENIA_FLAGS if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")

    for c in TYP_ZDARZENIA_FLAGS:
        bad = df.filter(~pl.col(c).is_in([0, 1])).height
        if bad:
            raise ValueError(f"{label}: {c} has values outside {{0,1}}: {bad} rows")

    n_flags = pl.sum_horizontal([pl.col(c) for c in TYP_ZDARZENIA_FLAGS])
    not_one = df.filter(n_flags != 1).height
    if not_one:
        raise ValueError(f"{label}: {not_one} rows with flag sum != 1 (exactly 1 expected)")
