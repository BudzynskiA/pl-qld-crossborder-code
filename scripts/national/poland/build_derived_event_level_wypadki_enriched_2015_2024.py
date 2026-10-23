"""
Builds `<PLQLD_POLAND_DIR>/data/derived/event_level/zdarzenia_wypadki_enriched_2015_2024.parquet`.

Extends `zdarzenia_wypadki_2015_2024.parquet` with aggregates from the participant
table (`uczestnik`) and up to three vehicles A/B/C from the vehicle table (`pojazdy`);
the vehicle A selection rule is a CLI parameter.

New columns:
  - ilosc_zmarlych_na_miejscu, ilosc_zmarlych_30d (killed at the scene / within 30 days)
  - ilosc_feriti, ilosc_feriti_m, ilosc_feriti_f (injured, total / male / female)
  - ilosc_morti_m, ilosc_morti_f (killed, male / female)
  - ilosc_pieszych, ilosc_pieszych_poszkodowanych (pedestrians, total / injured or killed)
  - ilosc_pojazdow (number of vehicles)
  - rodzaj_pojazdu_a/b/c, rok_produkcji_a/b/c (vehicle type, production year)
  - vehicle_a_rule (value of --vehicle-a)

Inputs (not part of this repository):
  - <PLQLD_POLAND_DIR>/data/derived/event_level/zdarzenia_wypadki_2015_2024.parquet
  - <PLQLD_POLAND_DIR>/data/merged/2015_2025/uczestnik/uczestnik_2015_2025.parquet
  - <PLQLD_POLAND_DIR>/data/merged/2015_2025/pojazdy/pojazdy_2015_2025.parquet

Run from the repository root:
  python scripts/national/poland/build_derived_event_level_wypadki_enriched_2015_2024.py
  python scripts/national/poland/build_derived_event_level_wypadki_enriched_2015_2024.py --vehicle-a first_id
"""
from __future__ import annotations

import argparse
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
    RODZ_PIESZY,
    SPRAWCA_TAK,
)

MERGED = POLAND / "data" / "merged" / "2015_2025"
BASE_PATH = POLAND / "data" / "derived" / "event_level" / "zdarzenia_wypadki_2015_2024.parquet"
UCZESTNIK = MERGED / "uczestnik" / "uczestnik_2015_2025.parquet"
POJAZDY = MERGED / "pojazdy" / "pojazdy_2015_2025.parquet"
OUT_DIR = POLAND / "data" / "derived" / "event_level"
OUT_PATH = OUT_DIR / "zdarzenia_wypadki_enriched_2015_2024.parquet"

COL_EVT = "ID zdarzenia"
COL_UCZ = "ID uczestnika"
COL_POJ = "ID pojazdu"

YEAR_FROM = 2015
YEAR_TO = 2024

INJURED_OR_DEAD = (POZIOM_LEKKO, POZIOM_CIEZKO, POZIOM_ZM_MIEJSCE, POZIOM_ZM_30)
FERITI = (POZIOM_LEKKO, POZIOM_CIEZKO)
DEAD = (POZIOM_ZM_MIEJSCE, POZIOM_ZM_30)

VEHICLE_A_RULES = ("sprawca", "first_id")


def _strip_id(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.String).str.strip_chars().str.replace(r"\.0$", "")


def _pick_column(schema_names: list[str], aliases: list[str], label: str) -> str:
    for name in aliases:
        if name in schema_names:
            return name
    raise SystemExit(f"Missing column {label!r} (looked for: {aliases})")


def _deduped_uczestnik_lf() -> pl.LazyFrame:
    lf = pl.scan_parquet(UCZESTNIK)
    schema = lf.collect_schema()
    for c in (COL_EVT, COL_UCZ, COL_POZIOM, "plec", "rodzaj_uczestnika", "czy_sprawca"):
        if c not in schema.names():
            raise SystemExit(f"Missing column {c!r} in {UCZESTNIK}")

    sort_cols = [COL_EVT, COL_UCZ]
    if "rok" in schema.names():
        sort_cols.append("rok")

    return (
        lf.filter(pl.col("rok").is_between(YEAR_FROM, YEAR_TO) if "rok" in schema.names() else pl.lit(True))
        .with_columns(
            _strip_id(pl.col(COL_EVT)).alias(COL_EVT),
            _strip_id(pl.col(COL_UCZ)).alias(COL_UCZ),
            _strip_id(pl.col(COL_POJ)).alias(COL_POJ),
            pl.col(COL_POZIOM).cast(pl.String).str.strip_chars().alias(COL_POZIOM),
            pl.col("plec").cast(pl.String).str.strip_chars().alias("plec"),
            pl.col("rodzaj_uczestnika").cast(pl.String).str.strip_chars().alias("rodzaj_uczestnika"),
            pl.col("czy_sprawca").cast(pl.String).str.strip_chars().alias("czy_sprawca"),
        )
        .sort(sort_cols)
        .unique(subset=[COL_EVT, COL_UCZ], keep="first")
    )


def _participant_enrichment(ucz: pl.LazyFrame) -> pl.LazyFrame:
    po = pl.col(COL_POZIOM)
    plec = pl.col("plec")
    rodz = pl.col("rodzaj_uczestnika")
    feriti = po.is_in(list(FERITI))
    dead = po.is_in(list(DEAD))
    injured = po.is_in(list(INJURED_OR_DEAD))

    return ucz.group_by(COL_EVT).agg(
        pl.col(COL_UCZ).filter(po == pl.lit(POZIOM_ZM_MIEJSCE)).n_unique().alias("ilosc_zmarlych_na_miejscu"),
        pl.col(COL_UCZ).filter(po == pl.lit(POZIOM_ZM_30)).n_unique().alias("ilosc_zmarlych_30d"),
        pl.col(COL_UCZ).filter(feriti).n_unique().alias("ilosc_feriti"),
        pl.col(COL_UCZ).filter(feriti & (plec == pl.lit("mezczyzna"))).n_unique().alias("ilosc_feriti_m"),
        pl.col(COL_UCZ).filter(feriti & (plec == pl.lit("kobieta"))).n_unique().alias("ilosc_feriti_f"),
        pl.col(COL_UCZ).filter(dead & (plec == pl.lit("mezczyzna"))).n_unique().alias("ilosc_morti_m"),
        pl.col(COL_UCZ).filter(dead & (plec == pl.lit("kobieta"))).n_unique().alias("ilosc_morti_f"),
        pl.col(COL_UCZ).filter(rodz == pl.lit(RODZ_PIESZY)).n_unique().alias("ilosc_pieszych"),
        pl.col(COL_UCZ)
        .filter((rodz == pl.lit(RODZ_PIESZY)) & injured)
        .n_unique()
        .alias("ilosc_pieszych_poszkodowanych"),
    )


def _pojazdy_base() -> pl.LazyFrame:
    lf = pl.scan_parquet(POJAZDY)
    schema = lf.collect_schema().names()
    rodz_col = _pick_column(schema, ["rodzaj_pojazdu", "Rodzaj pojazdu"], "rodzaj pojazdu")
    rok_col = _pick_column(schema, ["rok_produkcji", "Rok produkcji"], "rok produkcji")

    year_filter = pl.col("rok").is_between(YEAR_FROM, YEAR_TO) if "rok" in schema else pl.lit(True)

    return (
        lf.filter(year_filter)
        .with_columns(
            _strip_id(pl.col(COL_EVT)).alias(COL_EVT),
            _strip_id(pl.col(COL_POJ)).alias(COL_POJ),
            pl.col(rodz_col).cast(pl.String).str.strip_chars().alias("_rodzaj"),
            pl.col(rok_col).cast(pl.String).str.strip_chars().alias("_rok"),
        )
        .select(COL_EVT, COL_POJ, "_rodzaj", "_rok")
        .unique(subset=[COL_EVT, COL_POJ], keep="first")
        .sort([COL_EVT, COL_POJ])
    )


def _vehicle_a_ids(poj: pl.LazyFrame, ucz: pl.LazyFrame, rule: str) -> pl.LazyFrame:
    first_by_id = poj.group_by(COL_EVT).agg(pl.col(COL_POJ).first().alias("_id_a_fallback"))

    if rule == "first_id":
        return first_by_id.rename({"_id_a_fallback": "_id_a"})

    sprawca = (
        ucz.filter(pl.col("czy_sprawca") == pl.lit(SPRAWCA_TAK))
        .sort([COL_EVT, COL_UCZ])
        .group_by(COL_EVT)
        .agg(
            pl.col(COL_POJ)
            .filter(pl.col(COL_POJ).str.len_chars() > 0)
            .first()
            .alias("_id_a_sprawca")
        )
    )

    return (
        first_by_id.join(sprawca, on=COL_EVT, how="left")
        .with_columns(
            pl.when(pl.col("_id_a_sprawca").str.len_chars() > 0)
            .then(pl.col("_id_a_sprawca"))
            .otherwise(pl.col("_id_a_fallback"))
            .alias("_id_a")
        )
        .select(COL_EVT, "_id_a")
    )


def _vehicle_enrichment(poj: pl.LazyFrame, ucz: pl.LazyFrame, rule: str) -> pl.LazyFrame:
    id_a = _vehicle_a_ids(poj, ucz, rule)

    id_b = (
        poj.join(id_a, on=COL_EVT, how="left")
        .filter(pl.col(COL_POJ) != pl.col("_id_a"))
        .sort([COL_EVT, COL_POJ])
        .group_by(COL_EVT)
        .agg(pl.col(COL_POJ).first().alias("_id_b"))
    )

    id_c = (
        poj.join(id_a, on=COL_EVT, how="left")
        .join(id_b, on=COL_EVT, how="left")
        .filter(pl.col(COL_POJ) != pl.col("_id_a"))
        .filter(pl.col(COL_POJ) != pl.col("_id_b"))
        .sort([COL_EVT, COL_POJ])
        .group_by(COL_EVT)
        .agg(pl.col(COL_POJ).first().alias("_id_c"))
    )

    ilosc = poj.group_by(COL_EVT).agg(pl.col(COL_POJ).n_unique().alias("ilosc_pojazdow"))

    poj_attrs = poj.select(
        COL_EVT,
        pl.col(COL_POJ).alias("_veh_id"),
        pl.col("_rodzaj"),
        pl.col("_rok"),
    )

    def _slot(slot: str, id_col: str) -> pl.LazyFrame:
        return (
            id_a if slot == "a" else id_b if slot == "b" else id_c
        ).join(
            poj_attrs,
            left_on=[COL_EVT, id_col],
            right_on=[COL_EVT, "_veh_id"],
            how="left",
        ).select(
            COL_EVT,
            pl.col("_rodzaj").alias(f"rodzaj_pojazdu_{slot}"),
            pl.col("_rok").alias(f"rok_produkcji_{slot}"),
        )

    slots = _slot("a", "_id_a").join(_slot("b", "_id_b"), on=COL_EVT, how="left").join(
        _slot("c", "_id_c"), on=COL_EVT, how="left"
    )

    return (
        ilosc.join(slots, on=COL_EVT, how="left")
        .with_columns(pl.lit(rule).alias("vehicle_a_rule"))
    )


def _fill_uint32(lf: pl.LazyFrame, cols: list[str]) -> pl.LazyFrame:
    return lf.with_columns([pl.col(c).fill_null(0).cast(pl.UInt32).alias(c) for c in cols])


def main() -> None:
    parser = argparse.ArgumentParser(description="Event-level SEWIK wypadki enriched 2015-2024.")
    parser.add_argument(
        "--vehicle-a",
        choices=VEHICLE_A_RULES,
        default="sprawca",
        help="Vehicle A selection rule (default: vehicle of the at-fault participant, fallback: lowest vehicle ID).",
    )
    args = parser.parse_args()

    for path in (BASE_PATH, UCZESTNIK, POJAZDY):
        if not path.is_file():
            raise SystemExit(f"Missing file: {path}")

    base = pl.scan_parquet(BASE_PATH).with_columns(_strip_id(pl.col(COL_EVT)).alias(COL_EVT))
    ucz = _deduped_uczestnik_lf()
    poj = _pojazdy_base()

    part = _participant_enrichment(ucz)
    veh = _vehicle_enrichment(poj, ucz, args.vehicle_a)

    count_cols = [
        "ilosc_zmarlych_na_miejscu",
        "ilosc_zmarlych_30d",
        "ilosc_feriti",
        "ilosc_feriti_m",
        "ilosc_feriti_f",
        "ilosc_morti_m",
        "ilosc_morti_f",
        "ilosc_pieszych",
        "ilosc_pieszych_poszkodowanych",
        "ilosc_pojazdow",
    ]

    out = (
        base.join(part, on=COL_EVT, how="left")
        .join(veh, on=COL_EVT, how="left")
        .pipe(_fill_uint32, count_cols)
        .collect()
    )

    if out.height == 0:
        raise SystemExit("Empty result: check zdarzenia_wypadki_2015_2024.parquet.")

    split_sum = out.select(
        (pl.col("ilosc_zmarlych_na_miejscu") + pl.col("ilosc_zmarlych_30d")).alias("split_dead"),
        pl.col("ilosc_zmarlych").alias("dead"),
    )
    mismatch = split_sum.filter(pl.col("split_dead") != pl.col("dead")).height
    if mismatch:
        raise SystemExit(
            f"Death count check failed: {mismatch} rows where at-scene + 30-day deaths != ilosc_zmarlych of the base file."
        )

    feriti_mismatch = out.filter(pl.col("ilosc_feriti") != pl.col("ilosc_lekko_rannych") + pl.col("ilosc_ciezko_rannych")).height
    if feriti_mismatch:
        raise SystemExit(
            f"Injury count check failed: {feriti_mismatch} rows where ilosc_feriti != slightly + seriously injured."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_PATH, compression="snappy")

    n_a = out.filter(pl.col("rodzaj_pojazdu_a").str.len_chars() > 0).height
    n_b = out.filter(pl.col("rodzaj_pojazdu_b").str.len_chars() > 0).height
    n_c = out.filter(pl.col("rodzaj_pojazdu_c").str.len_chars() > 0).height
    print(
        f"Wrote {out.height:,} rows -> {display_path(OUT_PATH)} "
        f"(vehicle_a_rule={args.vehicle_a!r})"
    )
    print(f"Vehicles A/B/C filled: {n_a:,} / {n_b:,} / {n_c:,}")


if __name__ == "__main__":
    main()
