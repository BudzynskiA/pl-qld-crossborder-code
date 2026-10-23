"""
Builds the harmonized event-level PL + QLD panel (27 binary 0/1 flags + outcomes).

Input:
  <PLQLD_POLAND_DIR>/data/derived/event_level/zdarzenia_wypadki_enriched_2015_2024.parquet
    (fallback: zdarzenia_wypadki_2015_2024.parquet)
  <PLQLD_QUEENSLAND_DIR>/data/derived/event_level/event_level_qld_2015_2024.parquet

Output:
  <PLQLD_DATA_DIR>/event_level/event_level_pl_qld_2015_2024.parquet
  tables_csv/harmonized_flags_coverage_pl_qld.csv
  tables_csv/harmonized_flags_country_leakage_audit.csv
  tables_md/harmonization_event_level_pl_qld.md
  tables_md/harmonized_flags_country_leakage_audit.md

Run from the repository root:
  python scripts/harmonization/build_event_level_pl_qld_harmonized.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))
from _bootstrap import HARMONIZED_EVENT_LEVEL_DIR, PL_QLD_ROOT, POLAND, QUEENSLAND, TABLES_CSV, TABLES_MD, display_path, install  # noqa: E402

install()

import polars as pl  # noqa: E402

from harmonization.country_leakage_audit import (  # noqa: E402
    audit_flag_country_leakage,
    audit_group_unassigned_leakage,
    write_leakage_audit_md,
)
from harmonization.lokalizacja_flags import (  # noqa: E402
    LOKALIZACJA_FLAGS,
    PL_OBSZAR_COL,
    PL_RODZAJ_COL,
    QLD_AUTH_COL,
    QLD_REMOTENESS_COL,
    QLD_SPEED_COL,
    QLD_STATE_ROAD_COL,
    assert_lokalizacja_flags,
    pl_lokalizacja_flag_exprs,
    qld_lokalizacja_flag_exprs,
)
from harmonization.nawierzchnia_flags import (  # noqa: E402
    NAWIERZCHNIA_FLAGS,
    PL_NAW_COL,
    PL_STAN_COL,
    QLD_SURFACE_COL,
    assert_nawierzchnia_flags,
    pl_nawierzchnia_flag_exprs,
    qld_nawierzchnia_flag_exprs,
)
from harmonization.outcome_morti import (  # noqa: E402
    FERITI_COUNT,
    MORTI_COUNT,
    PL_COUNT_COL,
    PL_FERITI_COL,
    Y_MORTI,
    assert_outcome_morti,
    pl_morti_exprs,
    qld_morti_exprs,
)
from harmonization.pogoda_flags import (  # noqa: E402
    PL_COL as POGODA_PL_COL,
    POGODA_FLAGS,
    QLD_ATM_COL,
    assert_pogoda_flags,
    pl_pogoda_flag_exprs,
    qld_pogoda_flag_exprs,
)
from harmonization.typ_zdarzenia_flags import (  # noqa: E402
    PL_COL as TYP_PL_COL,
    QLD_NATURE_COL,
    QLD_TYPE_COL,
    TYP_ZDARZENIA_FLAGS,
    assert_typ_zdarzenia_flags,
    pl_typ_zdarzenia_flag_exprs,
    qld_typ_zdarzenia_flag_exprs,
)

PL_CANDIDATES = (
    POLAND / "data" / "derived" / "event_level" / "zdarzenia_wypadki_enriched_2015_2024.parquet",
    POLAND / "data" / "derived" / "event_level" / "zdarzenia_wypadki_2015_2024.parquet",
)
QLD_IN = QUEENSLAND / "data" / "derived" / "event_level" / "event_level_qld_2015_2024.parquet"
OUT_DIR = HARMONIZED_EVENT_LEVEL_DIR
OUT_PATH = OUT_DIR / "event_level_pl_qld_2015_2024.parquet"
OUT_MD = TABLES_MD / "harmonization_event_level_pl_qld.md"
OUT_COV = TABLES_CSV / "harmonized_flags_coverage_pl_qld.csv"
OUT_LEAK_CSV = TABLES_CSV / "harmonized_flags_country_leakage_audit.csv"
OUT_LEAK_MD = TABLES_MD / "harmonized_flags_country_leakage_audit.md"

COL_EVT_PL = "ID zdarzenia"
COL_EVT_QLD = "Crash_Ref_Number"

ALL_FLAG_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pogoda", POGODA_FLAGS),
    ("typ_zdarzenia", TYP_ZDARZENIA_FLAGS),
    ("nawierzchnia", NAWIERZCHNIA_FLAGS),
    ("lokalizacja", LOKALIZACJA_FLAGS),
)


def _resolve_pl_input() -> Path:
    for path in PL_CANDIDATES:
        if path.is_file():
            return path
    return PL_CANDIDATES[0]


def _coverage_rows(df: pl.DataFrame, country: str, cecha: str, flags: tuple[str, ...], year_col: str) -> list[dict]:
    n = df.height
    rows: list[dict] = []
    for flag in flags:
        rows.append(
            {
                "cecha": cecha,
                "kraj": country,
                "flaga": flag,
                "rok": None,
                "n_1": df.filter(pl.col(flag) == 1).height,
                "udzial_1": round(df.filter(pl.col(flag) == 1).height / n, 6) if n else 0.0,
                "panel_n": n,
            }
        )
    if year_col not in df.columns:
        return rows
    for flag in flags:
        by_year = (
            df.group_by(year_col)
            .agg(pl.len().alias("n"), (pl.col(flag) == 1).sum().alias("n_1"))
            .with_columns((pl.col("n_1") / pl.col("n")).round(6).alias("udzial_1"))
        )
        for row in by_year.iter_rows(named=True):
            rows.append(
                {
                    "cecha": cecha,
                    "kraj": country,
                    "flaga": flag,
                    "rok": row[year_col],
                    "n_1": row["n_1"],
                    "udzial_1": row["udzial_1"],
                    "panel_n": row["n"],
                }
            )
    return rows


def _write_spec_md(pl_df: pl.DataFrame, qld_df: pl.DataFrame, pl_source: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _cov_table(df: pl.DataFrame, flags: tuple[str, ...]) -> str:
        n = df.height
        lines = ["| flag | n(=1) | share |", "|-------|------:|-------:|"]
        for flag in sorted(flags):
            n1 = df.filter(pl.col(flag) == 1).height
            lines.append(f"| {flag} | {n1:,} | {n1 / n:.4f} |" if n else f"| {flag} | 0 | - |")
        return "\n".join(lines)

    def _outcome_table(df: pl.DataFrame, country: str) -> str:
        n = df.height
        if not n:
            return f"| {country} | 0 | 0 | - | - | - |"
        n1 = df.filter(pl.col(Y_MORTI) == 1).height
        mean_m = df[MORTI_COUNT].mean() or 0.0
        mean_f = df[FERITI_COUNT].mean() or 0.0
        return f"| {country} | {n:,} | {n1:,} | {n1 / n:.4f} | {mean_m:.4f} | {mean_f:.4f} |"

    text = f"""# Event-level harmonization PL ↔ QLD

**Generated:** {ts} (`scripts/harmonization/build_event_level_pl_qld_harmonized.py`)

**Panel file:** `<PLQLD_DATA_DIR>/event_level/event_level_pl_qld_2015_2024.parquet`

**PL input:** `{display_path(pl_source)}`

## Weather (`pogoda_*`)

Rules: [`../scripts/harmonization/pogoda_flags.py`](../scripts/harmonization/pogoda_flags.py)

PL: `warunki_atmosferyczne` tokens; QLD: `Crash_Atmospheric_Condition`.

### PL

{_cov_table(pl_df, POGODA_FLAGS)}

### QLD

{_cov_table(qld_df, POGODA_FLAGS)}

## Crash type (`typ_*`)

Rules: [`../scripts/harmonization/typ_zdarzenia_flags.py`](../scripts/harmonization/typ_zdarzenia_flags.py)

PL: `rodzaj_zdarzenia`; QLD: `Crash_Nature` (+ fallback `Crash_Type`).

### PL

{_cov_table(pl_df, TYP_ZDARZENIA_FLAGS)}

### QLD

{_cov_table(qld_df, TYP_ZDARZENIA_FLAGS)}

## Road surface (`naw_*`)

Rules: [`../scripts/harmonization/nawierzchnia_flags.py`](../scripts/harmonization/nawierzchnia_flags.py)

PL: `nawierzchnia_drogi` + `stan_nawierzchni`; QLD: `Crash_Road_Surface_Condition`.

### PL

{_cov_table(pl_df, NAWIERZCHNIA_FLAGS)}

### QLD

{_cov_table(qld_df, NAWIERZCHNIA_FLAGS)}

## Location (`lok_*`)

Rules: [`../scripts/harmonization/lokalizacja_flags.py`](../scripts/harmonization/lokalizacja_flags.py)

QLD proxy: motorway/freeway in `State_Road_Name`, `Loc_ABS_Remoteness`, `Crash_Controlling_Authority`, `Crash_Speed_Limit`.

### PL

{_cov_table(pl_df, LOKALIZACJA_FLAGS)}

### QLD

{_cov_table(qld_df, LOKALIZACJA_FLAGS)}

## Outcome: fatalities and injuries

Rules: [`../scripts/harmonization/outcome_morti.py`](../scripts/harmonization/outcome_morti.py)

| country | n | n(y=1) | prev | mean morti_count | mean feriti_count |
|------|--:|-------:|-----:|-----------------:|------------------:|
{_outcome_table(pl_df, "PL")}
{_outcome_table(qld_df, "QLD")}

## Country leakage audit

[`../tables_md/harmonized_flags_country_leakage_audit.md`](../tables_md/harmonized_flags_country_leakage_audit.md)

## CSV

[`../tables_csv/harmonized_flags_coverage_pl_qld.csv`](../tables_csv/harmonized_flags_coverage_pl_qld.csv)
"""
    OUT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    pl_in = _resolve_pl_input()
    if not pl_in.is_file():
        print(f"[err] PL file not found: {pl_in}", file=sys.stderr)
        print("  Build the derived Polish event-level file under PLQLD_POLAND_DIR and run again.", file=sys.stderr)
        return 1
    if not QLD_IN.is_file():
        print(f"[err] QLD file not found: {QLD_IN}", file=sys.stderr)
        return 1

    pl_schema = pl.scan_parquet(pl_in).collect_schema().names()
    qld_schema = pl.scan_parquet(QLD_IN).collect_schema().names()

    pl_checks = (
        (POGODA_PL_COL, "PL pogoda"),
        (TYP_PL_COL, "PL typ"),
        (PL_NAW_COL, "PL nawierzchnia_drogi"),
        (PL_STAN_COL, "PL stan_nawierzchni"),
        (PL_OBSZAR_COL, "PL obszar"),
        (PL_RODZAJ_COL, "PL rodzaj_drogi"),
        (PL_COUNT_COL, "PL ilosc_zmarlych"),
        (PL_FERITI_COL, "PL ilosc_feriti"),
    )
    qld_checks = (
        (QLD_ATM_COL, "QLD atmospheric"),
        (QLD_NATURE_COL, "QLD nature"),
        (QLD_TYPE_COL, "QLD type"),
        (QLD_SURFACE_COL, "QLD surface"),
        (QLD_REMOTENESS_COL, "QLD remoteness"),
        (QLD_STATE_ROAD_COL, "QLD state road"),
        (QLD_AUTH_COL, "QLD authority"),
        (QLD_SPEED_COL, "QLD speed"),
        ("morti_count", "QLD morti_count"),
        ("feriti_count", "QLD feriti_count"),
    )
    for col, label in pl_checks:
        if col not in pl_schema:
            print(f"[err] Missing column {col!r} ({label}) in {pl_in.name}", file=sys.stderr)
            return 1
    for col, label in qld_checks:
        if col not in qld_schema:
            print(f"[err] Missing column {col!r} ({label}) in {QLD_IN.name}", file=sys.stderr)
            return 1

    pl_df = (
        pl.scan_parquet(pl_in)
        .with_columns(
            [
                pl.lit("PL").alias("kraj"),
                pl.col(COL_EVT_PL).cast(pl.Utf8).str.strip_chars().alias("event_id"),
                *pl_pogoda_flag_exprs(POGODA_PL_COL),
                *pl_typ_zdarzenia_flag_exprs(TYP_PL_COL),
                *pl_nawierzchnia_flag_exprs(PL_NAW_COL, PL_STAN_COL),
                *pl_lokalizacja_flag_exprs(PL_OBSZAR_COL, PL_RODZAJ_COL),
                *pl_morti_exprs(PL_COUNT_COL, PL_FERITI_COL),
            ]
        )
        .collect()
    )

    qld_df = (
        pl.scan_parquet(QLD_IN)
        .with_columns(
            [
                pl.lit("QLD").alias("kraj"),
                (pl.lit("QLD_") + pl.col(COL_EVT_QLD).cast(pl.Utf8).str.strip_chars()).alias("event_id"),
                pl.col("Crash_Year").cast(pl.Int64, strict=False).alias("rok"),
                *qld_pogoda_flag_exprs(QLD_ATM_COL),
                *qld_typ_zdarzenia_flag_exprs(QLD_NATURE_COL, QLD_TYPE_COL),
                *qld_nawierzchnia_flag_exprs(QLD_SURFACE_COL),
                *qld_lokalizacja_flag_exprs(
                    QLD_REMOTENESS_COL,
                    QLD_STATE_ROAD_COL,
                    QLD_AUTH_COL,
                    QLD_SPEED_COL,
                ),
                *qld_morti_exprs(),
            ]
        )
        .collect()
    )

    assert_pogoda_flags(pl_df, "PL")
    assert_pogoda_flags(qld_df, "QLD")
    assert_typ_zdarzenia_flags(pl_df, "PL")
    assert_typ_zdarzenia_flags(qld_df, "QLD")
    assert_nawierzchnia_flags(pl_df, "PL")
    assert_nawierzchnia_flags(qld_df, "QLD")
    assert_lokalizacja_flags(pl_df, "PL", allow_unassigned=False)
    assert_lokalizacja_flags(qld_df, "QLD", allow_unassigned=False)
    assert_outcome_morti(pl_df, "PL")
    assert_outcome_morti(qld_df, "QLD")

    panel = pl.concat([pl_df, qld_df], how="diagonal")
    dup = panel.group_by("event_id").len().filter(pl.col("len") > 1).height
    if dup:
        print(f"[err] Duplicate event_id values in the panel: {dup}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(OUT_PATH, compression="snappy")

    cov_rows: list[dict] = []
    for cecha, flags in ALL_FLAG_GROUPS:
        cov_rows.extend(_coverage_rows(pl_df, "PL", cecha, flags, "rok"))
        cov_rows.extend(_coverage_rows(qld_df, "QLD", cecha, flags, "rok"))
    TABLES_CSV.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(cov_rows).write_csv(OUT_COV)

    leak_df = audit_flag_country_leakage(pl_df, qld_df, ALL_FLAG_GROUPS)
    unassigned = audit_group_unassigned_leakage(pl_df, qld_df, "lokalizacja", LOKALIZACJA_FLAGS)
    leak_df = pl.concat([leak_df, pl.DataFrame([unassigned])], how="diagonal")
    leak_df.write_csv(OUT_LEAK_CSV)
    write_leakage_audit_md(leak_df, OUT_LEAK_MD)

    _write_spec_md(pl_df, qld_df, pl_in)

    n_fail = leak_df.filter(pl.col("status") == "FAIL").height
    n_warn = leak_df.filter(pl.col("status") == "WARN").height
    print(
        f"Panel: PL {pl_df.height:,} + QLD {qld_df.height:,} = {panel.height:,} -> "
        f"{display_path(OUT_PATH)}"
    )
    print(f"Documentation: {display_path(OUT_MD)}")
    print(f"Coverage: {display_path(OUT_COV)}")
    print(f"Leakage audit: {display_path(OUT_LEAK_MD)} (OK/WARN/FAIL = "
          f"{leak_df.filter(pl.col('status')=='OK').height}/{n_warn}/{n_fail})")
    if n_fail:
        print(
            "[warn] FAIL in the audit (expected for climate/register flags without a TMR code). "
            "Before modelling: exclude these flags or extend the mapping.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
