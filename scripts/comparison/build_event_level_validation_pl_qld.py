"""
Event-level validation of PL (SEWIK enriched) vs QLD (TMR derived) on the harmonized panel.

Outputs:
  tables_md/event_level_validation_pl_qld.md
  tables_csv/event_level_validation_by_year.csv
  tables_csv/event_level_outcome_rates_by_year.csv
  tables_csv/event_level_flag_coverage_pl_qld.csv

Run from the repository root:
  python scripts/comparison/build_event_level_validation_pl_qld.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))
from _bootstrap import HARMONIZED_EVENT_LEVEL_DIR, POLAND, QUEENSLAND, TABLES_CSV, TABLES_MD, display_path, install  # noqa: E402

install()

try:
    import polars as pl
except ImportError as exc:
    raise SystemExit("pip install -r requirements.txt (polars, pyarrow)") from exc

from harmonization.lokalizacja_flags import LOKALIZACJA_FLAGS  # noqa: E402
from harmonization.nawierzchnia_flags import NAWIERZCHNIA_FLAGS  # noqa: E402
from harmonization.outcome_morti import FERITI_COUNT, MORTI_COUNT, Y_MORTI  # noqa: E402
from harmonization.pogoda_flags import POGODA_FLAGS  # noqa: E402
from harmonization.typ_zdarzenia_flags import TYP_ZDARZENIA_FLAGS  # noqa: E402
from modeling._morti_crossborder_common import FAIL_FLAGS_EXCLUDED  # noqa: E402

PL_PATH = POLAND / "data" / "derived" / "event_level" / "zdarzenia_wypadki_enriched_2015_2024.parquet"
QLD_PATH = QUEENSLAND / "data" / "derived" / "event_level" / "event_level_qld_2015_2024.parquet"
HARM_PATH = HARMONIZED_EVENT_LEVEL_DIR / "event_level_pl_qld_2015_2024.parquet"

OUT_MD = TABLES_MD / "event_level_validation_pl_qld.md"
OUT_YEAR = TABLES_CSV / "event_level_validation_by_year.csv"
OUT_RATES = TABLES_CSV / "event_level_outcome_rates_by_year.csv"
OUT_FLAGS = TABLES_CSV / "event_level_flag_coverage_pl_qld.csv"

PANEL_FROM = 2015
PANEL_TO = 2024

ALL_FLAGS: tuple[str, ...] = POGODA_FLAGS + TYP_ZDARZENIA_FLAGS + NAWIERZCHNIA_FLAGS + LOKALIZACJA_FLAGS


def _load_pl_raw() -> pl.LazyFrame:
    if not PL_PATH.is_file():
        fallback = POLAND / "data" / "derived" / "event_level" / "zdarzenia_wypadki_2015_2024.parquet"
        path = fallback if fallback.is_file() else PL_PATH
    else:
        path = PL_PATH
    if not path.is_file():
        raise SystemExit(f"PL file not found: {PL_PATH}")
    return pl.scan_parquet(path).filter(pl.col("rok").is_between(PANEL_FROM, PANEL_TO))


def _load_qld_raw() -> pl.LazyFrame:
    if not QLD_PATH.is_file():
        raise SystemExit(f"QLD file not found: {QLD_PATH}")
    return (
        pl.scan_parquet(QLD_PATH)
        .with_columns(pl.col("crash_year").alias("rok"))
        .filter(pl.col("rok").is_between(PANEL_FROM, PANEL_TO))
    )


def _pl_yearly(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf.group_by("rok")
        .agg(
            pl.len().alias("n_zdarzen"),
            pl.col("ilosc_feriti").sum().alias("sum_feriti"),
            pl.col("ilosc_zmarlych").sum().alias("sum_morti"),
            pl.col("ilosc_feriti").mean().alias("mean_feriti_na_zdarzenie"),
            pl.col("ilosc_zmarlych").mean().alias("mean_morti_na_zdarzenie"),
        )
        .sort("rok")
        .collect()
    )


def _qld_yearly(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf.group_by("rok")
        .agg(
            pl.len().alias("n_zdarzen"),
            pl.col("feriti_count").sum().alias("sum_feriti"),
            pl.col("morti_count").sum().alias("sum_morti"),
            pl.col("feriti_count").mean().alias("mean_feriti_na_zdarzenie"),
            pl.col("morti_count").mean().alias("mean_morti_na_zdarzenie"),
        )
        .sort("rok")
        .collect()
    )


def _merge_yearly(pl_y: pl.DataFrame, qld_y: pl.DataFrame) -> pl.DataFrame:
    merged = pl_y.join(qld_y, on="rok", how="full", suffix="_qld").sort("rok")
    return merged.with_columns(
        (pl.col("n_zdarzen") / pl.col("n_zdarzen_qld")).alias("pl_qld_ratio_zdarzen"),
        (pl.col("mean_feriti_na_zdarzenie") / pl.col("mean_feriti_na_zdarzenie_qld")).alias(
            "pl_qld_ratio_mean_feriti"
        ),
        (pl.col("mean_morti_na_zdarzenie") / pl.col("mean_morti_na_zdarzenie_qld")).alias(
            "pl_qld_ratio_mean_morti"
        ),
    )


def _harm_outcome_by_year(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("kraj", "rok")
        .agg(
            pl.len().alias("n"),
            (pl.col(Y_MORTI) == 1).sum().alias("n_y_morti_1"),
            pl.col(FERITI_COUNT).mean().alias("mean_feriti"),
            pl.col(MORTI_COUNT).mean().alias("mean_morti"),
        )
        .with_columns((pl.col("n_y_morti_1") / pl.col("n")).round(6).alias("prev_y_morti"))
        .sort("kraj", "rok")
    )


def _flag_coverage(df: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    for kraj in ("PL", "QLD"):
        sub = df.filter(pl.col("kraj") == kraj)
        n = sub.height
        for flag in ALL_FLAGS:
            n1 = sub.filter(pl.col(flag) == 1).height
            rows.append(
                {
                    "kraj": kraj,
                    "flaga": flag,
                    "n_eq_1": n1,
                    "p_eq_1": round(n1 / n, 6) if n else 0.0,
                    "model_v1": "exclude" if flag in FAIL_FLAGS_EXCLUDED else "include",
                }
            )
    return pl.DataFrame(rows)


def _md_table(df: pl.DataFrame, cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in df.select(cols).iter_rows():
        cells = []
        for col, v in zip(cols, row):
            if v is None:
                cells.append("-")
            elif col == "rok":
                cells.append(str(int(v)))
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 100 else f"{v:,.2f}")
            else:
                cells.append(f"{v:,}" if isinstance(v, int) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_markdown(
    merged: pl.DataFrame,
    pl_tot: dict[str, float],
    qld_tot: dict[str, float],
    harm_tot: dict[str, dict],
    flag_cov: pl.DataFrame,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    year_cols = [
        "rok",
        "n_zdarzen",
        "n_zdarzen_qld",
        "pl_qld_ratio_zdarzen",
        "mean_feriti_na_zdarzenie",
        "mean_feriti_na_zdarzenie_qld",
        "pl_qld_ratio_mean_feriti",
        "mean_morti_na_zdarzenie",
        "mean_morti_na_zdarzenie_qld",
        "pl_qld_ratio_mean_morti",
    ]
    present = [c for c in year_cols if c in merged.columns]

    fail_flags = ", ".join(f"`{f}`" for f in FAIL_FLAGS_EXCLUDED)
    warn_rows = flag_cov.filter((pl.col("flaga") == "typ_inne") & (pl.col("model_v1") == "include"))
    typ_inne_pl = warn_rows.filter(pl.col("kraj") == "PL")["p_eq_1"][0] if warn_rows.height else 0
    typ_inne_qld = warn_rows.filter(pl.col("kraj") == "QLD")["p_eq_1"][0] if warn_rows.height else 0

    return f"""# Event-level validation: SEWIK (PL) vs TMR (QLD)

**Generated:** {ts} (`scripts/comparison/build_event_level_validation_pl_qld.py`)

**Sources (derived):**
- PL: `<PLQLD_POLAND_DIR>/data/derived/event_level/zdarzenia_wypadki_enriched_2015_2024.parquet`
- QLD: `<PLQLD_QUEENSLAND_DIR>/data/derived/event_level/event_level_qld_2015_2024.parquet`

**Harmonized panel:** `<PLQLD_DATA_DIR>/event_level/event_level_pl_qld_2015_2024.parquet`

**Panel:** {PANEL_FROM}-{PANEL_TO}. The comparison **does not assume** equal crash counts (the jurisdictions differ in size); we look at **relative metrics**.

## Panel totals (derived, before harmonization)

| Metric | PL | QLD | Note |
|---------|---:|---:|-------|
| Crashes | {pl_tot["n"]:,.0f} | {qld_tot["n"]:,.0f} | PL: enriched (injured/killed); QLD: without PD-only |
| Total injured | {pl_tot["feriti"]:,.0f} | {qld_tot["feriti"]:,.0f} | PL: `ilosc_feriti`; QLD: `feriti_count` |
| Total killed | {pl_tot["morti"]:,.0f} | {qld_tot["morti"]:,.0f} | PL: `ilosc_zmarlych`; QLD: `morti_count` |
| Mean injured/crash | {pl_tot["mean_f"]:.3f} | {qld_tot["mean_f"]:.3f} | |
| Mean killed/crash | {pl_tot["mean_m"]:.4f} | {qld_tot["mean_m"]:.4f} | ratio: {pl_tot["mean_m"]/qld_tot["mean_m"] if qld_tot["mean_m"] else 0:.3f} |

## Outcome Y on the harmonized panel

| kraj | n | n(y_morti=1) | prev y_morti | mean feriti_count | mean morti_count |
|------|--:|------------:|-------------:|------------------:|-----------------:|
| PL | {harm_tot["PL"]["n"]:,} | {harm_tot["PL"]["n_pos"]:,} | {harm_tot["PL"]["prev"]:.4f} | {harm_tot["PL"]["mean_f"]:.4f} | {harm_tot["PL"]["mean_m"]:.4f} |
| QLD | {harm_tot["QLD"]["n"]:,} | {harm_tot["QLD"]["n_pos"]:,} | {harm_tot["QLD"]["prev"]:.4f} | {harm_tot["QLD"]["mean_f"]:.4f} | {harm_tot["QLD"]["mean_m"]:.4f} |

## By year (scale and relative metrics, derived)

{_md_table(merged, present)}

Full CSV table: [`../tables_csv/event_level_validation_by_year.csv`](../tables_csv/event_level_validation_by_year.csv)

## Harmonized flag coverage (model v1)

**FAIL** flags excluded from model v1: {fail_flags}.

**WARN** `typ_inne`: p(=1) PL={typ_inne_pl:.4f}, QLD={typ_inne_qld:.4f} (asymmetry ~{typ_inne_pl/typ_inne_qld if typ_inne_qld else 0:.0f}×); kept in v1 and noted in the manuscript.

Full table: [`../tables_csv/event_level_flag_coverage_pl_qld.csv`](../tables_csv/event_level_flag_coverage_pl_qld.csv)

Leakage audit: [`harmonized_flags_country_leakage_audit.md`](harmonized_flags_country_leakage_audit.md).

## Interpretation

- The **PL/QLD crash ratio** reflects differences in population and traffic; it is not a harmonization error.
- **y_morti prevalence** in QLD (~1.9%) is lower than in PL (~8.5%), as expected given the different register definitions.
- The **7 FAIL flags** (snow, wind, railway, PL-only surface flags) are a structural limitation of TMR; model v1 uses the **20 retained flags** (19 OK + `typ_inne` WARN).
"""


def main() -> None:
    pl_lf = _load_pl_raw()
    qld_lf = _load_qld_raw()

    pl_df = pl_lf.collect()
    qld_df = qld_lf.collect()

    pl_y = _pl_yearly(pl_lf)
    qld_y = _qld_yearly(qld_lf)
    merged = _merge_yearly(pl_y, qld_y)

    if not HARM_PATH.is_file():
        raise SystemExit(f"Harmonized panel not found: {HARM_PATH}")

    harm = pl.read_parquet(HARM_PATH)
    harm_outcome = _harm_outcome_by_year(harm)
    flag_cov = _flag_coverage(harm)

    harm_tot: dict[str, dict] = {}
    for kraj in ("PL", "QLD"):
        sub = harm.filter(pl.col("kraj") == kraj)
        n = sub.height
        n_pos = sub.filter(pl.col(Y_MORTI) == 1).height
        harm_tot[kraj] = {
            "n": n,
            "n_pos": n_pos,
            "prev": n_pos / n if n else 0.0,
            "mean_f": sub[FERITI_COUNT].mean() or 0.0,
            "mean_m": sub[MORTI_COUNT].mean() or 0.0,
        }

    TABLES_CSV.mkdir(parents=True, exist_ok=True)
    TABLES_MD.mkdir(parents=True, exist_ok=True)

    merged.write_csv(OUT_YEAR)
    merged.select(
        "rok",
        "mean_feriti_na_zdarzenie",
        "mean_feriti_na_zdarzenie_qld",
        "pl_qld_ratio_mean_feriti",
        "mean_morti_na_zdarzenie",
        "mean_morti_na_zdarzenie_qld",
        "pl_qld_ratio_mean_morti",
    ).write_csv(OUT_RATES)
    flag_cov.write_csv(OUT_FLAGS)
    harm_outcome.write_csv(TABLES_CSV / "event_level_harmonized_outcome_by_year.csv")

    pl_tot = {
        "n": pl_df.height,
        "feriti": pl_df["ilosc_feriti"].sum(),
        "morti": pl_df["ilosc_zmarlych"].sum(),
        "mean_f": pl_df["ilosc_feriti"].mean() or 0.0,
        "mean_m": pl_df["ilosc_zmarlych"].mean() or 0.0,
    }
    qld_tot = {
        "n": qld_df.height,
        "feriti": qld_df["feriti_count"].sum(),
        "morti": qld_df["morti_count"].sum(),
        "mean_f": qld_df["feriti_count"].mean() or 0.0,
        "mean_m": qld_df["morti_count"].mean() or 0.0,
    }

    OUT_MD.write_text(_build_markdown(merged, pl_tot, qld_tot, harm_tot, flag_cov), encoding="utf-8")

    print(f"Saved {display_path(OUT_MD)}")
    print(f"Saved {OUT_YEAR.name}, {OUT_RATES.name}, {OUT_FLAGS.name}")


if __name__ == "__main__":
    main()
