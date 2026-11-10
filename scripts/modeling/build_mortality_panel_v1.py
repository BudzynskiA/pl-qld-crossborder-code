"""
Builds the comparable sample v1 for the fatality model (y_morti).

Input:
  <PLQLD_DATA_DIR>/event_level/event_level_pl_qld_2015_2024.parquet

Outputs:
  <PLQLD_DATA_DIR>/event_level/event_level_pl_qld_mortality_v1.parquet
  tables_md/mortality_panel_v1.md
  tables_csv/mortality_panel_v1_by_year.csv

v1 filters:
  PL: no extra filter (already enriched: injured or killed)
  QLD: feriti_count > 0 OR y_morti = 1; exclude rows where the sum of lok_* = 0

Run from the repository root:
  python scripts/modeling/build_mortality_panel_v1.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))
from _bootstrap import HARMONIZED_EVENT_LEVEL_DIR, TABLES_CSV, TABLES_MD, display_path, install  # noqa: E402

install()

import polars as pl  # noqa: E402

from harmonization.lokalizacja_flags import LOKALIZACJA_FLAGS  # noqa: E402
from harmonization.outcome_morti import FERITI_COUNT, Y_MORTI, assert_outcome_morti  # noqa: E402

FULL_PANEL = HARMONIZED_EVENT_LEVEL_DIR / "event_level_pl_qld_2015_2024.parquet"
OUT_PANEL = HARMONIZED_EVENT_LEVEL_DIR / "event_level_pl_qld_mortality_v1.parquet"
OUT_MD = TABLES_MD / "mortality_panel_v1.md"
OUT_YEAR_CSV = TABLES_CSV / "mortality_panel_v1_by_year.csv"


def _lok_sum_expr() -> pl.Expr:
    return pl.sum_horizontal([pl.col(c) for c in LOKALIZACJA_FLAGS])


def _apply_v1_filters(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    lok_sum = _lok_sum_expr()
    pl_df = df.filter(pl.col("kraj") == "PL")
    qld_full = df.filter(pl.col("kraj") == "QLD")

    qld_casualty = qld_full.filter((pl.col(FERITI_COUNT) > 0) | (pl.col(Y_MORTI) == 1))
    qld_v1 = qld_casualty.filter(lok_sum > 0)

    stats = {
        "n_pl": pl_df.height,
        "n_qld_full": qld_full.height,
        "n_qld_casualty": qld_casualty.height,
        "n_qld_lok0_dropped": qld_casualty.filter(lok_sum == 0).height,
        "n_qld_v1": qld_v1.height,
        "n_total": pl_df.height + qld_v1.height,
    }
    panel = pl.concat([pl_df, qld_v1], how="diagonal")
    return panel, stats


def _prev_by_country(df: pl.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for kraj in ("PL", "QLD"):
        sub = df.filter(pl.col("kraj") == kraj)
        n = sub.height
        n_pos = sub.filter(pl.col(Y_MORTI) == 1).height if n else 0
        rows.append(
            {
                "kraj": kraj,
                "n": n,
                "n_y_morti_1": n_pos,
                "prev_y_morti": round(n_pos / n, 6) if n else 0.0,
                "mean_feriti_count": round(sub[FERITI_COUNT].mean() or 0.0, 4),
                "mean_morti_count": round(sub["morti_count"].mean() or 0.0, 4),
            }
        )
    return rows


def _by_year(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("kraj", "rok")
        .agg(
            pl.len().alias("n"),
            (pl.col(Y_MORTI) == 1).sum().alias("n_y_morti_1"),
            pl.col(FERITI_COUNT).mean().alias("mean_feriti_count"),
            pl.col("morti_count").mean().alias("mean_morti_count"),
        )
        .with_columns((pl.col("n_y_morti_1") / pl.col("n")).round(6).alias("prev_y_morti"))
        .sort("kraj", "rok")
    )


def _write_md(
    country_rows: list[dict],
    stats: dict,
    *,
    full_n_pl: int,
    full_n_qld: int,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Comparable sample v1: fatality (`y_morti`)",
        "",
        f"**Generated:** {ts} (`scripts/modeling/build_mortality_panel_v1.py`)",
        "",
        "**Full panel:** `<PLQLD_DATA_DIR>/event_level/event_level_pl_qld_2015_2024.parquet`",
        "",
        "**Sample v1:** `<PLQLD_DATA_DIR>/event_level/event_level_pl_qld_mortality_v1.parquet`",
        "",
        "## Definition of sample v1",
        "",
        "| Country | Rule |",
        "|------|--------|",
        f"| PL | no filter ({full_n_pl:,} rows, enriched SEWIK) |",
        f"| QLD | `{FERITI_COUNT} > 0` OR `{Y_MORTI} = 1`; exclude rows where the sum of `lok_*` = 0 |",
        "",
        f"Full QLD TMR (after excluding PD-only in derived): **{full_n_qld:,}** "
        f"→ after the casualty filter: **{stats['n_qld_casualty']:,}** "
        f"→ after excluding lok=0: **{stats['n_qld_v1']:,}** "
        f"(dropped lok=0: {stats['n_qld_lok0_dropped']:,}).",
        "",
        f"**Total v1:** {stats['n_total']:,} rows (PL {stats['n_pl']:,} + QLD {stats['n_qld_v1']:,}).",
        "",
        "## Prevalence, full window 2015-2024",
        "",
        "| kraj | n | n(y=1) | prev y_morti | mean feriti_count | mean morti_count |",
        "|------|--:|-------:|-------------:|------------------:|-------------------:|",
    ]
    for r in country_rows:
        lines.append(
            f"| {r['kraj']} | {r['n']:,} | {r['n_y_morti_1']:,} | {r['prev_y_morti']:.4f} | "
            f"{r['mean_feriti_count']:.4f} | {r['mean_morti_count']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Yearly CSV",
            "",
            f"[`../tables_csv/mortality_panel_v1_by_year.csv`](../tables_csv/mortality_panel_v1_by_year.csv)",
            "",
            "## Models",
            "",
            "CatBoost / logit on this sample (20 retained flags: 19 OK + `typ_inne` WARN; FAIL flags from the leakage audit excluded): "
            "`scripts/modeling/run_morti_crossborder_catboost.py`.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not FULL_PANEL.is_file():
        raise SystemExit(f"Panel not found: {FULL_PANEL}")

    full = pl.read_parquet(FULL_PANEL)
    if Y_MORTI not in full.columns:
        raise SystemExit(f"Missing {Y_MORTI}; run build_event_level_pl_qld_harmonized.py")

    full_n_pl = full.filter(pl.col("kraj") == "PL").height
    full_n_qld = full.filter(pl.col("kraj") == "QLD").height

    assert_outcome_morti(full.filter(pl.col("kraj") == "PL"), "PL")
    assert_outcome_morti(full.filter(pl.col("kraj") == "QLD"), "QLD")

    panel, stats = _apply_v1_filters(full)
    country_rows = _prev_by_country(panel)
    year_df = _by_year(panel)

    HARMONIZED_EVENT_LEVEL_DIR.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(OUT_PANEL, compression="snappy")

    TABLES_CSV.mkdir(parents=True, exist_ok=True)
    year_df.write_csv(OUT_YEAR_CSV)
    _write_md(country_rows, stats, full_n_pl=full_n_pl, full_n_qld=full_n_qld)

    for r in country_rows:
        print(
            f"{r['kraj']}: n={r['n']:,} prev_y_morti={r['prev_y_morti']:.4f} "
            f"mean_feriti={r['mean_feriti_count']:.3f}"
        )
    print(f"Panel v1: {stats['n_total']:,} -> {display_path(OUT_PANEL)}")
    print(f"Report: {display_path(OUT_MD)}")


if __name__ == "__main__":
    main()
