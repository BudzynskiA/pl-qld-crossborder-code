"""
Country leakage audit for harmonized 0/1 flags (cross-border PL ↔ QLD).

Checks whether a flag could act as a country proxy (e.g. =1 on one side only).

Statuses:
  OK   - n(=1) > 0 in both PL and QLD
  WARN - both sides have =1, but the frequencies are extremely asymmetric
  FAIL - n(=1) == 0 in PL or QLD (the flag separates countries at =1)

WARN thresholds (defaults):
  min(n_pl_1, n_qld_1) < 25
  max(p_pl, p_qld) / min(p_pl, p_qld) > 30  (when min(p) >= 1e-6)
"""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

DEFAULT_MIN_COUNT_WARN = 25
DEFAULT_RATE_RATIO_WARN = 30.0


def _flag_stats(df: pl.DataFrame, flag: str) -> tuple[int, float]:
    n = df.height
    n1 = df.filter(pl.col(flag) == 1).height if n else 0
    return n1, (n1 / n if n else 0.0)


def audit_flag_country_leakage(
    pl_df: pl.DataFrame,
    qld_df: pl.DataFrame,
    flag_groups: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    min_count_warn: int = DEFAULT_MIN_COUNT_WARN,
    rate_ratio_warn: float = DEFAULT_RATE_RATIO_WARN,
) -> pl.DataFrame:
    rows: list[dict] = []
    n_pl = pl_df.height
    n_qld = qld_df.height

    for cecha, flags in flag_groups:
        for flag in flags:
            n_pl_1, p_pl = _flag_stats(pl_df, flag)
            n_qld_1, p_qld = _flag_stats(qld_df, flag)

            if n_pl_1 == 0 or n_qld_1 == 0:
                status = "FAIL"
                if n_pl_1 == 0 and n_qld_1 == 0:
                    note = "no =1 on either side (dead feature)"
                elif n_pl_1 == 0:
                    note = "=1 in QLD only (structural leakage)"
                else:
                    note = "=1 in PL only (structural leakage / no TMR code)"
            else:
                note = ""
                min_n = min(n_pl_1, n_qld_1)
                lo, hi = min(p_pl, p_qld), max(p_pl, p_qld)
                rate_ratio = (hi / lo) if lo >= 1e-6 else float("inf")

                if min_n < min_count_warn or rate_ratio > rate_ratio_warn:
                    status = "WARN"
                    parts: list[str] = []
                    if min_n < min_count_warn:
                        parts.append(f"few =1 observations (min {min_n})")
                    if rate_ratio > rate_ratio_warn:
                        parts.append(f"p asymmetry: ratio {rate_ratio:.1f}")
                    note = "; ".join(parts)
                else:
                    status = "OK"

            rows.append(
                {
                    "cecha": cecha,
                    "flaga": flag,
                    "n_panel_pl": n_pl,
                    "n_panel_qld": n_qld,
                    "n_1_pl": n_pl_1,
                    "n_1_qld": n_qld_1,
                    "p_1_pl": round(p_pl, 6),
                    "p_1_qld": round(p_qld, 6),
                    "rate_ratio": round(max(p_pl, p_qld) / min(p_pl, p_qld), 2)
                    if min(p_pl, p_qld) >= 1e-6 and n_pl_1 > 0 and n_qld_1 > 0
                    else None,
                    "status": status,
                    "uwagi": note,
                }
            )

    return pl.DataFrame(rows)


def write_leakage_audit_md(
    audit_df: pl.DataFrame,
    path,
    *,
    script_rel: str = "scripts/harmonization/build_event_level_pl_qld_harmonized.py",
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_fail = audit_df.filter(pl.col("status") == "FAIL").height
    n_warn = audit_df.filter(pl.col("status") == "WARN").height
    n_ok = audit_df.filter(pl.col("status") == "OK").height

    lines = [
        "# Country leakage audit: harmonized flags PL ↔ QLD",
        "",
        f"**Generated:** {ts} (`{script_rel}` + `country_leakage_audit.py`)",
        "",
        "Goal: flags used in the cross-border model must not be `=1` on one side only.",
        "Different **frequencies** of the same flag between countries are OK; a country separator at `=1` is FAIL.",
        "FAIL is expected for climate/register categories absent from TMR (snow, ice, railway).",
        "",
        f"**Summary:** OK={n_ok}, WARN={n_warn}, FAIL={n_fail}",
        "",
    ]

    if n_fail:
        lines.extend(
            [
                "## FAIL (requires a mapping fix or excluding the flag from the model)",
                "",
                "| feature | flag | n(=1) PL | n(=1) QLD | notes |",
                "|-------|-------|----------:|----------:|-------|",
            ]
        )
        for row in audit_df.filter(pl.col("status") == "FAIL").iter_rows(named=True):
            lines.append(
                f"| {row['cecha']} | `{row['flaga']}` | {row['n_1_pl']:,} | {row['n_1_qld']:,} | {row['uwagi']} |"
            )
        lines.append("")

    if n_warn:
        lines.extend(
            [
                "## WARN (both sides have =1, use with care in the model)",
                "",
                "| feature | flag | p(=1) PL | p(=1) QLD | ratio | notes |",
                "|-------|-------|--------:|--------:|------:|-------|",
            ]
        )
        for row in audit_df.filter(pl.col("status") == "WARN").iter_rows(named=True):
            rr = row["rate_ratio"]
            rr_s = f"{rr:.1f}" if rr is not None else "-"
            lines.append(
                f"| {row['cecha']} | `{row['flaga']}` | {row['p_1_pl']:.4f} | {row['p_1_qld']:.4f} | {rr_s} | {row['uwagi']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## All flags",
            "",
            "| feature | flag | n(=1) PL | n(=1) QLD | p(=1) PL | p(=1) QLD | status |",
            "|-------|-------|----------:|----------:|---------:|---------:|--------|",
        ]
    )
    for row in audit_df.sort(["status", "cecha", "flaga"]).iter_rows(named=True):
        lines.append(
            f"| {row['cecha']} | `{row['flaga']}` | {row['n_1_pl']:,} | {row['n_1_qld']:,} | "
            f"{row['p_1_pl']:.4f} | {row['p_1_qld']:.4f} | **{row['status']}** |"
        )

    lines.extend(
        [
            "",
            "## CSV",
            "",
            "[`../tables_csv/harmonized_flags_country_leakage_audit.csv`](../tables_csv/harmonized_flags_country_leakage_audit.csv)",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def assert_no_structural_country_leakage(audit_df: pl.DataFrame) -> None:
    fails = audit_df.filter(pl.col("status") == "FAIL")
    if fails.is_empty():
        return
    detail = ", ".join(
        f"{r['flaga']} ({r['uwagi']})" for r in fails.select(["flaga", "uwagi"]).iter_rows(named=True)
    )
    raise ValueError(f"Country leakage audit: FAIL for flags: {detail}")


def audit_group_unassigned_leakage(
    pl_df: pl.DataFrame,
    qld_df: pl.DataFrame,
    group_name: str,
    flags: tuple[str, ...],
) -> dict:
    """Rows with flag sum = 0 in the group; WARN when the pattern occurs in one country only."""
    n_zero_pl = pl_df.filter(pl.sum_horizontal([pl.col(c) for c in flags]) == 0).height if pl_df.height else 0
    n_zero_qld = qld_df.filter(pl.sum_horizontal([pl.col(c) for c in flags]) == 0).height if qld_df.height else 0
    p_zero_pl = n_zero_pl / pl_df.height if pl_df.height else 0.0
    p_zero_qld = n_zero_qld / qld_df.height if qld_df.height else 0.0

    if n_zero_pl == 0 and n_zero_qld == 0:
        status = "OK"
        note = "no rows without an assigned category"
    elif n_zero_pl == 0 and n_zero_qld > 0:
        status = "WARN"
        note = "flag sum = 0 in QLD only (possible country proxy)"
    elif n_zero_qld == 0 and n_zero_pl > 0:
        status = "WARN"
        note = "flag sum = 0 in PL only (possible country proxy)"
    else:
        status = "OK"
        note = "both countries have rows without an assigned category"

    return {
        "cecha": group_name,
        "flaga": "(suma=0)",
        "n_panel_pl": pl_df.height,
        "n_panel_qld": qld_df.height,
        "n_1_pl": n_zero_pl,
        "n_1_qld": n_zero_qld,
        "p_1_pl": round(p_zero_pl, 6),
        "p_1_qld": round(p_zero_qld, 6),
        "rate_ratio": None,
        "status": status,
        "uwagi": note,
    }
