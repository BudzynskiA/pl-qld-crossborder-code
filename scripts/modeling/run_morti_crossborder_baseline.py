"""
Cross-border baseline: y_morti (fatal yes/no) on harmonized 0/1 flags.

Input (sample v1 by default):
  <PLQLD_DATA_DIR>/event_level/event_level_pl_qld_mortality_v1.parquet
  (--panel full -> event_level_pl_qld_2015_2024.parquet)

Outputs:
  tables_md/morti_crossborder_baseline.md
  tables_csv/morti_crossborder_baseline_metrics.csv

Experiments (temporal split: train year <= 2022, test year >= 2023):
  train PL -> test QLD, train QLD -> test PL, train PL -> test PL, train QLD -> test QLD

Run from the repository root:
  python scripts/modeling/run_morti_crossborder_baseline.py
  python scripts/modeling/run_morti_crossborder_baseline.py --panel full
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

from harmonization.outcome_morti import Y_MORTI  # noqa: E402
from modeling._morti_crossborder_common import (  # noqa: E402
    EXPERIMENTS,
    FAIL_FLAGS_EXCLUDED,
    FEATURE_FLAGS,
    TEST_YEAR_MIN,
    TRAIN_YEAR_MAX,
    load_panel,
    panel_specs,
    train_test,
)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
except ImportError as exc:
    raise SystemExit("pip install scikit-learn") from exc

OUT_MD = TABLES_MD / "morti_crossborder_baseline.md"
OUT_CSV = TABLES_CSV / "morti_crossborder_baseline_metrics.csv"


def _fit_eval(train: pl.DataFrame, test: pl.DataFrame) -> dict:
    x_train = train.select(FEATURE_FLAGS).to_numpy()
    y_train = train[Y_MORTI].to_numpy()
    x_test = test.select(FEATURE_FLAGS).to_numpy()
    y_test = test[Y_MORTI].to_numpy()

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]

    n_test = len(y_test)
    n_pos = int(y_test.sum())
    prev = n_pos / n_test if n_test else 0.0

    out: dict = {
        "n_train": train.height,
        "n_test": n_test,
        "n_pos_test": n_pos,
        "prev_test": round(prev, 6),
        "brier": round(float(brier_score_loss(y_test, proba)), 6),
    }
    if n_pos > 0 and n_pos < n_test:
        out["auc_roc"] = round(float(roc_auc_score(y_test, proba)), 6)
    else:
        out["auc_roc"] = None
    return out


def _write_md(rows: list[dict], *, panel: str, panel_path: Path, drop_lok_unassigned: bool) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rel_panel = display_path(panel_path)
    filt = "yes (without QLD lok=0)" if drop_lok_unassigned else "no (panel already filtered)" if panel == "v1" else "no"
    v1_link = (
        "\nDefinition of sample v1: [`mortality_panel_v1.md`](mortality_panel_v1.md).\n"
        if panel == "v1"
        else ""
    )
    excluded = ", ".join(f"`{f}`" for f in FAIL_FLAGS_EXCLUDED)
    lines = [
        "# Baseline cross-border: `y_morti` (logistic regression)",
        "",
        f"**Generated:** {ts} (`scripts/modeling/run_morti_crossborder_baseline.py`)",
        "",
        f"**Panel ({panel}):** `{rel_panel}`",
        "",
        f"**Y:** `{Y_MORTI}` = 1 when `morti_count` > 0 (PL `ilosc_zmarlych`; QLD `morti_count`).",
        f"**X:** {len(FEATURE_FLAGS)} OK harmonized flags (no `kraj`, no `rok`).",
        f"**Excluded (audit FAIL):** {excluded}.",
        f"**Split:** train year <= {TRAIN_YEAR_MAX}, test year >= {TEST_YEAR_MIN}.",
        f"**Drop lok unassigned:** {filt}.",
        v1_link,
        "| experiment | n_train | n_test | prev_test | AUC-ROC | Brier |",
        "|-------------|--------:|-------:|----------:|--------:|------:|",
    ]
    for r in rows:
        auc = f"{r['auc_roc']:.4f}" if r["auc_roc"] is not None else "-"
        lines.append(
            f"| {r['experiment']} | {r['n_train']:,} | {r['n_test']:,} | "
            f"{r['prev_test']:.4f} | {auc} | {r['brier']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## CSV",
            "",
            f"[`../tables_csv/morti_crossborder_baseline_metrics.csv`](../tables_csv/morti_crossborder_baseline_metrics.csv)",
            "",
            "Calibration (ECE, reliability): [`morti_crossborder_calibration.md`](morti_crossborder_calibration.md).",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Baseline y_morti cross-border PL<->QLD")
    parser.add_argument(
        "--panel",
        choices=("v1", "full"),
        default="v1",
        help="v1 = comparable sample (default); full = full harmonized panel",
    )
    parser.add_argument(
        "--drop-lok-unassigned",
        action="store_true",
        help="Only with --panel full: exclude QLD rows where the sum of lok_* = 0",
    )
    args = parser.parse_args()

    spec = panel_specs(HARMONIZED_EVENT_LEVEL_DIR)[args.panel]
    if not spec.path.is_file():
        hint = "build_mortality_panel_v1.py" if args.panel == "v1" else "build_event_level_pl_qld_harmonized.py"
        raise SystemExit(f"Panel not found: {spec.path}\nRun: python scripts/modeling/{hint}")

    drop_lok = args.drop_lok_unassigned and args.panel == "full"
    df = load_panel(spec.path, drop_lok_unassigned=drop_lok)
    if Y_MORTI not in df.columns:
        raise SystemExit(f"Missing {Y_MORTI} in the panel")

    rows: list[dict] = []
    for name, train_kraj, test_kraj in EXPERIMENTS:
        train, test = train_test(df, train_kraj, test_kraj)
        metrics = _fit_eval(train, test)
        rows.append({"experiment": name, "panel": args.panel, **metrics})

    TABLES_CSV.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(OUT_CSV)
    _write_md(rows, panel=args.panel, panel_path=spec.path, drop_lok_unassigned=drop_lok)

    for r in rows:
        auc = r["auc_roc"] if r["auc_roc"] is not None else "n/a"
        print(f"{r['experiment']}: n_test={r['n_test']:,} prev={r['prev_test']:.4f} AUC={auc} Brier={r['brier']:.4f}")
    print(f"Panel: {args.panel} -> {display_path(spec.path)}")
    print(f"Report: {display_path(OUT_MD)}")


if __name__ == "__main__":
    main()
