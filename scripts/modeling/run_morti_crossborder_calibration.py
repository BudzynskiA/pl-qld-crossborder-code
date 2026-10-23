"""
Calibration of the y_morti models on sample v1: logit + CatBoost.

Input:
  <PLQLD_DATA_DIR>/event_level/event_level_pl_qld_mortality_v1.parquet

Outputs:
  tables_md/morti_crossborder_calibration.md
  tables_csv/morti_crossborder_calibration_metrics.csv
  tables_csv/morti_crossborder_calibration_reliability.csv

Run from the repository root:
  python scripts/modeling/run_morti_crossborder_calibration.py
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
    CALIBRATION_BINS,
    EXPERIMENTS,
    FEATURE_FLAGS,
    TEST_YEAR_MIN,
    TRAIN_YEAR_MAX,
    calibration_metrics,
    load_panel,
    panel_specs,
    reliability_table,
    train_test,
)

try:
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    raise SystemExit("pip install -r requirements.txt") from exc

OUT_MD = TABLES_MD / "morti_crossborder_calibration.md"
OUT_METRICS_CSV = TABLES_CSV / "morti_crossborder_calibration_metrics.csv"
OUT_REL_CSV = TABLES_CSV / "morti_crossborder_calibration_reliability.csv"

CATBOOST_ITERATIONS = 500


def _fit_logit_proba(train: pl.DataFrame, test: pl.DataFrame):
    x_train = train.select(FEATURE_FLAGS).to_numpy()
    y_train = train[Y_MORTI].to_numpy()
    x_test = test.select(FEATURE_FLAGS).to_numpy()
    y_test = test[Y_MORTI].to_numpy()
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    return y_test, model.predict_proba(x_test)[:, 1]


def _fit_catboost_proba(train: pl.DataFrame, test: pl.DataFrame):
    x_train = train.select(FEATURE_FLAGS).to_numpy()
    y_train = train[Y_MORTI].to_numpy()
    x_test = test.select(FEATURE_FLAGS).to_numpy()
    y_test = test[Y_MORTI].to_numpy()
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=CATBOOST_ITERATIONS,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )
    model.fit(x_train, y_train, eval_set=(x_test, y_test), use_best_model=True)
    return y_test, model.predict_proba(x_test)[:, 1]


def _eval_row(
    *,
    model: str,
    experiment: str,
    y_test,
    proba,
) -> dict:
    n_pos = int(y_test.sum())
    n_test = len(y_test)
    cal = calibration_metrics(y_test, proba)
    row = {
        "model": model,
        "experiment": experiment,
        "n_test": n_test,
        "n_pos_test": n_pos,
        "prev_test": round(n_pos / n_test, 6) if n_test else 0.0,
        **cal,
    }
    if n_pos > 0 and n_pos < n_test:
        row["auc_roc"] = round(float(roc_auc_score(y_test, proba)), 6)
    else:
        row["auc_roc"] = None
    return row


def _write_md(metric_rows: list[dict]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Cross-border calibration: `y_morti` (sample v1)",
        "",
        f"**Generated:** {ts} (`scripts/modeling/run_morti_crossborder_calibration.py`)",
        "",
        "**Panel:** `<PLQLD_DATA_DIR>/event_level/event_level_pl_qld_mortality_v1.parquet`",
        "",
        f"**Split:** train year <= {TRAIN_YEAR_MAX}, test year >= {TEST_YEAR_MIN}.",
        f"**Reliability binning:** {CALIBRATION_BINS} quantile bins (ECE = weighted mean of |mean_pred - obs_rate|).",
        "",
        "Definition of sample v1: [`mortality_panel_v1.md`](mortality_panel_v1.md).",
        "",
        "Logit: [`morti_crossborder_baseline.md`](morti_crossborder_baseline.md). "
        "CatBoost: [`morti_crossborder_catboost.md`](morti_crossborder_catboost.md).",
        "",
        "| model | experiment | prev_test | AUC-ROC | Brier | ECE | MCE |",
        "|-------|-------------|----------:|--------:|------:|----:|----:|",
    ]
    for r in metric_rows:
        auc = f"{r['auc_roc']:.4f}" if r["auc_roc"] is not None else "-"
        lines.append(
            f"| {r['model']} | {r['experiment']} | {r['prev_test']:.4f} | {auc} | "
            f"{r['brier']:.4f} | {r['ece']:.4f} | {r['mce']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reliability (CSV)",
            "",
            f"[`../tables_csv/morti_crossborder_calibration_reliability.csv`]"
            f"(../tables_csv/morti_crossborder_calibration_reliability.csv)",
            "",
            "## Interpretation",
            "",
            "A high ECE at low prev_test (QLD) means poor absolute calibration of P(y=1), "
            "even when the cross-border AUC (ranking) stays moderate. "
            "Compare ECE mainly within the pair train X -> test X vs train X -> test Y.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    spec = panel_specs(HARMONIZED_EVENT_LEVEL_DIR)["v1"]
    if not spec.path.is_file():
        raise SystemExit(
            f"Sample v1 not found: {spec.path}\n"
            "Run: python scripts/modeling/build_mortality_panel_v1.py"
        )

    df = load_panel(spec.path)
    metric_rows: list[dict] = []
    rel_rows: list[dict] = []

    fitters = (
        ("logit", _fit_logit_proba),
        ("catboost", _fit_catboost_proba),
    )

    for model_name, fit_fn in fitters:
        for name, train_kraj, test_kraj in EXPERIMENTS:
            train, test = train_test(df, train_kraj, test_kraj)
            y_test, proba = fit_fn(train, test)
            metric_rows.append(_eval_row(model=model_name, experiment=name, y_test=y_test, proba=proba))
            rel = reliability_table(y_test, proba).with_columns(
                pl.lit(model_name).alias("model"),
                pl.lit(name).alias("experiment"),
            )
            rel_rows.extend(rel.to_dicts())

    TABLES_CSV.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(metric_rows).write_csv(OUT_METRICS_CSV)
    pl.DataFrame(rel_rows).write_csv(OUT_REL_CSV)
    _write_md(metric_rows)

    for r in metric_rows:
        print(
            f"{r['model']} | {r['experiment']}: AUC={r.get('auc_roc')} "
            f"Brier={r['brier']} ECE={r['ece']}"
        )
    print(f"Report: {display_path(OUT_MD)}")


if __name__ == "__main__":
    main()
