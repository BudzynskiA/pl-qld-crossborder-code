"""
Cross-border CatBoost: y_morti on sample v1 with the 20 OK harmonized flags.

Input:
  <PLQLD_DATA_DIR>/event_level/event_level_pl_qld_mortality_v1.parquet
  (first: python scripts/modeling/build_mortality_panel_v1.py)

Outputs:
  tables_md/morti_crossborder_catboost.md
  tables_csv/morti_crossborder_catboost_metrics.csv

Experiments (temporal split: train year <= 2022, test year >= 2023):
  train PL -> test QLD, train QLD -> test PL, train PL -> test PL, train QLD -> test QLD

Run from the repository root:
  python scripts/modeling/run_morti_crossborder_catboost.py
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
    CATBOOST_MAX_ITERATIONS,
    CATBOOST_VAL_YEAR,
    catboost_val_proba,
    load_panel,
    panel_specs,
    train_test,
)

try:
    import catboost  # noqa: F401
    from sklearn.metrics import brier_score_loss, roc_auc_score
except ImportError as exc:
    raise SystemExit("pip install -r requirements.txt (catboost, scikit-learn)") from exc

OUT_MD = TABLES_MD / "morti_crossborder_catboost.md"
OUT_CSV = TABLES_CSV / "morti_crossborder_catboost_metrics.csv"

def _fit_eval(train: pl.DataFrame, test: pl.DataFrame) -> dict:
    y_test = test[Y_MORTI].to_numpy()

    proba, best_iter = catboost_val_proba(train, test, FEATURE_FLAGS)

    n_test = len(y_test)
    n_pos = int(y_test.sum())
    prev = n_pos / n_test if n_test else 0.0

    out: dict = {
        "n_train": train.height,
        "n_test": n_test,
        "n_pos_test": n_pos,
        "prev_test": round(prev, 6),
        "brier": round(float(brier_score_loss(y_test, proba)), 6),
        "best_iteration": best_iter,
    }
    if n_pos > 0 and n_pos < n_test:
        out["auc_roc"] = round(float(roc_auc_score(y_test, proba)), 6)
    else:
        out["auc_roc"] = None
    return out


def _write_md(rows: list[dict]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    excluded = ", ".join(f"`{f}`" for f in FAIL_FLAGS_EXCLUDED)
    lines = [
        "# CatBoost cross-border: `y_morti` (sample v1)",
        "",
        f"**Generated:** {ts} (`scripts/modeling/run_morti_crossborder_catboost.py`)",
        "",
        "**Panel:** `<PLQLD_DATA_DIR>/event_level/event_level_pl_qld_mortality_v1.parquet`",
        "",
        f"**Y:** `{Y_MORTI}` = 1 when `morti_count` > 0.",
        f"**X:** {len(FEATURE_FLAGS)} OK harmonized flags (no `kraj`, no `rok`).",
        f"**Excluded (audit FAIL):** {excluded}.",
        f"**Split:** train year <= {TRAIN_YEAR_MAX}, test year >= {TEST_YEAR_MIN}.",
        f"**Model:** CatBoostClassifier (Logloss, depth=6, at most {CATBOOST_MAX_ITERATIONS} iterations, "
        f"auto_class_weights=Balanced; iterations selected by early stopping on year {CATBOOST_VAL_YEAR} "
        "of the source country, then refit on all training years; test records not used for model selection).",
        "",
        "Definition of sample v1: [`mortality_panel_v1.md`](mortality_panel_v1.md).",
        "",
        "Comparison with the logit on sample v1: [`morti_crossborder_baseline.md`](morti_crossborder_baseline.md). "
        "Calibration: [`morti_crossborder_calibration.md`](morti_crossborder_calibration.md).",
        "",
        "| experiment | n_train | n_test | prev_test | AUC-ROC | Brier | best_iter |",
        "|-------------|--------:|-------:|----------:|--------:|------:|----------:|",
    ]
    for r in rows:
        auc = f"{r['auc_roc']:.4f}" if r["auc_roc"] is not None else "-"
        lines.append(
            f"| {r['experiment']} | {r['n_train']:,} | {r['n_test']:,} | "
            f"{r['prev_test']:.4f} | {auc} | {r['brier']:.4f} | {r['best_iteration']} |"
        )
    lines.extend(
        [
            "",
            "## CSV",
            "",
            f"[`../tables_csv/morti_crossborder_catboost_metrics.csv`]"
            f"(../tables_csv/morti_crossborder_catboost_metrics.csv)",
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
    if Y_MORTI not in df.columns:
        raise SystemExit(f"Missing {Y_MORTI} in panel v1")

    rows: list[dict] = []
    for name, train_kraj, test_kraj in EXPERIMENTS:
        train, test = train_test(df, train_kraj, test_kraj)
        metrics = _fit_eval(train, test)
        rows.append({"experiment": name, **metrics})

    TABLES_CSV.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(OUT_CSV)
    _write_md(rows)

    for r in rows:
        auc = r["auc_roc"] if r["auc_roc"] is not None else "n/a"
        print(
            f"{r['experiment']}: n_test={r['n_test']:,} prev={r['prev_test']:.4f} "
            f"AUC={auc} Brier={r['brier']:.4f} iter={r['best_iteration']}"
        )
    print(f"Report: {display_path(OUT_MD)}")


if __name__ == "__main__":
    main()
