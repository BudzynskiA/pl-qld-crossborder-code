"""
Sensitivity: y_morti models without the WARN flag `typ_inne` (19 features vs 20 in v1).

Compares cross-border and in-country logit: AUC, Brier, ECE on the same split
(training year <= TRAIN_YEAR_MAX, test year >= TEST_YEAR_MIN).

Input:
  <PLQLD_DATA_DIR>/event_level/event_level_pl_qld_mortality_v1.parquet

Output:
  tables_md/morti_crossborder_typ_inne_sensitivity.md
  tables_csv/morti_crossborder_typ_inne_sensitivity.csv

Run from the repository root:
  python scripts/modeling/run_morti_crossborder_typ_inne_sensitivity.py
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
    FEATURE_FLAGS,
    FEATURE_FLAGS_NO_TYP_INNE,
    TEST_YEAR_MIN,
    TRAIN_YEAR_MAX,
    TYP_INNE_FLAG,
    calibration_metrics,
    load_panel,
    panel_specs,
    train_test,
)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
except ImportError as exc:
    raise SystemExit("pip install -r requirements.txt") from exc

OUT_MD = TABLES_MD / "morti_crossborder_typ_inne_sensitivity.md"
OUT_CSV = TABLES_CSV / "morti_crossborder_typ_inne_sensitivity.csv"


def _fit_eval(train: pl.DataFrame, test: pl.DataFrame, feature_flags: tuple[str, ...]) -> dict:
    x_train = train.select(feature_flags).to_numpy()
    y_train = train[Y_MORTI].to_numpy()
    x_test = test.select(feature_flags).to_numpy()
    y_test = test[Y_MORTI].to_numpy()

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]

    n_test = len(y_test)
    n_pos = int(y_test.sum())
    prev = n_pos / n_test if n_test else 0.0
    cal = calibration_metrics(y_test, proba)

    out: dict = {
        "n_train": train.height,
        "n_test": n_test,
        "n_pos_test": n_pos,
        "prev_test": round(prev, 6),
        "brier": round(float(brier_score_loss(y_test, proba)), 6),
        "ece": cal["ece"],
        "mce": cal["mce"],
    }
    if n_pos > 0 and n_pos < n_test:
        out["auc_roc"] = round(float(roc_auc_score(y_test, proba)), 6)
    else:
        out["auc_roc"] = None
    return out


def _write_md(rows: list[dict]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Sensitivity: excluding `typ_inne` (logit, sample v1)",
        "",
        f"**Generated:** {ts} (`scripts/modeling/run_morti_crossborder_typ_inne_sensitivity.py`)",
        "",
        f"**Comparison:** v1 ({len(FEATURE_FLAGS)} flags, with `{TYP_INNE_FLAG}` WARN) vs "
        f"sensitivity ({len(FEATURE_FLAGS_NO_TYP_INNE)} flags, without `{TYP_INNE_FLAG}`).",
        f"**Split:** training year <= {TRAIN_YEAR_MAX}, test year >= {TEST_YEAR_MIN}.",
        "",
        "| variant | experiment | n_features | AUC-ROC | Brier | ECE | delta_AUC vs v1 |",
        "|---------|-------------|----------:|--------:|------:|----:|----------------:|",
    ]
    by_exp: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_exp.setdefault(r["experiment"], {})[r["variant"]] = r

    for exp in (e[0] for e in EXPERIMENTS):
        v1 = by_exp[exp]["v1_20flags"]
        no = by_exp[exp]["no_typ_inne_19flags"]
        delta = (
            f"{no['auc_roc'] - v1['auc_roc']:+.4f}"
            if v1["auc_roc"] is not None and no["auc_roc"] is not None
            else "n/a"
        )
        auc_v1 = f"{v1['auc_roc']:.4f}" if v1["auc_roc"] is not None else "n/a"
        auc_no = f"{no['auc_roc']:.4f}" if no["auc_roc"] is not None else "n/a"
        lines.append(
            f"| v1 (20 flags) | {exp} | 20 | {auc_v1} | {v1['brier']:.4f} | {v1['ece']:.4f} | - |"
        )
        lines.append(
            f"| without typ_inne | {exp} | 19 | {auc_no} | {no['brier']:.4f} | {no['ece']:.4f} | {delta} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Flag `{TYP_INNE_FLAG}` has a strong prevalence asymmetry PL vs QLD (~31x). "
            "Removing this feature tests whether the cross-border results are sensitive to residual crash-type mapping.",
            "",
            "## CSV",
            "",
            f"[`../tables_csv/morti_crossborder_typ_inne_sensitivity.csv`]"
            f"(../tables_csv/morti_crossborder_typ_inne_sensitivity.csv)",
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

    variants = (
        ("v1_20flags", FEATURE_FLAGS),
        ("no_typ_inne_19flags", FEATURE_FLAGS_NO_TYP_INNE),
    )

    rows: list[dict] = []
    for variant_name, flags in variants:
        df = load_panel(spec.path, feature_flags=flags)
        for name, train_kraj, test_kraj in EXPERIMENTS:
            train, test = train_test(df, train_kraj, test_kraj)
            metrics = _fit_eval(train, test, flags)
            rows.append(
                {
                    "variant": variant_name,
                    "experiment": name,
                    "n_features": len(flags),
                    **metrics,
                }
            )
            auc = metrics["auc_roc"] if metrics["auc_roc"] is not None else "n/a"
            print(f"{variant_name} | {name}: AUC={auc} Brier={metrics['brier']:.4f} ECE={metrics['ece']:.4f}")

    TABLES_CSV.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(OUT_CSV)
    _write_md(rows)
    print(f"Report: {display_path(OUT_MD)}")


if __name__ == "__main__":
    main()
