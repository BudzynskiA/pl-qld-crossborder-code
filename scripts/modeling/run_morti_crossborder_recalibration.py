"""
Cross-border recalibration of y_morti models + covariate shift diagnostics (sample v1).

Addresses the reviewer concern that naive transfer gives poor absolute calibration
of P(y=1). Shows that a small calibration sample from the target country (training
years) reduces ECE without changing the test set (years >= TEST_YEAR_MIN),
and quantifies the distribution shift of the 20 features between PL and QLD.

Design per experiment (train A -> test B):
  1. base model: logit (class_weight=balanced) trained on A, years <= TRAIN_YEAR_MAX;
  2. recalibrator (Platt / isotonic) fitted on B, years <= TRAIN_YEAR_MAX
     (small target sample), input = base model probability;
  3. evaluation on B, years >= TEST_YEAR_MIN (same set as the main tables):
     ECE/Brier before vs after recalibration.

Input:
  <PLQLD_DATA_DIR>/event_level/event_level_pl_qld_mortality_v1.parquet

Output:
  tables_md/morti_crossborder_recalibration.md
  tables_csv/morti_crossborder_recalibration_metrics.csv
  tables_csv/morti_crossborder_covariate_shift.csv

Run from the repository root:
  python scripts/modeling/run_morti_crossborder_recalibration.py
  PL_QLD_EXCLUDE_TYP_INNE=1 python scripts/modeling/run_morti_crossborder_recalibration.py
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

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from harmonization.outcome_morti import Y_MORTI  # noqa: E402
from modeling._morti_crossborder_common import (  # noqa: E402
    CALIBRATION_BINS,
    EXPERIMENTS,
    FAIL_FLAGS_EXCLUDED,
    TEST_YEAR_MIN,
    TRAIN_YEAR_MAX,
    calibration_metrics,
    feature_flags_from_env,
    load_panel,
    panel_specs,
    subset,
)

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    raise SystemExit("pip install -r requirements.txt") from exc

OUT_MD = TABLES_MD / "morti_crossborder_recalibration.md"
OUT_METRICS_CSV = TABLES_CSV / "morti_crossborder_recalibration_metrics.csv"
OUT_SHIFT_CSV = TABLES_CSV / "morti_crossborder_covariate_shift.csv"

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _fit_base_logit(train: pl.DataFrame, feature_flags: tuple[str, ...]):
    x = train.select(feature_flags).to_numpy()
    y = train[Y_MORTI].to_numpy()
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(x, y)
    return model


def _proba(model, frame: pl.DataFrame, feature_flags: tuple[str, ...]) -> np.ndarray:
    x = frame.select(feature_flags).to_numpy()
    return model.predict_proba(x)[:, 1]


def _platt(base_proba_cal: np.ndarray, y_cal: np.ndarray, base_proba_eval: np.ndarray) -> np.ndarray:
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(_logit(base_proba_cal).reshape(-1, 1), y_cal)
    return lr.predict_proba(_logit(base_proba_eval).reshape(-1, 1))[:, 1]


def _isotonic(base_proba_cal: np.ndarray, y_cal: np.ndarray, base_proba_eval: np.ndarray) -> np.ndarray:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(base_proba_cal, y_cal)
    return iso.predict(base_proba_eval)


def _metrics(tag: str, experiment: str, y_eval: np.ndarray, proba: np.ndarray) -> dict:
    n_pos = int(y_eval.sum())
    n = len(y_eval)
    cal = calibration_metrics(y_eval, proba)
    auc = (
        round(float(roc_auc_score(y_eval, proba)), 6)
        if 0 < n_pos < n
        else None
    )
    prev = n_pos / n if n else 0.0
    brier_ref = prev * (1.0 - prev)
    bss = (
        round(1.0 - cal["brier"] / brier_ref, 6)
        if cal["brier"] is not None and brier_ref > 0
        else None
    )
    return {
        "experiment": experiment,
        "variant": tag,
        "n_eval": n,
        "n_pos_eval": n_pos,
        "prev_eval": round(prev, 6) if n else 0.0,
        "auc_roc": auc,
        "brier": cal["brier"],
        "bss": bss,
        "ece": cal["ece"],
        "mce": cal["mce"],
    }


def _covariate_shift(df: pl.DataFrame, feature_flags: tuple[str, ...]) -> list[dict]:
    pl_train = subset(df, "PL", year_min=None, year_max=TRAIN_YEAR_MAX)
    qld_train = subset(df, "QLD", year_min=None, year_max=TRAIN_YEAR_MAX)
    rows: list[dict] = []
    for flag in feature_flags:
        p_pl = float(pl_train[flag].mean()) if pl_train.height else 0.0
        p_qld = float(qld_train[flag].mean()) if qld_train.height else 0.0
        rows.append(
            {
                "flag": flag,
                "prev_pl_train": round(p_pl, 6),
                "prev_qld_train": round(p_qld, 6),
                "abs_diff": round(abs(p_pl - p_qld), 6),
                "psi": round(_psi(p_pl, p_qld), 6),
            }
        )
    rows.sort(key=lambda r: r["abs_diff"], reverse=True)
    return rows


def _psi(p: float, q: float) -> float:
    """Population Stability Index for a binary flag (two states: 1 and 0)."""
    total = 0.0
    for a, b in ((p, q), (1.0 - p, 1.0 - q)):
        a = min(max(a, _EPS), 1.0 - _EPS)
        b = min(max(b, _EPS), 1.0 - _EPS)
        total += (a - b) * np.log(a / b)
    return float(total)


def _write_md(metric_rows: list[dict], shift_rows: list[dict], *, feature_flags: tuple[str, ...]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    excluded = ", ".join(f"`{f}`" for f in FAIL_FLAGS_EXCLUDED)
    typ_note = " (without `typ_inne`)" if len(feature_flags) == 19 else ""
    lines = [
        "# Cross-border recalibration + covariate shift: `y_morti` (sample v1)",
        "",
        f"**Generated:** {ts} (`scripts/modeling/run_morti_crossborder_recalibration.py`)",
        "",
        f"**Split:** base model trained on the source country (year <= {TRAIN_YEAR_MAX}); "
        f"recalibrator fitted on the target country (year <= {TRAIN_YEAR_MAX}); "
        f"evaluation on the target country (year >= {TEST_YEAR_MIN}).",
        f"**Features X:** {len(feature_flags)} harmonized flags{typ_note}. "
        f"**Excluded (audit FAIL):** {excluded}.",
        f"**ECE binning:** {CALIBRATION_BINS} quantile bins.",
        "",
        "## Calibration before vs after recalibration",
        "",
        "BSS = 1 - Brier / [p*(1-p)], p = test prevalence (constant forecast as baseline). "
        "BSS < 0 means worse than the constant-prevalence forecast; ~0 or positive = calibration restored.",
        "",
        "| experiment | variant | prev_eval | AUC-ROC | Brier | BSS | ECE | MCE |",
        "|-------------|---------|----------:|--------:|------:|----:|----:|----:|",
    ]
    for r in metric_rows:
        auc = f"{r['auc_roc']:.4f}" if r["auc_roc"] is not None else "n/a"
        bss = f"{r['bss']:.4f}" if r["bss"] is not None else "n/a"
        lines.append(
            f"| {r['experiment']} | {r['variant']} | {r['prev_eval']:.4f} | {auc} | "
            f"{r['brier']:.4f} | {bss} | {r['ece']:.4f} | {r['mce']:.4f} |"
        )
    lines.extend(
        [
            "",
            "AUC-ROC is invariant to monotonic recalibration (Platt/isotonic), "
            "so the ranking does not change; the drop in ECE/Brier shows the improvement in absolute "
            "calibration after adding a small target sample.",
            "",
            "## Covariate shift (training years, flag prevalence PL vs QLD)",
            "",
            "PSI > 0.25 is usually treated as a strong distribution shift.",
            "",
            "| flag | prev PL | prev QLD | |difference| | PSI |",
            "|-------|--------:|---------:|----------:|----:|",
        ]
    )
    for r in shift_rows:
        lines.append(
            f"| {r['flag']} | {r['prev_pl_train']:.4f} | {r['prev_qld_train']:.4f} | "
            f"{r['abs_diff']:.4f} | {r['psi']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## CSV",
            "",
            f"[`../tables_csv/morti_crossborder_recalibration_metrics.csv`]"
            f"(../tables_csv/morti_crossborder_recalibration_metrics.csv), "
            f"[`../tables_csv/morti_crossborder_covariate_shift.csv`]"
            f"(../tables_csv/morti_crossborder_covariate_shift.csv)",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    feature_flags = feature_flags_from_env()
    spec = panel_specs(HARMONIZED_EVENT_LEVEL_DIR)["v1"]
    if not spec.path.is_file():
        raise SystemExit(
            f"Sample v1 not found: {spec.path}\n"
            "Run: python scripts/modeling/build_mortality_panel_v1.py"
        )

    df = load_panel(spec.path, feature_flags=feature_flags)
    metric_rows: list[dict] = []

    for name, train_kraj, test_kraj in EXPERIMENTS:
        src_train = subset(df, train_kraj, year_min=None, year_max=TRAIN_YEAR_MAX)
        tgt_cal = subset(df, test_kraj, year_min=None, year_max=TRAIN_YEAR_MAX)
        tgt_eval = subset(df, test_kraj, year_min=TEST_YEAR_MIN, year_max=None)

        base = _fit_base_logit(src_train, feature_flags)
        y_eval = tgt_eval[Y_MORTI].to_numpy()

        base_eval = _proba(base, tgt_eval, feature_flags)
        base_cal = _proba(base, tgt_cal, feature_flags)
        y_cal = tgt_cal[Y_MORTI].to_numpy()

        metric_rows.append(_metrics("base", name, y_eval, base_eval))
        if 0 < int(y_cal.sum()) < len(y_cal):
            metric_rows.append(_metrics("platt", name, y_eval, _platt(base_cal, y_cal, base_eval)))
            metric_rows.append(_metrics("isotonic", name, y_eval, _isotonic(base_cal, y_cal, base_eval)))

    shift_rows = _covariate_shift(df, feature_flags)

    TABLES_CSV.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(metric_rows).write_csv(OUT_METRICS_CSV)
    pl.DataFrame(shift_rows).write_csv(OUT_SHIFT_CSV)
    _write_md(metric_rows, shift_rows, feature_flags=feature_flags)

    for r in metric_rows:
        print(
            f"{r['experiment']} | {r['variant']}: AUC={r['auc_roc']} "
            f"Brier={r['brier']} BSS={r['bss']} ECE={r['ece']}"
        )
    print(f"Report: {display_path(OUT_MD)}")


if __name__ == "__main__":
    main()
