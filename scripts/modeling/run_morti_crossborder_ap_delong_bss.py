"""
Additional cross-border discrimination and calibration metrics: `y_morti` (sample v1).

Extends the main tables with three items, on the same split and the same
models as the other scripts (logit + CatBoost, training year <= TRAIN_YEAR_MAX,
test year >= TEST_YEAR_MIN):

1. Average Precision (PR-AUC) per experiment and model, with a bootstrap 95% CI
   (n_boot = 1000, seed = 42, resampling of test rows, percentiles 2.5/97.5).
   Column baseline_ap = test prevalence (reference for a rare positive class).
2. Brier Skill Score (BSS) = 1 - Brier_model / Brier_baseline, where the baseline
   predicts a constant equal to the test-set prevalence.
3. DeLong test for the AUC difference on THE SAME test set:
     - QLD fold: AUC(train QLD -> test QLD) vs AUC(train PL -> test QLD),
     - PL fold: AUC(train PL -> test PL) vs AUC(train QLD -> test PL),
   separately for logit and CatBoost. Fast DeLong implementation (midrank), pure numpy.

Input:
  <PLQLD_DATA_DIR>/event_level/event_level_pl_qld_mortality_v1.parquet

Output:
  tables_csv/morti_crossborder_ap_ci.csv
  tables_md/morti_crossborder_ap_ci.md
  tables_csv/morti_crossborder_bss.csv
  tables_md/morti_crossborder_bss.md
  tables_csv/morti_crossborder_delong.csv
  tables_md/morti_crossborder_delong.md

Run from the repository root:
  python scripts/modeling/run_morti_crossborder_ap_delong_bss.py
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
    EXPERIMENTS,
    TEST_YEAR_MIN,
    TRAIN_YEAR_MAX,
    feature_flags_from_env,
    load_panel,
    panel_specs,
    train_test,
)

try:
    from scipy.stats import norm
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
except ImportError as exc:
    raise SystemExit("pip install -r requirements.txt") from exc

try:
    from catboost import CatBoostClassifier

    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

OUT_AP_CSV = TABLES_CSV / "morti_crossborder_ap_ci.csv"
OUT_AP_MD = TABLES_MD / "morti_crossborder_ap_ci.md"
OUT_BSS_CSV = TABLES_CSV / "morti_crossborder_bss.csv"
OUT_BSS_MD = TABLES_MD / "morti_crossborder_bss.md"
OUT_DELONG_CSV = TABLES_CSV / "morti_crossborder_delong.csv"
OUT_DELONG_MD = TABLES_MD / "morti_crossborder_delong.md"

CATBOOST_ITERATIONS = 500
N_BOOT = 1000
BOOT_SEED = 42
CI_LO_PCT = 2.5
CI_HI_PCT = 97.5


def _fit_logit_proba(train: pl.DataFrame, test: pl.DataFrame, feature_flags: tuple[str, ...]):
    x_train = train.select(feature_flags).to_numpy()
    y_train = train[Y_MORTI].to_numpy()
    x_test = test.select(feature_flags).to_numpy()
    y_test = test[Y_MORTI].to_numpy()
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    return y_test, model.predict_proba(x_test)[:, 1]


def _fit_catboost_proba(train: pl.DataFrame, test: pl.DataFrame, feature_flags: tuple[str, ...]):
    x_train = train.select(feature_flags).to_numpy()
    y_train = train[Y_MORTI].to_numpy()
    x_test = test.select(feature_flags).to_numpy()
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


def _bootstrap_ap_ci(y_test: np.ndarray, proba: np.ndarray) -> dict:
    n = len(y_test)
    n_pos = int(y_test.sum())
    point = round(float(average_precision_score(y_test, proba)), 6)
    rng = np.random.default_rng(BOOT_SEED)
    boot: list[float] = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        y_bs = y_test[idx]
        if y_bs.sum() == 0:
            continue
        boot.append(float(average_precision_score(y_bs, proba[idx])))
    arr = np.asarray(boot)
    return {
        "n_test": n,
        "n_pos_test": n_pos,
        "prev_test": round(n_pos / n, 6) if n else 0.0,
        "baseline_ap": round(n_pos / n, 6) if n else 0.0,
        "ap": point,
        "ci_lo": round(float(np.percentile(arr, CI_LO_PCT)), 6),
        "ci_hi": round(float(np.percentile(arr, CI_HI_PCT)), 6),
        "n_boot": len(boot),
    }


def _bss(y_test: np.ndarray, proba: np.ndarray) -> dict:
    n = len(y_test)
    n_pos = int(y_test.sum())
    prev = n_pos / n if n else 0.0
    brier_model = float(brier_score_loss(y_test, proba))
    brier_baseline = float(np.mean((prev - y_test) ** 2))
    bss = 1.0 - brier_model / brier_baseline if brier_baseline > 0 else None
    return {
        "n_test": n,
        "n_pos_test": n_pos,
        "prev_test": round(prev, 6),
        "brier": round(brier_model, 6),
        "brier_baseline": round(brier_baseline, 6),
        "bss": round(bss, 6) if bss is not None else None,
    }


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(predictions: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions.shape[1] - m
    positives = predictions[:, :m]
    negatives = predictions[:, m:]
    k = predictions.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positives[r, :])
        ty[r, :] = _compute_midrank(negatives[r, :])
        tz[r, :] = _compute_midrank(predictions[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def _delong_test(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray) -> dict:
    y = y_true.astype(np.int64)
    proba_a = proba_a.astype(np.float64)
    proba_b = proba_b.astype(np.float64)
    order = np.argsort(-y, kind="mergesort")
    label_1_count = int(y.sum())
    predictions = np.vstack((proba_a, proba_b))[:, order]
    aucs, cov = _fast_delong(predictions, label_1_count)
    for auc_val, proba in ((aucs[0], proba_a), (aucs[1], proba_b)):
        ref = float(roc_auc_score(y, proba))
        if abs(float(auc_val) - ref) > 1e-6:
            raise RuntimeError(f"DeLong AUC {auc_val:.6f} != roc_auc_score {ref:.6f}")
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 0:
        z = 0.0
        p = 1.0
    else:
        z = diff / np.sqrt(var)
        p = float(2.0 * norm.sf(abs(z)))
    return {
        "auc_a": round(float(aucs[0]), 6),
        "auc_b": round(float(aucs[1]), 6),
        "auc_diff": round(diff, 6),
        "z": round(float(z), 6),
        "p_value": p,
    }


def _fmt_p(p: float) -> str:
    if p is None:
        return "-"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _write_ap_md(rows: list[dict], *, catboost_used: bool) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Average Precision (PR-AUC) cross-border: `y_morti` (sample v1)",
        "",
        f"**Generated:** {ts} (`scripts/modeling/run_morti_crossborder_ap_delong_bss.py`)",
        "",
        f"**Split:** training year <= {TRAIN_YEAR_MAX}, test year >= {TEST_YEAR_MIN}.",
        f"**CI method:** model trained once per experiment; percentile bootstrap "
        f"(n_boot = {N_BOOT}, seed = {BOOT_SEED}) over test-set rows; "
        f"AP computed on each resample, interval = percentiles {CI_LO_PCT}% and {CI_HI_PCT}%. "
        "baseline_ap = test prevalence (AP of a random classifier).",
        "",
    ]
    if not catboost_used:
        lines += ["**Note:** CatBoost not available, only logit reported.", ""]
    lines += [
        "| model | experiment | n_test | n_pos_test | prev_test | baseline_ap | AP | CI 2.5% | CI 97.5% | n_boot |",
        "|-------|-------------|-------:|-----------:|----------:|------------:|---:|--------:|---------:|-------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['experiment']} | {r['n_test']:,} | {r['n_pos_test']:,} | "
            f"{r['prev_test']:.4f} | {r['baseline_ap']:.4f} | {r['ap']:.3f} | {r['ci_lo']:.3f} | "
            f"{r['ci_hi']:.3f} | {r['n_boot']} |"
        )
    lines += [
        "",
        "## CSV",
        "",
        "[`../tables_csv/morti_crossborder_ap_ci.csv`](../tables_csv/morti_crossborder_ap_ci.csv)",
        "",
    ]
    OUT_AP_MD.write_text("\n".join(lines), encoding="utf-8")


def _write_bss_md(rows: list[dict]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Brier Skill Score cross-border: `y_morti` (sample v1)",
        "",
        f"**Generated:** {ts} (`scripts/modeling/run_morti_crossborder_ap_delong_bss.py`)",
        "",
        f"**Split:** training year <= {TRAIN_YEAR_MAX}, test year >= {TEST_YEAR_MIN}.",
        "**BSS** = 1 - Brier_model / Brier_baseline; the baseline predicts a constant equal to "
        "the test-set prevalence. BSS > 0 = better than the constant baseline; BSS < 0 = worse. "
        "Raw Brier is not comparable between countries with different prevalence, BSS is.",
        "",
        "| model | experiment | prev_test | Brier | Brier_baseline | BSS |",
        "|-------|-------------|----------:|------:|---------------:|----:|",
    ]
    for r in rows:
        bss = f"{r['bss']:.4f}" if r["bss"] is not None else "n/a"
        lines.append(
            f"| {r['model']} | {r['experiment']} | {r['prev_test']:.4f} | {r['brier']:.4f} | "
            f"{r['brier_baseline']:.4f} | {bss} |"
        )
    lines += [
        "",
        "## CSV",
        "",
        "[`../tables_csv/morti_crossborder_bss.csv`](../tables_csv/morti_crossborder_bss.csv)",
        "",
    ]
    OUT_BSS_MD.write_text("\n".join(lines), encoding="utf-8")


def _write_delong_md(rows: list[dict]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# DeLong test (AUC difference on the same test set): `y_morti` (sample v1)",
        "",
        f"**Generated:** {ts} (`scripts/modeling/run_morti_crossborder_ap_delong_bss.py`)",
        "",
        f"**Split:** training year <= {TRAIN_YEAR_MAX}, test year >= {TEST_YEAR_MIN}.",
        "**Method:** fast DeLong (Sun & Xu, midrank) on pairs of models evaluated on "
        "an identical test set. `auc_diff = AUC(in-country) - AUC(cross-border)`, "
        "z and two-sided p. p < 0.05 = statistically significant difference in discrimination.",
        "",
        "| model | test fold | A (in-country) | B (cross-border) | AUC A | AUC B | AUC diff | z | p |",
        "|-------|-----------|----------------|------------------|------:|------:|---------:|--:|--:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['fold']} | {r['a_label']} | {r['b_label']} | "
            f"{r['auc_a']:.3f} | {r['auc_b']:.3f} | {r['auc_diff']:.3f} | {r['z']:.2f} | "
            f"{_fmt_p(r['p_value'])} |"
        )
    lines += [
        "",
        "## CSV",
        "",
        "[`../tables_csv/morti_crossborder_delong.csv`](../tables_csv/morti_crossborder_delong.csv)",
        "",
    ]
    OUT_DELONG_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    feature_flags = feature_flags_from_env()
    spec = panel_specs(HARMONIZED_EVENT_LEVEL_DIR)["v1"]
    if not spec.path.is_file():
        raise SystemExit(
            f"Sample v1 not found: {spec.path}\n"
            "Run: python scripts/modeling/build_mortality_panel_v1.py"
        )

    df = load_panel(spec.path, feature_flags=feature_flags)

    fitters = [("logit", _fit_logit_proba)]
    if _HAS_CATBOOST:
        fitters.append(("catboost", _fit_catboost_proba))

    proba_store: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    ap_rows: list[dict] = []
    bss_rows: list[dict] = []
    for model_name, fit_fn in fitters:
        for name, train_kraj, test_kraj in EXPERIMENTS:
            train, test = train_test(df, train_kraj, test_kraj)
            y_test, proba = fit_fn(train, test, feature_flags)
            proba_store[(model_name, name)] = (y_test, proba)
            ap_rows.append({"model": model_name, "experiment": name, **_bootstrap_ap_ci(y_test, proba)})
            bss_rows.append({"model": model_name, "experiment": name, **_bss(y_test, proba)})
            print(
                f"{model_name} | {name}: AP={ap_rows[-1]['ap']:.3f} "
                f"CI=({ap_rows[-1]['ci_lo']:.3f}, {ap_rows[-1]['ci_hi']:.3f}) "
                f"BSS={bss_rows[-1]['bss']}"
            )

    delong_pairs = (
        ("test QLD", "train QLD -> test QLD", "train PL -> test QLD"),
        ("test PL", "train PL -> test PL", "train QLD -> test PL"),
    )
    delong_rows: list[dict] = []
    for model_name, _ in fitters:
        for fold, a_exp, b_exp in delong_pairs:
            y_a, proba_a = proba_store[(model_name, a_exp)]
            y_b, proba_b = proba_store[(model_name, b_exp)]
            if not np.array_equal(y_a, y_b):
                raise RuntimeError(f"Different test sets for fold {fold} ({model_name})")
            res = _delong_test(y_a, proba_a, proba_b)
            delong_rows.append(
                {"model": model_name, "fold": fold, "a_label": a_exp, "b_label": b_exp, **res}
            )
            print(
                f"DeLong {model_name} | {fold}: diff={res['auc_diff']:.3f} "
                f"z={res['z']:.2f} p={_fmt_p(res['p_value'])}"
            )

    TABLES_CSV.mkdir(parents=True, exist_ok=True)
    TABLES_MD.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(ap_rows).select(
        ["model", "experiment", "n_test", "n_pos_test", "prev_test", "baseline_ap", "ap", "ci_lo", "ci_hi", "n_boot"]
    ).write_csv(OUT_AP_CSV)
    pl.DataFrame(bss_rows).select(
        ["model", "experiment", "prev_test", "brier", "brier_baseline", "bss"]
    ).write_csv(OUT_BSS_CSV)
    pl.DataFrame(delong_rows).select(
        ["model", "fold", "a_label", "b_label", "auc_a", "auc_b", "auc_diff", "z", "p_value"]
    ).write_csv(OUT_DELONG_CSV)

    _write_ap_md(ap_rows, catboost_used=_HAS_CATBOOST)
    _write_bss_md(bss_rows)
    _write_delong_md(delong_rows)

    print(
        f"Reports: {display_path(OUT_AP_MD)}, "
        f"{display_path(OUT_BSS_MD)}, {display_path(OUT_DELONG_MD)}"
    )


if __name__ == "__main__":
    main()
