"""
Sensitivity analyses added in the manuscript revision: y_morti cross-border PL <-> QLD.

Blocks (same test set: year >= TEST_YEAR_MIN, sample v1):
  covid      : training without years 2020-2021 (logit + CatBoost)
  location   : alternative lok_* mappings (logit):
                qld_strict : QLD motorway by road name only (motorway/freeway)
                pl_broad   : PL motorway + expressway (counterpart of the QLD proxy)
                no_location: without all lok_* flags (17 flags)
              (dropping lok_autostrada alone is a reparametrization: lok_* are mutually exclusive and exhaustive)
  learners   : RF, HistGradientBoosting, CatBoost, XGBoost (20 flags):
                catboost_val2022  : best iteration on year 2022 of the source country, refit on the full training set
                catboost_fixed500 : no early stopping
                xgboost_val2022   : same iteration-selection protocol as catboost_val2022
  ksi        : severity threshold: only crashes with a fatality or a seriously injured/hospitalised person (logit)
              (+ in the md: counts/prevalence of the fatal_or_serious sample and the QLD Crash_Severity distribution)
  native     : in-country CatBoost: 20 flags vs 20 flags + national fields
              (information cost of harmonization; national fields are not shared)
  extended   : extended contract (logit): 20 flags + lighting (4) + speed band (5)
              PL `oswietlenie` / QLD `Crash_Lighting_Condition` (1:1);
              PL `predkosc_dopuszczalna` -> TMR bands (<=50, 60, 70, 80-90, >=100)
  unweighted : logit without class weights (separates calibration error due to class weights vs due to transfer)
  delong     : in-country vs cross-border for catboost_val2022 (same test fold)
Each row: 95% CI for AUC (DeLong variance).

Input:
  <PLQLD_DATA_DIR>/event_level/event_level_pl_qld_mortality_v1.parquet

Output:
  tables_md/morti_crossborder_revision_sensitivity.md
  tables_csv/morti_crossborder_revision_sensitivity.csv

Run from the repository root:
  python scripts/modeling/run_morti_crossborder_revision_sensitivity.py
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

from harmonization.lokalizacja_flags import LOKALIZACJA_FLAGS  # noqa: E402
from harmonization.outcome_morti import Y_MORTI  # noqa: E402
from modeling.run_morti_crossborder_ap_delong_bss import _delong_test, _fast_delong  # noqa: E402
from modeling._morti_crossborder_common import (  # noqa: E402
    EXPERIMENTS,
    FEATURE_FLAGS,
    TEST_YEAR_MIN,
    TRAIN_YEAR_MAX,
    calibration_metrics,
    panel_specs,
    subset,
)

try:
    from catboost import CatBoostClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit("pip install -r requirements.txt (catboost, xgboost, scikit-learn)") from exc

OUT_MD = TABLES_MD / "morti_crossborder_revision_sensitivity.md"
OUT_CSV = TABLES_CSV / "morti_crossborder_revision_sensitivity.csv"

COVID_YEARS: tuple[int, ...] = (2020, 2021)
MOTORWAY_FLAG = "lok_autostrada"
CATBOOST_VAL_YEAR = TRAIN_YEAR_MAX

PL_NATIVE_CAT: tuple[str, ...] = (
    "oswietlenie",
    "skrzyzowanie",
    "geometria_drogi",
    "charakterystyka_miejsca",
    "sygnalizacja_swietlna",
    "rodzaj_pojazdu_a",
)
PL_NATIVE_NUM: tuple[str, ...] = ("predkosc_num", "hour", "ilosc_pojazdow", "ilosc_pieszych")
QLD_NATIVE_CAT: tuple[str, ...] = (
    "Crash_Speed_Limit",
    "Crash_Lighting_Condition",
    "Crash_Roadway_Feature",
    "Crash_Traffic_Control",
    "Crash_Road_Horiz_Align",
    "Crash_Road_Vert_Align",
)
QLD_NATIVE_NUM: tuple[str, ...] = (
    "Crash_Hour",
    "Count_Unit_Car",
    "Count_Unit_Motorcycle_Moped",
    "Count_Unit_Truck",
    "Count_Unit_Bus",
    "Count_Unit_Bicycle",
    "Count_Unit_Pedestrian",
)
LIGHT_FLAGS: tuple[str, ...] = ("osw_dzien", "osw_noc_oswietlona", "osw_noc_nieoswietlona", "osw_swit_zmrok")
SPEED_FLAGS: tuple[str, ...] = ("pred_do_50", "pred_60", "pred_70", "pred_80_90", "pred_100_plus")
PL_LIGHT_CODES: tuple[str, ...] = ("swiatlo_dzienne", "noc_oswietlona", "noc_nieoswietlona", "swit_zmrok")
QLD_LIGHT_CODES: tuple[str, ...] = ("Daylight", "Darkness - Lighted", "Darkness - Not lighted", "Dawn/Dusk")
QLD_SPEED_CODES: tuple[str, ...] = ("0 - 50 km/h", "60 km/h", "70 km/h", "80 - 90 km/h", "100 - 110 km/h")

NATIVE_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "PL": (PL_NATIVE_CAT, PL_NATIVE_NUM),
    "QLD": (QLD_NATIVE_CAT, QLD_NATIVE_NUM),
}


def _load_full_panel(path: Path) -> pl.DataFrame:
    raw_cols = [
        "obszar",
        "rodzaj_drogi",
        "State_Road_Name",
        "Loc_ABS_Remoteness",
        "data_zdarzenia",
        "predkosc_dopuszczalna",
        "ilosc_ciezko_rannych",
        "Crash_Severity",
        *PL_NATIVE_CAT,
        "ilosc_pojazdow",
        "ilosc_pieszych",
        *QLD_NATIVE_CAT,
        *QLD_NATIVE_NUM,
    ]
    df = pl.read_parquet(path, columns=["kraj", "rok", Y_MORTI, *FEATURE_FLAGS, *raw_cols])
    return df.with_columns(
        pl.col("predkosc_dopuszczalna").cast(pl.Float64, strict=False).alias("predkosc_num"),
        pl.col("data_zdarzenia").str.slice(11, 2).cast(pl.Int64, strict=False).alias("hour"),
    )


def _with_location_variant(df: pl.DataFrame, variant: str) -> pl.DataFrame:
    """Overwrite lok_* according to the mapping variant (other flags unchanged)."""
    is_qld = pl.col("kraj") == "QLD"
    is_pl = pl.col("kraj") == "PL"
    if variant == "qld_strict":
        srn = pl.col("State_Road_Name").fill_null("").str.to_lowercase()
        strict = srn.str.contains("motorway") | srn.str.contains("freeway")
        major = pl.col("Loc_ABS_Remoteness").fill_null("") == "Major Cities"
        return df.with_columns(
            pl.when(is_qld).then(strict.cast(pl.UInt8)).otherwise(pl.col("lok_autostrada")).alias("lok_autostrada"),
            pl.when(is_qld).then((major & ~strict).cast(pl.UInt8)).otherwise(pl.col("lok_miejski")).alias("lok_miejski"),
            pl.when(is_qld)
            .then((~major & ~strict).cast(pl.UInt8))
            .otherwise(pl.col("lok_poza_miastem"))
            .alias("lok_poza_miastem"),
        )
    if variant == "pl_broad":
        rodzaj = pl.col("rodzaj_drogi").fill_null("").str.to_lowercase()
        ekspresowa = rodzaj == "ekspresowa"
        return df.with_columns(
            pl.when(is_pl & ekspresowa).then(pl.lit(1, dtype=pl.UInt8)).otherwise(pl.col("lok_autostrada")).alias("lok_autostrada"),
            pl.when(is_pl & ekspresowa).then(pl.lit(0, dtype=pl.UInt8)).otherwise(pl.col("lok_poza_miastem")).alias("lok_poza_miastem"),
        )
    raise ValueError(variant)


def _with_extended_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Lighting and speed-band flags (candidates for extending the contract)."""
    is_pl = pl.col("kraj") == "PL"
    v = pl.col("predkosc_num")
    pl_speed = (v <= 50, (v > 50) & (v <= 60), (v > 60) & (v <= 70), (v > 70) & (v <= 90), v > 90)
    exprs: list[pl.Expr] = []
    for flag, pl_code, qld_code in zip(LIGHT_FLAGS, PL_LIGHT_CODES, QLD_LIGHT_CODES):
        exprs.append(
            pl.when(is_pl)
            .then(pl.col("oswietlenie") == pl_code)
            .otherwise(pl.col("Crash_Lighting_Condition") == qld_code)
            .fill_null(False)
            .cast(pl.UInt8)
            .alias(flag)
        )
    for flag, pl_rule, qld_code in zip(SPEED_FLAGS, pl_speed, QLD_SPEED_CODES):
        exprs.append(
            pl.when(is_pl)
            .then(pl_rule)
            .otherwise(pl.col("Crash_Speed_Limit") == qld_code)
            .fill_null(False)
            .cast(pl.UInt8)
            .alias(flag)
        )
    return df.with_columns(exprs)


def _ksi_only(df: pl.DataFrame) -> pl.DataFrame:
    """Crashes with a fatality or a seriously injured (PL) / hospitalised (QLD) person."""
    pl_ksi = (pl.col("kraj") == "PL") & ((pl.col("ilosc_ciezko_rannych") > 0) | (pl.col(Y_MORTI) == 1))
    qld_ksi = (pl.col("kraj") == "QLD") & pl.col("Crash_Severity").is_in(["Fatal", "Hospitalisation"])
    return df.filter(pl_ksi | qld_ksi)


def _split(df: pl.DataFrame, train_kraj: str, test_kraj: str, *, drop_years: tuple[int, ...] = ()) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = subset(df, train_kraj, year_min=None, year_max=TRAIN_YEAR_MAX)
    if drop_years:
        train = train.filter(~pl.col("rok").is_in(list(drop_years)))
    test = subset(df, test_kraj, year_min=TEST_YEAR_MIN, year_max=None)
    return train, test


def _make_logit():
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)


def _make_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        iterations=500,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )


def _make_rf():
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=50, class_weight="balanced", n_jobs=-1, random_state=42
    )


def _make_hgb():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_depth=6, class_weight="balanced", random_state=42
    )


def _make_xgb(*, n_estimators: int = 500, scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=6,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def _scale_pos_weight(y: np.ndarray) -> float:
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    return n_neg / max(n_pos, 1.0)


def _auc_ci(y_test: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """95% CI for AUC from the DeLong variance (normal approximation)."""
    y = y_test.astype(np.int64)
    order = np.argsort(-y, kind="mergesort")
    aucs, cov = _fast_delong(proba.astype(np.float64)[np.newaxis, order], int(y.sum()))
    se = float(np.sqrt(np.atleast_2d(cov)[0, 0]))
    auc = float(aucs[0])
    return auc - 1.96 * se, auc + 1.96 * se


def _score(y_test: np.ndarray, proba: np.ndarray) -> dict:
    cal = calibration_metrics(y_test, proba)
    ci_lo, ci_hi = _auc_ci(y_test, proba)
    return {
        "n_test": int(len(y_test)),
        "prev_test": round(float(y_test.mean()), 6),
        "auc_roc": round(float(roc_auc_score(y_test, proba)), 6),
        "auc_ci_lo": round(ci_lo, 6),
        "auc_ci_hi": round(ci_hi, 6),
        "ap": round(float(average_precision_score(y_test, proba)), 6),
        "brier": round(float(brier_score_loss(y_test, proba)), 6),
        "ece": cal["ece"],
    }


def _fit_eval(model, train: pl.DataFrame, test: pl.DataFrame, flags: tuple[str, ...]) -> dict:
    x_train = train.select(flags).to_numpy()
    y_train = train[Y_MORTI].to_numpy()
    x_test = test.select(flags).to_numpy()
    y_test = test[Y_MORTI].to_numpy()
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    return {"n_train": train.height, "n_features": len(flags), **_score(y_test, proba)}


def _catboost_val_proba(train: pl.DataFrame, test: pl.DataFrame, flags: tuple[str, ...]) -> tuple[np.ndarray, int]:
    """Early stopping on year CATBOOST_VAL_YEAR of the source country; refit on the full training set; test fold untouched."""
    fit = train.filter(pl.col("rok") < CATBOOST_VAL_YEAR)
    val = train.filter(pl.col("rok") == CATBOOST_VAL_YEAR)
    probe = _make_catboost()
    probe.fit(
        fit.select(flags).to_numpy(),
        fit[Y_MORTI].to_numpy(),
        eval_set=(val.select(flags).to_numpy(), val[Y_MORTI].to_numpy()),
        use_best_model=True,
    )
    best_iter = max(int(probe.get_best_iteration() or 0) + 1, 1)
    model = _make_catboost()
    model.set_params(iterations=best_iter)
    model.fit(train.select(flags).to_numpy(), train[Y_MORTI].to_numpy())
    return model.predict_proba(test.select(flags).to_numpy())[:, 1], best_iter


def _fit_eval_catboost_val(train: pl.DataFrame, test: pl.DataFrame, flags: tuple[str, ...]) -> tuple[dict, np.ndarray]:
    proba, best_iter = _catboost_val_proba(train, test, flags)
    y_test = test[Y_MORTI].to_numpy()
    cal = calibration_metrics(y_test, proba)
    out = {"n_train": train.height, "n_features": len(flags), **_score(y_test, proba), "mce": cal["mce"], "best_iter": best_iter}
    return out, proba


def _xgb_val_proba(train: pl.DataFrame, test: pl.DataFrame, flags: tuple[str, ...]) -> tuple[np.ndarray, int]:
    """Early stopping on year CATBOOST_VAL_YEAR of the source country; refit on the full training set; test fold untouched."""
    fit = train.filter(pl.col("rok") < CATBOOST_VAL_YEAR)
    val = train.filter(pl.col("rok") == CATBOOST_VAL_YEAR)
    y_fit = fit[Y_MORTI].to_numpy()
    spw = _scale_pos_weight(y_fit)
    probe = _make_xgb(scale_pos_weight=spw)
    probe.set_params(early_stopping_rounds=50)
    probe.fit(
        fit.select(flags).to_numpy(),
        y_fit,
        eval_set=[(val.select(flags).to_numpy(), val[Y_MORTI].to_numpy())],
        verbose=False,
    )
    best_iter = max(int(getattr(probe, "best_iteration", 0) or 0) + 1, 1)
    model = _make_xgb(n_estimators=best_iter, scale_pos_weight=_scale_pos_weight(train[Y_MORTI].to_numpy()))
    model.fit(train.select(flags).to_numpy(), train[Y_MORTI].to_numpy(), verbose=False)
    return model.predict_proba(test.select(flags).to_numpy())[:, 1], best_iter


def _fit_eval_xgb_val(train: pl.DataFrame, test: pl.DataFrame, flags: tuple[str, ...]) -> dict:
    proba, best_iter = _xgb_val_proba(train, test, flags)
    y_test = test[Y_MORTI].to_numpy()
    cal = calibration_metrics(y_test, proba)
    return {
        "n_train": train.height,
        "n_features": len(flags),
        **_score(y_test, proba),
        "mce": cal["mce"],
        "best_iter": best_iter,
    }


def _fit_eval_native(train: pl.DataFrame, test: pl.DataFrame, kraj: str) -> dict:
    cat_cols, num_cols = NATIVE_FIELDS[kraj]
    cols = [*FEATURE_FLAGS, *num_cols, *cat_cols]

    def _frame(d: pl.DataFrame):
        return d.select(
            [pl.col(c) for c in FEATURE_FLAGS]
            + [pl.col(c).cast(pl.Float64, strict=False) for c in num_cols]
            + [pl.col(c).cast(pl.Utf8).fill_null("NA") for c in cat_cols]
        ).to_pandas()

    # Same protocol as catboost_val2022: best iteration on the validation year, refit on the full training set.
    fit = train.filter(pl.col("rok") < CATBOOST_VAL_YEAR)
    val = train.filter(pl.col("rok") == CATBOOST_VAL_YEAR)
    probe = _make_catboost()
    probe.fit(
        _frame(fit),
        fit[Y_MORTI].to_numpy(),
        cat_features=list(cat_cols),
        eval_set=(_frame(val), val[Y_MORTI].to_numpy()),
        use_best_model=True,
    )
    best_iter = max(int(probe.get_best_iteration() or 0) + 1, 1)
    model = _make_catboost()
    model.set_params(iterations=best_iter)
    model.fit(_frame(train), train[Y_MORTI].to_numpy(), cat_features=list(cat_cols))
    proba = model.predict_proba(_frame(test))[:, 1]
    return {
        "n_train": train.height,
        "n_features": len(cols),
        **_score(test[Y_MORTI].to_numpy(), proba),
        "best_iter": best_iter,
    }


def _run(df: pl.DataFrame) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    cb_proba: dict[tuple[str, str], np.ndarray] = {}
    df_ksi = _ksi_only(df)
    df_ext = _with_extended_flags(df)
    ext_sets = (
        ("plus_lighting_24", (*FEATURE_FLAGS, *LIGHT_FLAGS)),
        ("plus_speed_25", (*FEATURE_FLAGS, *SPEED_FLAGS)),
        ("plus_lighting_speed_29", (*FEATURE_FLAGS, *LIGHT_FLAGS, *SPEED_FLAGS)),
    )

    def _add(block: str, variant: str, model_name: str, experiment: str, metrics: dict) -> None:
        rows.append({"block": block, "variant": variant, "model": model_name, "experiment": experiment, **metrics})
        print(
            f"{block:9s} | {variant:22s} | {model_name:16s} | {experiment:22s} "
            f"AUC={metrics['auc_roc']:.4f} AP={metrics['ap']:.4f} ECE={metrics['ece']:.4f}"
        )

    for name, train_kraj, test_kraj in EXPERIMENTS:
        # covid
        for variant, drop in (("all_train_years", ()), ("excl_2020_2021", COVID_YEARS)):
            train, test = _split(df, train_kraj, test_kraj, drop_years=drop)
            _add("covid", variant, "logit", name, _fit_eval(_make_logit(), train, test, FEATURE_FLAGS))
            _add("covid", variant, "catboost_val2022", name, _fit_eval_catboost_val(train, test, FEATURE_FLAGS)[0])

        # location
        no_location = tuple(f for f in FEATURE_FLAGS if f not in LOKALIZACJA_FLAGS)
        train, test = _split(df, train_kraj, test_kraj)
        _add("location", "v1_mapping", "logit", name, _fit_eval(_make_logit(), train, test, FEATURE_FLAGS))
        for variant in ("qld_strict", "pl_broad"):
            tr, te = _split(_with_location_variant(df, variant), train_kraj, test_kraj)
            _add("location", variant, "logit", name, _fit_eval(_make_logit(), tr, te, FEATURE_FLAGS))

        _add("location", "no_location", "logit", name, _fit_eval(_make_logit(), train, test, no_location))

        # learners
        _add("learners", "20flags", "logit", name, _fit_eval(_make_logit(), train, test, FEATURE_FLAGS))
        cb_metrics, cb_proba[(train_kraj, test_kraj)] = _fit_eval_catboost_val(train, test, FEATURE_FLAGS)
        _add("learners", "20flags", "catboost_val2022", name, cb_metrics)
        _add("learners", "20flags", "catboost_fixed500", name, _fit_eval(_make_catboost(), train, test, FEATURE_FLAGS))
        _add("learners", "20flags", "random_forest", name, _fit_eval(_make_rf(), train, test, FEATURE_FLAGS))
        _add("learners", "20flags", "hist_gb", name, _fit_eval(_make_hgb(), train, test, FEATURE_FLAGS))
        _add("learners", "20flags", "xgboost_val2022", name, _fit_eval_xgb_val(train, test, FEATURE_FLAGS))

        # unweighted
        _add(
            "unweighted",
            "no_class_weights",
            "logit",
            name,
            _fit_eval(LogisticRegression(max_iter=2000, random_state=42), train, test, FEATURE_FLAGS),
        )

        # ksi
        tr_k, te_k = _split(df_ksi, train_kraj, test_kraj)
        _add("ksi", "fatal_or_serious_only", "logit", name, _fit_eval(_make_logit(), tr_k, te_k, FEATURE_FLAGS))

        # extended
        tr_e, te_e = _split(df_ext, train_kraj, test_kraj)
        for variant, flags in ext_sets:
            _add("extended", variant, "logit", name, _fit_eval(_make_logit(), tr_e, te_e, flags))

        # native (in-country only)
        if train_kraj == test_kraj:
            _add("native", "20flags", "catboost_val2022", name, cb_metrics)
            _add("native", "20flags_plus_native", "catboost_val2022", name, _fit_eval_native(train, test, train_kraj))

    delong_rows: list[dict] = []
    for test_kraj, source in (("QLD", "PL"), ("PL", "QLD")):
        y_test = subset(df, test_kraj, year_min=TEST_YEAR_MIN, year_max=None)[Y_MORTI].to_numpy()
        res = _delong_test(y_test, cb_proba[(test_kraj, test_kraj)], cb_proba[(source, test_kraj)])
        delong_rows.append({"model": "catboost_val2022", "test_fold": test_kraj, **res})
        print(f"delong    | catboost_val2022 | test {test_kraj}: in={res['auc_a']:.4f} cross={res['auc_b']:.4f} "
              f"diff={res['auc_diff']:.4f} p={res['p_value']:.2e}")
    return rows, delong_rows


def _ksi_sample_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Counts and y_morti prevalence in the fatal_or_serious sample (definition as in _ksi_only)."""
    n_v1 = {r["kraj"]: r["n"] for r in df.group_by("kraj").agg(pl.len().alias("n")).iter_rows(named=True)}
    ksi = _ksi_only(df)
    rows = []
    for r in (
        ksi.group_by("kraj")
        .agg(
            pl.len().alias("n"),
            pl.col(Y_MORTI).sum().alias("n_deaths"),
            pl.col(Y_MORTI).mean().alias("prev_y_morti"),
        )
        .sort("kraj")
        .iter_rows(named=True)
    ):
        n_full = int(n_v1[r["kraj"]])
        n = int(r["n"])
        rows.append(
            {
                "kraj": r["kraj"],
                "n": n,
                "share_of_v1": n / n_full if n_full else 0.0,
                "n_v1": n_full,
                "n_deaths": int(r["n_deaths"]),
                "prev_y_morti": float(r["prev_y_morti"]),
            }
        )
    return pl.DataFrame(rows)


def _qld_crash_severity_dist(df: pl.DataFrame) -> pl.DataFrame:
    """Distribution of Crash_Severity in QLD in sample v1 (n and share)."""
    qld = df.filter(pl.col("kraj") == "QLD")
    n_total = qld.height
    return (
        qld.group_by("Crash_Severity")
        .agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / n_total).alias("share"))
        .sort("n", descending=True)
    )


def _write_md(
    rows: list[dict],
    delong_rows: list[dict],
    prevalence: pl.DataFrame,
    ksi_stats: pl.DataFrame,
    qld_severity: pl.DataFrame,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Revision sensitivity analyses: COVID, location mapping, learners, national fields",
        "",
        f"**Generated:** {ts} (`scripts/modeling/run_morti_crossborder_revision_sensitivity.py`)",
        "",
        f"**Split:** training year <= {TRAIN_YEAR_MAX}, test year >= {TEST_YEAR_MIN}; sample v1; "
        f"{len(FEATURE_FLAGS)} flags (unless stated otherwise).",
        "",
        "| block | variant | model | experiment | n_train | n_features | AUC-ROC | AUC 95% CI | AP | Brier | ECE |",
        "|------|---------|-------|-------------|--------:|-----------:|--------:|-----------|---:|------:|----:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['block']} | {r['variant']} | {r['model']} | {r['experiment']} | {r['n_train']:,} | "
            f"{r['n_features']} | {r['auc_roc']:.4f} | {r['auc_ci_lo']:.3f} to {r['auc_ci_hi']:.3f} | "
            f"{r['ap']:.4f} | {r['brier']:.4f} | {r['ece']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Early stopping on 2022 (catboost_val2022 / xgboost_val2022): MCE and best iteration",
            "",
            "| model | variant | experiment | n_test | prev_test | AUC-ROC | Brier | ECE | MCE | best_iter |",
            "|-------|---------|-------------|-------:|----------:|--------:|------:|----:|----:|----------:|",
        ]
    )
    for r in rows:
        if r["model"] in ("catboost_val2022", "xgboost_val2022") and "mce" in r:
            lines.append(
                f"| {r['model']} | {r['block']}/{r['variant']} | {r['experiment']} | {r['n_test']:,} | "
                f"{r['prev_test']:.4f} | {r['auc_roc']:.4f} | {r['brier']:.4f} | {r['ece']:.4f} | "
                f"{r['mce']:.4f} | {r['best_iter']} |"
            )
    lines.extend(
        [
            "",
            "## DeLong: in-country minus cross-border (catboost_val2022)",
            "",
            "| test fold | AUC in-country | AUC cross-border | difference | z | p |",
            "|-----------|---------------:|-----------------:|--------:|--:|--:|",
        ]
    )
    for r in delong_rows:
        lines.append(
            f"| {r['test_fold']} | {r['auc_a']:.4f} | {r['auc_b']:.4f} | {r['auc_diff']:.4f} | "
            f"{r['z']:.2f} | {r['p_value']:.2e} |"
        )
    lines.extend(
        [
            "",
            "## Share of lok_autostrada by mapping variant (full panel v1)",
            "",
            "| variant | country | n | share lok_autostrada | prev y_morti (motorway=1) |",
            "|---------|------|--:|----------------------:|----------------------------:|",
        ]
    )
    for r in prevalence.iter_rows(named=True):
        lines.append(
            f"| {r['variant']} | {r['kraj']} | {r['n']:,} | {r['share']:.4f} | {r['prev_fatal']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Sample fatal_or_serious_only (definition as in block `ksi`, full panel v1)",
            "",
            "| country | n | n_v1 | share of v1 | n deaths (y_morti=1) | prev y_morti |",
            "|------|--:|-----:|------------:|---------------------:|-------------:|",
        ]
    )
    for r in ksi_stats.iter_rows(named=True):
        lines.append(
            f"| {r['kraj']} | {r['n']:,} | {r['n_v1']:,} | {r['share_of_v1']:.4f} | "
            f"{r['n_deaths']:,} | {r['prev_y_morti']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Distribution of Crash_Severity in QLD (sample v1)",
            "",
            "| Crash_Severity | n | share |",
            "|----------------|--:|-------:|",
        ]
    )
    for r in qld_severity.iter_rows(named=True):
        sev = r["Crash_Severity"] if r["Crash_Severity"] is not None else "(null)"
        lines.append(f"| {sev} | {r['n']:,} | {r['share']:.4f} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"`catboost_val2022` and `xgboost_val2022` select the best iteration on year {CATBOOST_VAL_YEAR} "
            "of the source country and refit on the full training set; `catboost_fixed500` is CatBoost without eval_set.",
            "- Block `ksi`: PL `ilosc_ciezko_rannych` > 0 or fatality; QLD `Crash_Severity` in (Fatal, Hospitalisation).",
            "- Block `native`: national fields (speed, lighting, intersection, hour, number of vehicles, etc.) "
            "are not harmonized; the result is an in-country reference point, not a transferable model.",
            "",
            "## CSV",
            "",
            "[`../tables_csv/morti_crossborder_revision_sensitivity.csv`]"
            "(../tables_csv/morti_crossborder_revision_sensitivity.csv)",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def _extended_prevalence(df: pl.DataFrame) -> pl.DataFrame:
    d = _with_extended_flags(df)
    rows = []
    for flag in (*LIGHT_FLAGS, *SPEED_FLAGS):
        share = {r["kraj"]: r["share"] for r in d.group_by("kraj").agg(pl.col(flag).mean().alias("share")).iter_rows(named=True)}
        rows.append({"flag": flag, "share_pl": share["PL"], "share_qld": share["QLD"]})
    return pl.DataFrame(rows)


def _motorway_prevalence(df: pl.DataFrame) -> pl.DataFrame:
    frames = []
    for variant in ("v1_mapping", "qld_strict", "pl_broad"):
        d = df if variant == "v1_mapping" else _with_location_variant(df, variant)
        frames.append(
            d.group_by("kraj")
            .agg(
                pl.len().alias("n"),
                pl.col(MOTORWAY_FLAG).mean().alias("share"),
                pl.col(Y_MORTI).filter(pl.col(MOTORWAY_FLAG) == 1).mean().alias("prev_fatal"),
            )
            .with_columns(pl.lit(variant).alias("variant"))
            .sort("kraj")
        )
    return pl.concat(frames)


def main() -> None:
    spec = panel_specs(HARMONIZED_EVENT_LEVEL_DIR)["v1"]
    if not spec.path.is_file():
        raise SystemExit(
            f"Sample v1 not found: {spec.path}\n"
            "Run: python scripts/modeling/build_mortality_panel_v1.py"
        )

    df = _load_full_panel(spec.path)
    rows, delong_rows = _run(df)
    ksi_stats = _ksi_sample_stats(df)
    qld_severity = _qld_crash_severity_dist(df)

    TABLES_CSV.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT_CSV)
    _write_md(rows, delong_rows, _motorway_prevalence(df), ksi_stats, qld_severity)
    ext = _extended_prevalence(df)
    with OUT_MD.open("a", encoding="utf-8") as fh:
        fh.write("\n## Extended contract: flag shares (full panel v1)\n\n| flag | PL | QLD |\n|-------|---:|----:|\n")
        for r in ext.iter_rows(named=True):
            fh.write(f"| {r['flag']} | {r['share_pl']:.4f} | {r['share_qld']:.4f} |\n")
    # Avoid Windows console cp1250 crashes on polars DataFrame repr.
    print(ext.write_csv())
    print(f"Report: {display_path(OUT_MD)}")


if __name__ == "__main__":
    main()
