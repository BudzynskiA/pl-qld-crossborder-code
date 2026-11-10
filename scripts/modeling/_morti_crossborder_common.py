"""Shared constants and helpers for the cross-border PL<->QLD y_morti models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from harmonization.lokalizacja_flags import LOKALIZACJA_FLAGS
from harmonization.nawierzchnia_flags import NAWIERZCHNIA_FLAGS
from harmonization.outcome_morti import Y_MORTI
from harmonization.pogoda_flags import POGODA_FLAGS
from harmonization.typ_zdarzenia_flags import TYP_ZDARZENIA_FLAGS

# Leakage audit (stage 5): FAIL flags, excluded from model v1 (=1 separates the countries).
FAIL_FLAGS_EXCLUDED: tuple[str, ...] = (
    "pogoda_snieg_grad",
    "pogoda_wiatr",
    "typ_kolej",
    "naw_sdrucciolevole",
    "naw_lod",
    "naw_snieg",
    "naw_uszkodzona",
)

# WARN: typ_inne is kept in v1 (frequency asymmetry PL vs QLD, ratio ~31).
ALL_HARMONIZED_FLAGS: tuple[str, ...] = (
    POGODA_FLAGS + TYP_ZDARZENIA_FLAGS + NAWIERZCHNIA_FLAGS + LOKALIZACJA_FLAGS
)

FEATURE_FLAGS: tuple[str, ...] = tuple(f for f in ALL_HARMONIZED_FLAGS if f not in FAIL_FLAGS_EXCLUDED)

# WARN from the leakage audit: frequency asymmetry PL vs QLD (~31x).
TYP_INNE_FLAG = "typ_inne"
FEATURE_FLAGS_NO_TYP_INNE: tuple[str, ...] = tuple(f for f in FEATURE_FLAGS if f != TYP_INNE_FLAG)


def resolve_feature_flags(*, exclude_typ_inne: bool = False) -> tuple[str, ...]:
    """19 flags without typ_inne (sensitivity) or the full 20 v1 flags."""
    return FEATURE_FLAGS_NO_TYP_INNE if exclude_typ_inne else FEATURE_FLAGS


def feature_flags_from_env() -> tuple[str, ...]:
    """PL_QLD_EXCLUDE_TYP_INNE=1 -> 19 flags without typ_inne."""
    import os

    return resolve_feature_flags(exclude_typ_inne=os.environ.get("PL_QLD_EXCLUDE_TYP_INNE", "").strip() in ("1", "true", "yes"))


TRAIN_YEAR_MAX = 2022
TEST_YEAR_MIN = 2023
CALIBRATION_BINS = 10

EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    ("train PL -> test QLD", "PL", "QLD"),
    ("train QLD -> test PL", "QLD", "PL"),
    ("train PL -> test PL", "PL", "PL"),
    ("train QLD -> test QLD", "QLD", "QLD"),
)


@dataclass(frozen=True)
class PanelSpec:
    key: str
    path: Path
    label: str


def panel_specs(harmonized_dir: Path) -> dict[str, PanelSpec]:
    return {
        "full": PanelSpec(
            key="full",
            path=harmonized_dir / "event_level_pl_qld_2015_2024.parquet",
            label="full harmonized panel",
        ),
        "v1": PanelSpec(
            key="v1",
            path=harmonized_dir / "event_level_pl_qld_mortality_v1.parquet",
            label="comparable sample v1",
        ),
    }


def load_panel(
    path: Path,
    *,
    drop_lok_unassigned: bool = False,
    feature_flags: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    flags = feature_flags or FEATURE_FLAGS
    cols = ["kraj", "rok", Y_MORTI, *flags]
    df = pl.read_parquet(path, columns=cols)
    if drop_lok_unassigned:
        lok_sum = pl.sum_horizontal([pl.col(c) for c in LOKALIZACJA_FLAGS])
        df = df.filter(lok_sum > 0)
    return df


def subset(df: pl.DataFrame, kraj: str, year_min: int | None, year_max: int | None) -> pl.DataFrame:
    out = df.filter(pl.col("kraj") == kraj)
    if year_min is not None:
        out = out.filter(pl.col("rok") >= year_min)
    if year_max is not None:
        out = out.filter(pl.col("rok") <= year_max)
    return out


def train_test(df: pl.DataFrame, train_kraj: str, test_kraj: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = subset(df, train_kraj, year_min=None, year_max=TRAIN_YEAR_MAX)
    test = subset(df, test_kraj, year_min=TEST_YEAR_MIN, year_max=None)
    return train, test


def reliability_table(y_true: np.ndarray, proba: np.ndarray, *, n_bins: int = CALIBRATION_BINS) -> pl.DataFrame:
    """Quantile binning: mean P(y) vs observed positive rate."""
    order = np.argsort(proba)
    y_sorted = y_true[order]
    p_sorted = proba[order]
    n = len(y_true)
    if n == 0:
        return pl.DataFrame(
            schema={
                "bin": pl.Int64,
                "n": pl.Int64,
                "mean_pred": pl.Float64,
                "obs_rate": pl.Float64,
                "cal_gap": pl.Float64,
            }
        )

    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    rows: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if lo >= hi:
            continue
        y_bin = y_sorted[lo:hi]
        p_bin = p_sorted[lo:hi]
        obs = float(y_bin.mean())
        pred = float(p_bin.mean())
        rows.append(
            {
                "bin": i + 1,
                "n": hi - lo,
                "mean_pred": round(pred, 6),
                "obs_rate": round(obs, 6),
                "cal_gap": round(abs(pred - obs), 6),
            }
        )
    return pl.DataFrame(rows)


def calibration_metrics(y_true: np.ndarray, proba: np.ndarray, *, n_bins: int = CALIBRATION_BINS) -> dict:
    from sklearn.metrics import brier_score_loss

    rel = reliability_table(y_true, proba, n_bins=n_bins)
    if rel.is_empty():
        return {"brier": None, "ece": None, "mce": None, "n_test": 0}

    weights = rel["n"].to_numpy() / rel["n"].sum()
    gaps = rel["cal_gap"].to_numpy()
    ece = float((weights * gaps).sum())
    mce = float(gaps.max())
    return {
        "n_test": int(len(y_true)),
        "brier": round(float(brier_score_loss(y_true, proba)), 6),
        "ece": round(ece, 6),
        "mce": round(mce, 6),
    }


# CatBoost: the number of boosting iterations is selected by early stopping on the last
# training year (CATBOOST_VAL_YEAR) of the source country; the model is then refitted on all
# training years with that number of iterations. Test records are never used for model selection.
CATBOOST_MAX_ITERATIONS = 500
CATBOOST_VAL_YEAR = TRAIN_YEAR_MAX


def make_catboost(iterations: int = CATBOOST_MAX_ITERATIONS):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        loss_function="Logloss",
        iterations=iterations,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )


def catboost_val_proba(
    train: pl.DataFrame,
    test: pl.DataFrame,
    flags: tuple[str, ...],
) -> tuple[np.ndarray, int]:
    """Early stopping on CATBOOST_VAL_YEAR of the source country, refit on the full training set.

    Returns (predicted probabilities on the test set, number of iterations used).
    """
    fit = train.filter(pl.col("rok") < CATBOOST_VAL_YEAR)
    val = train.filter(pl.col("rok") == CATBOOST_VAL_YEAR)
    probe = make_catboost()
    probe.fit(
        fit.select(flags).to_numpy(),
        fit[Y_MORTI].to_numpy(),
        eval_set=(val.select(flags).to_numpy(), val[Y_MORTI].to_numpy()),
        use_best_model=True,
    )
    best_iter = max(int(probe.get_best_iteration() or 0) + 1, 1)
    model = make_catboost(iterations=best_iter)
    model.fit(train.select(flags).to_numpy(), train[Y_MORTI].to_numpy())
    return model.predict_proba(test.select(flags).to_numpy())[:, 1], best_iter
