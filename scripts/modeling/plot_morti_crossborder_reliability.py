"""
Reliability curves (cross-border logit) from the calibration CSV -> PNG in figures/.

Input:
  tables_csv/morti_crossborder_calibration_reliability.csv

Output:
  figures/morti_crossborder_reliability_logit.png

Run from the repository root:
  python scripts/modeling/plot_morti_crossborder_reliability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))
from _bootstrap import FIGURES, TABLES_CSV, display_path, install  # noqa: E402

install()

import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

REL_CSV = TABLES_CSV / "morti_crossborder_calibration_reliability.csv"
FIG_DIR = FIGURES
OUT_PNG = FIG_DIR / "morti_crossborder_reliability_logit.png"

CROSS_EXPERIMENTS = (
    "train PL -> test QLD",
    "train QLD -> test PL",
)
PANEL_TITLES = {
    "train PL -> test QLD": "Trained on Poland, tested on Queensland",
    "train QLD -> test PL": "Trained on Queensland, tested on Poland",
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Reliability curves logit cross-border")
    parser.parse_args()

    if not REL_CSV.is_file():
        raise SystemExit(
            f"File not found: {REL_CSV}\nRun: python scripts/modeling/run_morti_crossborder_calibration.py"
        )

    df = pl.read_csv(REL_CSV).filter(
        (pl.col("model") == "logit") & pl.col("experiment").is_in(CROSS_EXPERIMENTS)
    )
    if df.is_empty():
        raise SystemExit("No cross-border logit rows in the reliability CSV")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    label_fs = 11
    title_fs = 11
    tick_fs = 10
    legend_fs = 10
    for ax, exp in zip(axes, CROSS_EXPERIMENTS):
        sub = df.filter(pl.col("experiment") == exp).sort("bin")
        ax.plot(sub["mean_pred"], sub["obs_rate"], "o-", label="Observed vs predicted")
        lim = max(float(sub["mean_pred"].max()), float(sub["obs_rate"].max()), 0.05)
        ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="Perfect calibration")
        ax.set_title(PANEL_TITLES[exp], fontsize=title_fs)
        ax.set_xlabel("Mean predicted probability", fontsize=label_fs)
        ax.set_ylabel("Observed fatal-crash rate", fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.legend(fontsize=legend_fs)
        ax.set_xlim(0, lim * 1.05)
        ax.set_ylim(0, lim * 1.05)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {display_path(OUT_PNG)}")


if __name__ == "__main__":
    main()
