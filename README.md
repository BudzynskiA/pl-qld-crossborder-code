# Cross-border fatal-crash modelling: Poland (SEWIK) and Queensland (TMR)

Analysis code for the article *Harmonizing police crash registers for cross-border fatal-crash modelling: Poland and Queensland* (Budzyński A., Paz A.), Accident Analysis and Prevention, 2026, <https://doi.org/10.1016/j.aap.2026.108871>.

The code builds event-level crash records from the Polish police register (SEWIK) and the Queensland road crash file (TMR), harmonizes them into shared binary indicators (weather, crash type, pavement, location), audits the indicators for country leakage, and evaluates fatal-crash models (logistic regression, CatBoost, plus RF, HistGradientBoosting and XGBoost in the sensitivity analysis) in four transfer settings (PL→QLD, QLD→PL, PL→PL, QLD→QLD; train 2015-2022, test 2023-2024). Evaluation covers AUC with confidence intervals, average precision, DeLong tests, Brier skill score, calibration and recalibration (Platt, isotonic), and covariate shift.

## Data

**No data are included.**

| Source | Access |
|--------|--------|
| Poland: SEWIK (Police Accident and Collision Recording System) | Not publicly available. Microdata are provided by the Polish National Police Headquarters (Komenda Główna Policji) on request. |
| Queensland: Department of Transport and Main Roads (TMR), *Crash data from Queensland roads: Road crash locations* (`_1_crash_locations.csv`) | Open data: <https://www.data.qld.gov.au/dataset/crash-data-from-queensland-roads> |

Point the scripts to local copies with environment variables (see `scripts/_bootstrap.py`):

| Variable | Expected content | Default |
|----------|------------------|---------|
| `PLQLD_POLAND_DIR` | `data/merged/2015_2025/{zdarzenia,uczestnik,pojazdy}/*.parquet` (merged SEWIK tables); derived files are written to `data/derived/event_level/` | `data/poland` |
| `PLQLD_QUEENSLAND_DIR` | `data/raw/tmr/_1_crash_locations.csv`; derived files are written to `data/derived/event_level/` | `data/queensland` |
| `QLD_TMR_RAW_CSV` | optional: explicit path to the TMR CSV | - |
| `PLQLD_DATA_DIR` | harmonized panels written by the scripts | `data/harmonized` |
| `PL_QLD_EXCLUDE_TYP_INNE` | optional: `1` drops the "other crash type" indicator from the predictor set | - |

The merged SEWIK tables (`zdarzenia` events, `uczestnik` participants, `pojazdy` vehicles, 2015-2025, one parquet file per table) come from the national SEWIK export after joining yearly files and normalizing participant codes to snake_case labels (`scripts/national/poland/sewik_codes.py`). That upstream step is not part of this repository.

## Contents

| Path | What |
|------|------|
| `scripts/national/poland/` | Polish event-level input: injury crashes 2015-2024 (`build_derived_event_level_wypadki_2015_2024.py`), enriched with participant and vehicle aggregates (`build_derived_event_level_wypadki_enriched_2015_2024.py`) |
| `scripts/national/queensland/` | Queensland event-level input from the TMR CSV, 2015-2024, property-damage-only crashes excluded (`build_derived_event_level_qld_2015_2024.py`) |
| `scripts/harmonization/` | indicator definitions by domain (`pogoda_flags.py` weather, `typ_zdarzenia_flags.py` crash type, `nawierzchnia_flags.py` pavement, `lokalizacja_flags.py` location, `outcome_morti.py` outcome), harmonized panel builder, country-leakage audit |
| `scripts/comparison/` | validation of the two registers before modelling (yearly counts, outcome rates, indicator coverage) |
| `scripts/modeling/` | comparable modelling sample, logit and CatBoost, calibration, recalibration, AP/DeLong/BSS, sensitivity analyses, reliability figure |
| `docs/crosswalk_S1.md` | harmonization crosswalk (Supplementary Table S1): source fields and codes per indicator |
| `results/` | aggregate result tables (CSV) produced by the scripts, as used in the manuscript |

Column and variable names follow the SEWIK source fields and are partly Polish (e.g. `rok` year, `kraj` country, `ilosc_zmarlych` number killed, `y_morti` fatal-crash indicator, `feriti` injured). Indicator prefixes: `pogoda_*` weather, `typ_*` crash type, `naw_*` pavement, `lok_*` location.

## Run order

From the repository root, after `pip install -r requirements.txt` (Python 3.10+):

```bash
# national inputs
python scripts/national/poland/build_derived_event_level_wypadki_2015_2024.py
python scripts/national/poland/build_derived_event_level_wypadki_enriched_2015_2024.py
python scripts/national/queensland/build_derived_event_level_qld_2015_2024.py

# harmonization and validation
python scripts/harmonization/build_event_level_pl_qld_harmonized.py
python scripts/comparison/build_event_level_validation_pl_qld.py

# modelling
python scripts/modeling/build_mortality_panel_v1.py
python scripts/modeling/run_morti_crossborder_baseline.py
python scripts/modeling/run_morti_crossborder_catboost.py
python scripts/modeling/run_morti_crossborder_calibration.py
python scripts/modeling/run_morti_crossborder_recalibration.py
python scripts/modeling/run_morti_crossborder_ap_delong_bss.py
python scripts/modeling/run_morti_crossborder_typ_inne_sensitivity.py
python scripts/modeling/run_morti_crossborder_revision_sensitivity.py
python scripts/modeling/plot_morti_crossborder_reliability.py
```

Outputs go to `tables_csv/`, `tables_md/` and `figures/` (not tracked). The copies in `results/` are the tables reported in the paper.

CatBoost selects the number of boosting iterations by early stopping on the last training year (2022) of the source jurisdiction and is then refitted on all training years; test records are not used for model selection. The CatBoost results reported in the paper (Tables 7, 8, 11 and 13) are in `results/morti_crossborder_revision_sensitivity.csv` (model `catboost_val2022`).

## Citation

If you use this code, please cite:

> Budzyński, A., Paz, A. (2026). Harmonizing police crash registers for cross-border fatal-crash modelling: Poland and Queensland. *Accident Analysis and Prevention*, 108871. https://doi.org/10.1016/j.aap.2026.108871

## License

All rights reserved until a license is chosen (see `LICENSE`).
