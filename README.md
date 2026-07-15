# Store Sales: Time Series Forecasting

## Project Overview

This repository contains an undergraduate introductory research project based on Kaggle's **Store Sales - Time Series Forecasting** competition. The goal is to develop a clear and reproducible forecasting study for daily sales across product families and Favorita stores in Ecuador.

The work is organized in stages. Stage 1 establishes the project structure and audits the raw data, Stage 2 provides descriptive exploratory analysis, Stage 3 establishes chronological validation and simple baselines, and Stage 4 engineers leakage-safe features. Model comparison belongs to later stages.

## Dataset

The raw competition files are stored in `data/raw/`. The main forecasting unit is defined by `date`, `store_nbr`, and `family`, and the training target is `sales`.

The data include historical sales, the 16-day test horizon, store metadata, oil prices, holidays and events, daily transactions, and a sample submission. Raw files must remain unchanged.

## Repository Structure

```text
store-sales-time-series-forecasting/
|-- README.md
|-- .gitignore
|-- requirements.txt
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|-- reports/
|   |-- data_audit.md
|   `-- figures/
|-- src/
|   |-- __init__.py
|   `-- audit_data.py
`-- submissions/
```

## Setup and Data Audit

Create and activate a Python virtual environment, then install the Stage 1 dependency:

```bash
python -m pip install -r requirements.txt
```

From the repository root, generate the audit report with:

```bash
python -m src.audit_data
```

The script supports either `data/raw/train.csv` or a ZIP archive such as `data/raw/train.csv.zip`. It prints a concise progress summary and writes [the raw data audit](reports/data_audit.md).

## Planned Workflow

1. Project setup and raw data audit
2. Exploratory data analysis
3. Time-based validation and simple baselines
4. Interpretable feature engineering
5. Model comparison
6. Error analysis and final reporting

Later stages will prioritize reproducibility, avoid data leakage, and compare methods under a consistent time-based validation design.

## Stage 1 Status

Stage 1 is complete. The repository now has a reproducible structure, a simple audit command, and a generated report covering dimensions, data types, missing values, duplicates, date ranges, key uniqueness, and cross-file consistency.

No exploratory analysis, feature engineering, predictive modeling, validation, or submission generation is included in this stage. Known oil, holiday, and transaction data issues are documented for later work.

## Stage 2 Status

Stage 2 is complete. The reproducible [EDA notebook](notebooks/02_exploratory_data_analysis.ipynb) examines temporal, weekday, monthly, family, store, promotion, transaction, holiday, earthquake, and oil-price patterns. A concise [EDA summary](reports/eda_summary.md) presents the main findings and selected figures.

The analysis is descriptive and associative. It does not include predictive modeling, a validation split, model features, or submission generation; those tasks remain for later stages.

## Stage 3 Status

Stage 3 is complete. The [validation and baseline notebook](notebooks/03_validation_and_baselines.ipynb) and [baseline results report](reports/baseline_results.md) use the final 16 training dates (`2017-07-31` through `2017-08-15`) as a leakage-safe holdout.

The best transparent baseline is the **8-week weekday mean**, with validation RMSLE **0.520631**. The reusable baseline module can refresh the generated score table with:

```bash
python -m src.baselines
```

No machine-learning model, Kaggle test prediction, or submission was included in Stage 3.

## Stage 4 Status

Stage 4 is complete. The [feature-engineering notebook](notebooks/04_feature_engineering.ipynb) and [feature-engineering report](reports/feature_engineering.md) document one shared feature build reused across four fixed Ridge checks.

The best check uses calendar, store, promotion, lag, and rolling features, producing 29 encoded columns and validation RMSLE **0.494938**. This improves on the Stage 3 benchmark of `0.520631`. Simplified holiday and oil features did not improve the check and are deferred.

Refresh the generated feature tables with:

```bash
python -m src.features
```

Ridge is used only to verify feature information. Stage 5 model comparison remains incomplete, and no Kaggle test prediction or submission has been generated.
