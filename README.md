# Store Sales: Time Series Forecasting

## Project Overview

This repository contains an undergraduate introductory research project based on Kaggle's **Store Sales - Time Series Forecasting** competition. The goal is to develop a clear and reproducible forecasting study for daily sales across product families and Favorita stores in Ecuador.

The work is organized in stages. Stage 1 establishes the project structure and audits the raw data; exploratory analysis, validation, feature engineering, and modeling belong to later stages.

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
