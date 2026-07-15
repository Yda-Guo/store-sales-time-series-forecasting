# Store Sales: Time Series Forecasting

## Project Overview

This repository contains a time series forecasting project based on Kaggle's **Store Sales - Time Series Forecasting** competition.

The goal is to predict daily sales for different product families across Favorita stores in Ecuador. The project focuses on building a clear and reproducible forecasting workflow, including data exploration, time-based validation, baseline construction, feature engineering, model comparison, and error analysis.

This project is designed as an undergraduate-level introduction to applied forecasting and data-driven research.

## Research Questions

The project aims to explore the following questions:

1. How do sales patterns vary across stores, product families, weekdays, and seasons?
2. How much predictive value is provided by promotions, holidays, oil prices, and store characteristics?
3. How should validation be designed for a short-horizon time series forecasting problem?
4. How do simple statistical baselines compare with interpretable machine learning models?
5. Which stores, product families, or dates are the most difficult to forecast?

## Dataset

The raw competition data are stored in:

```text
data/raw/
```

The dataset contains the following files:

* `train.csv.zip`: historical daily sales and promotion information
* `test.csv`: observations for the 15-day forecasting horizon
* `sample_submission.csv`: required Kaggle submission format
* `stores.csv`: store location, type, and cluster information
* `oil.csv`: daily oil price data
* `holidays_events.csv`: holiday and event information
* `transactions.csv`: daily store-level transaction counts

The main forecasting unit is defined by:

* `date`
* `store_nbr`
* `family`

The target variable is:

* `sales`

## Important Context

Several external and calendar-related factors may affect supermarket sales:

* product promotions
* weekends and seasonal patterns
* national and local holidays
* transferred holidays and bridge days
* public-sector salary payment dates
* oil price changes
* the April 2016 Ecuador earthquake

These factors will be examined carefully before being included in the forecasting models.

## Planned Workflow

### Stage 1: Data Audit and Project Setup

* inspect all raw data files
* check date ranges, missing values, duplicates, and variable types
* verify the relationship between training and test data
* create a clean and reproducible project structure
* document important data quality issues

### Stage 2: Exploratory Data Analysis

* analyze overall sales trends
* compare sales across stores and product families
* examine weekday, monthly, and seasonal patterns
* study promotion effects
* inspect holidays, transactions, and oil prices
* identify unusual periods and possible structural changes

### Stage 3: Validation Design and Baselines

* construct a time-based validation period
* match the validation horizon to the 15-day test horizon
* implement simple forecasting baselines
* compare recent-value, moving-average, and seasonal baselines
* establish an initial RMSLE benchmark

### Stage 4: Feature Engineering

* create calendar features
* generate lagged sales features
* generate rolling summary features
* add promotion information
* merge store metadata
* construct holiday-related variables
* add selected external variables when justified

### Stage 5: Interpretable Model Comparison

* train a small number of suitable models
* compare model performance under the same validation design
* examine feature importance or model coefficients
* evaluate whether additional features improve forecasting accuracy

Candidate models may include:

* regularized linear regression
* random forest
* gradient boosting model

### Stage 6: Error Analysis and Final Submission

* analyze forecast errors by store and product family
* identify difficult forecasting cases
* retrain the selected model using the available training data
* generate predictions for the Kaggle test set
* create a valid submission file
* summarize findings, limitations, and possible extensions

## Evaluation Metric

The competition uses Root Mean Squared Logarithmic Error:

```text
RMSLE
```

Because the metric is based on logarithmic differences, the project will pay attention to large relative errors and to the treatment of low-sales series.

## Project Principles

The project follows several principles:

* use time-based validation
* avoid data leakage
* start with simple baselines
* keep feature engineering interpretable
* compare models under consistent conditions
* document assumptions and limitations
* prioritize reproducibility and clarity

## Expected Outputs

The final repository is expected to include:

* data audit results
* exploratory analysis figures
* validation design
* baseline model results
* feature engineering code
* model comparison results
* error analysis
* Kaggle submission file
* final project report

## Repository Status

The repository currently contains the raw competition data and initial project documentation.

Modeling and analysis will be completed in later stages.
