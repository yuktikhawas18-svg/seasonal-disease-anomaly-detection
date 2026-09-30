# Seasonal Disease Anomaly Detection with Forecast Intervals and Alert Persistence

## 📌 Project Overview

This project is a time-series based disease surveillance system designed to identify unusual changes in weekly disease case counts.

The system uses historical disease surveillance data, time-series forecasting, prediction intervals, anomaly detection, and alert persistence to identify unusual patterns that may require further investigation.

> Important: An anomaly detected by this system is a statistical signal and should not be interpreted as a confirmed disease outbreak.

---

## 🎯 Objectives

The main objectives of the project are:

- Process weekly disease surveillance data.
- Perform data quality checks and cleaning.
- Analyze trends and seasonal patterns.
- Decompose time-series data using STL.
- Build baseline forecasting models.
- Build statistical and machine-learning forecasting models.
- Generate forecast prediction intervals.
- Detect observations outside expected forecast ranges.
- Assign anomaly severity levels.
- Detect persistent anomalies across consecutive weeks.
- Compare disease activity across geographic locations.
- Generate explanations for detected alerts.
- Evaluate the forecasting and anomaly detection pipeline.

---

## 🏗️ Project Architecture

The planned system follows this workflow:

Data Source
↓
Data Ingestion
↓
Schema Validation
↓
Data Quality Checks
↓
Data Cleaning
↓
Time-Series Preparation
↓
Seasonality Analysis
↓
Forecasting
↓
Forecast Intervals
↓
Anomaly Detection
↓
Alert Persistence
↓
Geographic Comparison
↓
FastAPI + PostgreSQL
↓
Streamlit Dashboard

---

## 📊 Dataset

### Primary Dataset

The primary dataset used during development is:

**CDC NNDSS Weekly Data**

The dataset contains weekly disease surveillance information including:

- Disease label
- Reporting area
- MMWR year
- MMWR week
- Current week case count
- Reporting flags

The current prototype focuses on:

**Dengue virus infections, Dengue**

The CDC dataset represents U.S. disease surveillance data and should not be interpreted as Indian surveillance data.

CDC surveillance data can be provisional and may be revised.

---

## 🧪 Synthetic Dataset

A synthetic dataset generator is also included for controlled testing.

File:

`src/data/generate_synthetic_data.py`

The generator creates weekly disease data with:

- Seasonal patterns
- Trend
- Random noise
- Multiple geographic locations
- Persistent outbreaks
- Isolated spikes
- Short persistence anomalies
- Reporting spikes
- Missing values
- Ground-truth anomaly labels

This allows anomaly detection performance to be evaluated against known injected anomalies.

Output:

`data/processed/synthetic_disease_data.csv`

---

# 📁 Project Structure

```text
seasonal-disease-anomaly-detection/
│
├── data/
│   ├── raw/
│   │   └── CDC dataset
│   │
│   └── processed/
│       ├── dengue_raw_filtered.csv
│       ├── location_summary.csv
│       ├── dengue_working.csv
│       ├── dengue_weekly.csv
│       ├── data_quality_report.csv
│       ├── synthetic_disease_data.csv
│       ├── forecast_intervals.csv
│       ├── anomaly_results.csv
│       ├── alert_persistence.csv
│       ├── geographic_comparison.csv
│       └── model result files
│
├── notebooks/
│
├── src/
│   │
│   ├── data/
│   │   ├── load_data.py
│   │   ├── filter_dengue.py
│   │   ├── select_locations.py
│   │   ├── create_working_dataset.py
│   │   ├── create_weekly_dataset.py
│   │   ├── data_quality.py
│   │   └── generate_synthetic_data.py
│   │
│   ├── analysis/
│   │   ├── eda.py
│   │   ├── seasonality.py
│   │   └── geographic_comparison.py
│   │
│   ├── models/
│   │   ├── seasonal_naive.py
│   │   ├── sarima_forecast.py
│   │   ├── xgboost_forecast.py
│   │   └── forecast_intervals.py
│   │
│   └── anomaly/
│       ├── anomaly_detection.py
│       └── alert_persistence.py
│
├── tests/
│
├── docs/
│
├── .gitignore
├── requirements.txt
└── README.md