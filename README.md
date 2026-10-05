# Road Traffic Collision Severity Prediction (KSI)
## Machine Learning Framework using UK STATS19 Open Data (2021–2025)

## Overview

This repository contains a Data Science project predicting road accident severity in the UK (**Killed or Seriously Injured - KSI** vs Slight injury) using official Department for Transport (DfT) STATS19 data from 2021 to 2025.

The dataset contains 513,801 collision records and 937,265 vehicle records.

---

## Methodology

1. **Target Variable**: Binary classification for accident severity (KSI = 1 vs Slight = 0).
2. **Data Leakage Control**: Exclusion of post-accident features (casualty counts, police attendance flags, post-accident severity adjustments).
3. **Multi-Table Data Integration**: Merging pre-collision vehicle data (`Vehicles` table) with accident data (`Collisions` table).
4. **Temporal Split**: Training on 2021–2024 data (412,276 records) and evaluating on 2025 data (101,525 records).
5. **Feature Engineering**: Cyclical encoding for time features, interaction terms, and missing value cleaning.
6. **Model Calibration & Evaluation**: Evaluating Logistic Regression and Gradient Boosting models using PR-AUC and ROC-AUC, followed by probability calibration.

---

## Model Benchmark (Test Set 2025)

| Model Architecture | ROC-AUC | PR-AUC | Relative PR Gain |
| :--- | :---: | :---: | :---: |
| **Baseline Model (Prior)** | 0.5000 | 0.2624 | 0.0% |
| **Logistic Regression** | 0.6165 | 0.3622 | +38.0% |
| **Gradient Boosting (Collisions Only)** | 0.6435 | 0.3869 | +47.4% |
| **Gradient Boosting (Collisions + Vehicles)** | **0.6913** | **0.4438** | **+69.1%** |

---

## Repository Contents

- `01_road_safety_ksi.ipynb`: Main Jupyter Notebook with data processing, modeling, and evaluation.
- `Executive_Briefing_Project_2_Road_Safety_KSI.pdf`: Executive briefing document.
- `README.md`: Project description.

---

## Data Source

Data is obtained from the UK Department for Transport (DfT) Road Safety Open Data collection under the Open Government Licence v3.0.
