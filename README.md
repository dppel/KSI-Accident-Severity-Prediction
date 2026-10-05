# 🚗 Road Traffic Collision Severity Prediction (KSI)
## Machine Learning Framework using UK STATS19 Open Data (2021–2025)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)
![Dataset](https://img.shields.io/badge/Dataset-UK%20STATS19-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📌 Project Overview

This repository contains an end-to-end Machine Learning project to predict **Road Traffic Collision Severity** (focusing on **KSI: Killed or Seriously Injured** vs Slight injury) using official UK **Department for Transport (DfT) STATS19** open data spanning 5 years (**2021–2025**).

The dataset covers **513,801 road collision records** and **937,265 involved vehicle records**.

### 🎯 Key Objectives:
1. **Target Identification**: Predict binary collision severity: **KSI ($1$)** (Fatal/Serious) vs **Slight ($0$)**.
2. **Strict Data Leakage Control**: Rigorously audit and exclude post-collision outcome variables (e.g. casualty counts, police attendance, post-event adjusted severity flags) to guarantee production validity.
3. **Multi-Table Data Integration**: Join pre-collision vehicle attributes (`Vehicles` table) with collision characteristics (`Collisions` table).
4. **Temporal Split**: Train on historical collisions (**2021–2024**, 412,276 records) and evaluate on the most recent complete year (**2025**, 101,525 records).
5. **Probability Calibration & Thresholding**: Calibrate predicted probabilities via Platt Scaling and optimize decision thresholds for cost-sensitive risk classification.

---

## 📊 Model Performance Benchmark (Test Set 2025)

Because KSI collisions are imbalanced (~26.24% positive class), evaluation prioritizes **PR-AUC (Average Precision)** alongside **ROC-AUC**:

| Model Architecture | ROC-AUC | PR-AUC | Relative PR Gain | Deployment Status |
| :--- | :---: | :---: | :---: | :--- |
| **a) Baseline Model (Prior)** | 0.5000 | 0.2624 | 0.0% | Baseline |
| **b) Logistic Regression** | 0.6165 | 0.3622 | +38.0% | Linear Baseline |
| **c) Initial Gradient Boosting (Collisions Only)** | 0.6435 | 0.3869 | +47.4% | v1.0 Model |
| **d) Enhanced Calibrated Model (Collisions + Vehicles)** ⭐ | **0.6913** | **0.4438** | **+69.1%** | **v2.0 Optimal** |

> 💡 **Realistic ROC-AUC (~0.69)**: Confirms complete absence of data leakage. Collision severity depends partly on unobservable kinetic and physiological variables at the instant of impact.

---

## 🛠️ Feature Engineering & Methodology

- **Multi-Table STATS19 Join**: Merged pre-collision vehicle attributes:
  - `has_vulnerable_user`: Presence of motorcycles, pedal cycles, e-scooters, mobility scooters.
  - `has_heavy_vehicle`: Presence of HGVs (trucks) or buses/coaches.
  - `min_driver_age`, `has_young_driver` (<25), `has_elderly_driver` (>70).
  - `max_vehicle_age`, `mean_engine_capacity`.
- **Cyclical Encoding**: Sin/Cos transformations for `hour` (0-23) and `month` (1-12).
- **High-Risk Interaction Terms**:
  - `vulnerable_x_speed`: Interaction between vulnerable road users and speed limit.
  - `speed_x_rural`: Speed limit in rural/unlit road environments.
  - `speed_x_unlit_night`: High speed limit during unlit nighttime conditions.
- **Probability Calibration & Cost-Sensitive Thresholding**: Applied `CalibratedClassifierCV` (Platt Scaling) and optimized the probability decision threshold at **24.62%** (achieving **67% Recall** for KSI risk identification).

---

## 📁 Repository Structure

```
├── 01_road_safety_ksi.ipynb                  # Complete Jupyter Notebook analysis
├── Executive_Briefing_Project_2_Road_Safety_KSI.pdf  # Executive Briefing Report (PDF)
├── README.md                                  # Project documentation
└── .gitignore                                 # Git ignore configuration
```

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.10+ is installed along with required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Analysis

Clone the repository and open the Jupyter notebook:

```bash
git clone https://github.com/dppel/KSI-Accident-Severity-Prediction.git
cd KSI-Accident-Severity-Prediction
jupyter notebook 01_road_safety_ksi.ipynb
```

---

## 🌐 Data Source

Data is sourced from the official UK **Department for Transport (DfT)** Road Safety open data collection published under the **Open Government Licence v3.0 (OGL)**:
- [UK DfT Road Safety Open Data (data.gov.uk)](https://data.gov.uk/dataset/cb7ae6f0-466c-4829-873b-5a0225d576a4/road-safety-data)

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
