<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Random%20Forest-Best%20Model-228B22?style=for-the-badge"/>
<img src="https://img.shields.io/badge/R²%20Score-0.99994-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/MAPE-0.41%25-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Capstone-Project-gold?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Deployment-Pickle-lightgrey?style=for-the-badge"/>

<br/><br/>

# 📦 Capstone Project: Delivery Cost Prediction Using Machine Learning

### *End-to-end logistics cost forecasting — from raw data to interactive deployment*

<br/>

> **Business Problem:** Logistics companies lose revenue when pricing is inaccurate — they either overcharge customers or undercharge for complex deliveries. This capstone project builds a full ML pipeline that predicts delivery costs with **99.99% explained variance**, then narrows it to just 5 features for a lean, production-ready model.

<br/>

---

</div>

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Statistical Analysis](#-statistical-analysis)
- [Feature Engineering](#-feature-engineering)
- [Models & Results](#-models--results)
- [Key Findings](#-key-findings)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)

---

## 📘 Project Overview

This **capstone project** covers the complete data science lifecycle for a logistics pricing problem:

```
Data Understanding → Cleaning → EDA → Statistical Analysis →
Outlier Handling → Feature Engineering → Model Building →
Model Comparison → Feature Selection → Final Model → Deployment
```

The goal is to build a model that logistics companies can use to **accurately estimate delivery costs** before dispatch — enabling data-driven pricing, loss reduction, and operational planning.

---

## 📂 Dataset

**Source:** [Delivery Logistics Dataset — Kaggle](https://www.kaggle.com/datasets/kundanbedmutha/delivery-logistics-dataset-india-multi-partner)  
**File:** `Delivery_Logistics.csv`  
**Target Variable:** `delivery_cost` (Total delivery charge in USD)

| Feature | Description | Type |
|---|---|---|
| `delivery_partner` | Logistics company (Amazon, DHL, FedEx, etc.) | Categorical |
| `package_type` | Type of goods (Electronics, Furniture, Fragile, etc.) | Categorical |
| `vehicle_type` | Delivery vehicle (Bike, Van, Truck, EV, etc.) | Categorical |
| `delivery_mode` | Speed tier (Same Day, Express, Two Day, Standard) | Categorical |
| `region` | Geographic zone (North, South, East, West, Central) | Categorical |
| `weather_condition` | Conditions at delivery (Clear, Rainy, Stormy, etc.) | Categorical |
| `distance_km` | Delivery distance in kilometres | Float |
| `package_weight_kg` | Package weight in kg | Float |
| `delayed` | Whether delivery was delayed (0/1) | Binary |
| `delivery_status` | Final outcome (Delivered, Delayed, Failed) | Categorical |
| `delivery_rating` | Customer satisfaction score (1–5) | Integer |
| **`delivery_cost`** | 🎯 **Target** — Total delivery charge (USD) | Float |

> **Dropped columns:** `delivery_id` (non-predictive ID), `delivery_time_hours` and `expected_time_hours` (constant-value columns)

---

## 🔬 Project Workflow

```
Raw Logistics Data
   │
   ├── Data Cleaning
   │       ├── Dropped: delivery_id, delivery_time_hours, expected_time_hours
   │       ├── Verified: no missing values, no duplicates
   │       └── Fixed: delayed column mapped from yes/no → 1/0
   │
   ├── Exploratory Data Analysis (EDA)
   │       ├── Univariate: KDE, histogram, boxplot per numeric feature
   │       ├── Countplots per categorical feature
   │       ├── Bivariate: cost vs distance (scatter + regplot)
   │       ├── Bivariate: cost vs vehicle type, weather, package weight
   │       └── Multivariate: Pearson correlation heatmap
   │
   ├── Statistical Analysis
   │       ├── Skewness & kurtosis measured across all numeric cols
   │       ├── Point-biserial correlation: delayed vs delivery_cost
   │       └── One-Way ANOVA: all multi-class categorical cols vs cost
   │
   ├── Feature Engineering
   │       └── 4 domain-informed features derived
   │
   ├── Encoding
   │       └── Ordinal mapping for all 7 categorical columns
   │
   ├── Model Development
   │       └── 5 regressors trained on 80/20 split with StandardScaler
   │
   ├── Feature Importance & Selection
   │       └── Top 5 features identified → lean final model
   │
   └── Deployment
           ├── Interactive CLI input for real-time cost prediction
           ├── Model saved with Pickle
           └── Load & predict pipeline validated
```

---

## 📐 Statistical Analysis

### ANOVA — Categorical Features vs Delivery Cost

| Feature | Result |
|---|---|
| `delivery_mode` | ✅ Significant impact on delivery cost |
| `vehicle_type` | ✅ Significant impact on delivery cost |
| `weather_condition` | ✅ Significant impact on delivery cost |
| `package_type` | ✅ Significant impact on delivery cost |
| `region` | ✅ Significant impact on delivery cost |
| `delivery_partner` | ✅ Significant impact on delivery cost |
| `delivery_status` | ✅ Significant impact on delivery cost |

### Point-Biserial Correlation — Binary vs Delivery Cost

| Feature | Result |
|---|---|
| `delayed` | ✅ Statistically significant impact on delivery cost (p < 0.05) |

> Every feature in the dataset has a statistically significant impact on delivery cost — validating that no feature was dropped from modeling.

---

## 🧪 Feature Engineering

4 domain-driven features were engineered to capture logistics-specific interactions:

| Feature | Formula | Business Logic |
|---|---|---|
| `weight_per_km` | `package_weight_kg / distance_km` | Weight burden per unit of distance — heavier loads over longer routes cost more |
| `bad_weather_flag` | 1 if weather in {rainy, stormy, foggy} | Weather surcharge trigger |
| `priority_flag` | 1 if `delivery_mode == express` | Express delivery premium indicator |
| `priority_distance_interaction` | `priority_flag × distance_km` | Express delivery cost amplified by distance |

---

## 📈 Models & Results

Five regression models evaluated on an 80/20 split with StandardScaler:

| Model | MAE | MAPE | R² Score |
|---|---|---|---|
| **Random Forest** ✅ | **2.3267** | **0.41%** | **0.99994** |
| XGBoost | 4.0913 | 0.74% | 0.99985 |
| Decision Tree | 5.5866 | 0.93% | 0.99971 |
| Linear Regression | 11.3496 | 2.23% | 0.99898 |
| K-Nearest Neighbors | 74.0985 | 13.48% | 0.95156 |

### 🏆 Winner: Random Forest Regressor

Random Forest dominates across all metrics — lowest MAE, lowest MAPE, and highest R². The **top-5 feature model** further reduced MAE by ~1 point, confirming the selected features are the signal-dense core of the data.

> **KNN underperformed significantly** — a known limitation when features have different scales and the search space is high-dimensional, even with StandardScaler applied.

---

## 💡 Key Findings

**Top 5 Features by Random Forest Importance:**

| Rank | Feature | Insight |
|---|---|---|
| 1 | `distance_km` | Primary driver — longer distance = higher cost |
| 2 | `package_weight_kg` | Heavier packages incur higher base costs |
| 3 | `delivery_mode` | Express/same-day modes command premium pricing |
| 4 | `priority_distance_interaction` | Express deliveries over long distances carry the highest surcharges |
| 5 | `priority_flag` | Express mode alone is a strong cost signal |

**Business Insights:**
- **Distance and weight** are the two structural pillars of delivery pricing — logistics firms should ensure their pricing tables reflect this linearity
- **Express mode** has a multiplicative effect — not just a flat surcharge but an amplifier of distance costs
- **Weather surcharges** are statistically significant but not in the top-5 — indicating they affect cost but with less magnitude than operational factors

---

## 🚀 Deployment

The final model includes an **interactive CLI** for real-time cost estimation:

```python
# Interactive cost predictor
distance = float(input('Enter Distance in Km: '))
weight = float(input('Enter Weight in Kg: '))
mode = int(input('Enter Delivery Mode (1=Same Day, 2=Express, 3=Two Day, 4=Standard): '))
priority_flag = int(input('Enter Priority Flag (1=Express, 0=Other): '))
priority_distance_interaction = distance * priority_flag

test = [[distance, weight, mode, priority_distance_interaction, priority_flag]]
test_scaled = scaler.transform(test)

cost = loaded_model.predict(test_scaled)
print(f'Estimated Delivery Cost: ${cost[0]:.2f}')
```

**Load saved model:**
```python
import pickle

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Use with scaler — ensure scaler is fitted on the same 5 features
```

---

## 🛠 Tech Stack

```
Language      Python 3.10+
ML Library    scikit-learn, XGBoost
Statistics    scipy (pointbiserialr, f_oneway / ANOVA)
Data          pandas, NumPy
Viz           matplotlib, seaborn
Deployment    Pickle + interactive CLI
Environment   Google Colab / Jupyter Notebook
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Pfuglo1/delivery-cost-prediction.git
cd delivery-cost-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy openpyxl

# 3. Launch the notebook
jupyter notebook Project_6_Delivery_Cost_Prediction_Using_Machine_Learning.ipynb
```

> 📂 Place `Delivery_Logistics.csv` in the same directory before running.

---

## 🗂 Project Structure

```
delivery-cost-prediction/
│
├── Project_6_Delivery_Cost_Prediction_Using_Machine_Learning.ipynb  # Full pipeline
├── Capstone_Project_Delivery_Cost_Prediction.docx                   # Project brief
├── model.pkl                                                         # Saved Random Forest model
├── final_data.xlsx                                                   # Top-5 feature dataset
├── README.md                                                         # You are here
└── Delivery_Logistics.csv                                            # Raw dataset
```

---

<div align="center">

**Capstone Project — Full ML Pipeline from Raw Data to Live Prediction 🚚**

*R² = 0.99994 — predicting delivery costs to within cents.*

*Found this useful? Give it a ⭐!*

</div>
