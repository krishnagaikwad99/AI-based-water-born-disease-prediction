# 💧 AquaWatch — AI-Based Early Warning System for Water-Borne Diseases

> **An end-to-end AI system that predicts village-level water-borne disease outbreak risk using water quality, environmental, and historical case data — with Explainable AI, GIS visualization, forecasting, and LLM-generated reports.**

AquaWatch is a **prototype web-based Early Warning System (EWS)** designed to identify villages at **Low 🟢, Medium 🟠, or High 🔴 risk** of a water-borne disease outbreak.

The system combines **Machine Learning + Explainable AI + GIS + Time-Series Forecasting + LLMs** inside an interactive **Streamlit dashboard**.

---

## 🎯 One-Line Project Pitch

> **"I built an end-to-end ML system that predicts whether a village is at Low, Medium, or High risk of a water-borne disease outbreak using water quality and environmental data. It includes SHAP-based explainability, GIS risk mapping, short-term forecasting, and an LLM that converts predictions into plain-language reports for health workers — all through a Streamlit dashboard."**

---

# 🚨 Problem Statement

Water-borne diseases such as:

* 🦠 Cholera
* 🦠 Typhoid
* 🦠 Diarrhea
* 🦠 Hepatitis A

can become more prevalent when environmental and water-quality conditions deteriorate.

Important risk factors include:

* 💧 Contaminated water
* 🌫️ High turbidity
* 🧪 Low residual chlorine
* 🦠 Bacterial contamination
* 🌧️ Heavy rainfall
* 🌊 Flooding
* 🚽 Poor sanitation
* 📈 Previous disease cases
* 🌦️ Seasonal patterns such as monsoon spikes

### 🎯 Goal

The goal of AquaWatch is to provide an **early warning**, rather than simply reporting an outbreak after it has already occurred.

The system takes village-level environmental and historical information and produces:

> **Risk Level → Explanation → Forecast → Recommendation → Human-readable Report**

---

# 💡 Project Objective

The system answers four important questions:

### 1️⃣ What is the current risk?

Predict whether the village is:

🟢 **Low Risk**
🟠 **Medium Risk**
🔴 **High Risk**

### 2️⃣ Why is the village at this risk?

Using **SHAP Explainable AI**, the system identifies the most important factors contributing to an individual prediction.

### 3️⃣ What could happen next week?

A lightweight forecasting model predicts the **next week's risk level**.

### 4️⃣ What should a health worker know?

An LLM converts the structured prediction and SHAP results into a short, plain-language report.

---

# 🏗️ System Architecture

```text
                 📊 Data Sources
                      │
                      ▼
            🧹 Data Cleaning
                      │
                      ▼
          🔍 Exploratory Data Analysis
                      │
                      ▼
          ⚙️ Feature Engineering
                      │
                      ▼
              🤖 ML Model
          Logistic Regression
                 + Random Forest
                      │
          ┌───────────┼────────────┐
          ▼           ▼            ▼
       🎯 Risk      🔎 SHAP      📈 Forecast
      Prediction   Explainability   Model
          │           │            │
          └───────────┼────────────┘
                      ▼
               🗺️ GIS Risk Map
                      │
                      ▼
            💡 Recommendation Engine
                      │
                      ▼
             🤖 LLM Report Generator
                      │
                      ▼
             🖥️ Streamlit Dashboard
```

### Architecture Philosophy

The project deliberately uses a **simple script/notebook-driven architecture** instead of unnecessary microservices.

There is:

* ❌ No unnecessary REST API layer
* ❌ No microservices
* ❌ No Docker
* ❌ No IoT hardware

This keeps the system:

* Easy to understand
* Easy to debug
* Easy to demonstrate
* Appropriate for the current project complexity

---

# 📁 Project Structure

```text
water_disease_ews/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── raw/
│   │   └── simulated_dataset.csv
│   │
│   └── processed/
│       └── processed_data.csv
│
├── models/
│   └── *.pkl
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_shap.ipynb
│   └── ...
│
├── src/
│   └── generate_dataset.py
│
├── app.py
├── village_risk_map.html
├── test_pipeline.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

## Why Simulated Data?

One of the biggest challenges was finding a single public dataset containing:

* Village-level water quality
* Disease case counts
* Population
* Sanitation information
* Environmental conditions
* Historical weekly observations

These datasets exist individually across different sources, but combining them into one consistent village-level dataset would require substantial data engineering and validation.

Therefore, AquaWatch uses a **synthetic dataset generated using domain-based rules**.

> ⚠️ The dataset is explicitly simulated and is **not real medical surveillance data**.

---

## 🧠 Domain-Based Data Generation

The synthetic data was not generated using completely random values.

Instead, domain-inspired relationships were encoded into the generator:

```python
risk_score = (
    0.35 * turbidity
    + 0.30 * (1 - chlorine)
    + 0.20 * rainfall
    + 0.25 * (1 - sanitation)
    + 0.30 * bacterial_contamination
    + 0.15 * previous_week_cases
)
```

This allows the dataset to contain meaningful relationships between environmental conditions and disease risk.

### Dataset Size

| Property            | Value |
| ------------------- | ----: |
| 🏘️ Villages        |    20 |
| 📅 Weeks            |    52 |
| 📊 Total rows       | 1,040 |
| 📋 Original columns |    18 |
| 🔢 Random seed      |    42 |

The fixed random seed makes the dataset **reproducible**.

---

# 🔬 Phase-by-Phase Development

## 1️⃣ Dataset Design & Loading

The dataset follows a panel structure:

> **One row = One village + One week**

The 18 original columns contain information related to:

* Water quality
* Environment
* Population
* Sanitation
* Historical disease cases
* Disease type
* Target risk level

The target is divided into:

```text
Low    → 0
Medium → 1
High   → 2
```

### Why Classification?

Raw disease case counts were highly skewed.

Instead of predicting an exact case count, the project predicts actionable risk categories:

🟢 Low → 🟠 Medium → 🔴 High

This is easier to interpret for an early-warning dashboard.

---

# 🧹 2️⃣ Data Cleaning & Preprocessing

The dataset was checked for:

* Missing values
* Duplicate records
* Incorrect ordering
* Invalid data types

The original simulated dataset contained:

```text
Missing values → 0
Duplicate rows  → 0
```

However, defensive preprocessing was still implemented.

For example:

```text
Numeric missing values → Median imputation
Duplicate rows         → Removed
```

The data was also sorted by:

```text
Village → Date
```

This is important because time ordering matters for forecasting.

---

# 📊 3️⃣ Exploratory Data Analysis

EDA was used not only for visualization but also to verify whether the generated dataset behaved as expected.

### Observations

As risk increased:

* 🌫️ Turbidity increased
* 🧪 Chlorine decreased
* 🦠 Bacterial contamination increased
* 📈 Historical cases increased

The dataset also showed a visible **monsoon-season increase in cases**.

### Correlation Analysis

The strongest relationships with case counts included:

**Positive:**

* Turbidity
* Bacterial contamination
* Historical cases

**Negative:**

* Residual chlorine
* Sanitation index

This helped guide subsequent feature engineering.

---

# ⚙️ 4️⃣ Feature Engineering

Several raw variables were combined into meaningful risk indicators.

### 💧 Water Quality Risk

```text
water_quality_risk_score
=
turbidity
+
(1 - chlorine)
+
bacterial_contamination
```

### 🌧️ Environmental Risk

```text
environmental_risk_score
=
rainfall
+
temperature-adjusted component
```

### 🚽 Sanitation Gap

```text
sanitation_gap = 1 - sanitation_index
```

This was intentionally reversed so that:

> **Higher value = Worse condition**

This consistent direction also makes SHAP explanations easier to interpret.

### Categorical Encoding

Categorical features such as:

```text
water_source_type
disease_type
```

were one-hot encoded.

---

# 🤖 5️⃣ Machine Learning Models

Two models were trained:

### Logistic Regression

Used as the simple baseline because it is:

* Fast
* Easy to interpret
* Computationally inexpensive

### Random Forest

Used because it can capture:

* Non-linear relationships
* Feature interactions
* Complex decision boundaries

---

## 🚨 Important: Preventing Data Leakage

One of the most important decisions in the project was excluding:

```text
case_count
```

from the model features.

Why?

Because the target risk level was generated using disease case information.

If `case_count` were included, the model could simply learn the answer from the variable that helped generate the answer.

That would result in artificially high performance.

> **The model must predict risk from available warning signals, not read the answer key.**

---

# 📈 6️⃣ Model Evaluation

An 80/20 stratified train-test split was used.

| Model               |  Accuracy | Weighted F1 |
| ------------------- | --------: | ----------: |
| Logistic Regression |     67.3% |       0.675 |
| 🌲 Random Forest    | **68.3%** |   **0.680** |

Random Forest was selected as the primary model.

However, the improvement over Logistic Regression was relatively small.

This was treated as an **honest finding**, rather than artificially exaggerating the model's performance.

---

# 🎯 7️⃣ Prediction Pipeline

The inference pipeline follows:

```text
New Input
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Column Alignment
   ↓
Random Forest
   ↓
Risk Prediction
   ↓
Probability / Confidence
```

A common real-world issue was also handled:

### Column Mismatch

New input columns may not arrive in exactly the same order as training data.

Therefore, the inference pipeline explicitly enforces:

```python
X_train.columns
```

before making predictions.

---

# 🔎 8️⃣ Explainable AI with SHAP

AquaWatch uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

### Why SHAP?

Random Forest feature importance tells us:

> "Which features are generally important?"

SHAP tells us:

> "Why was THIS particular village classified as High Risk?"

This distinction is important for an early-warning system.

### Example

A High-Risk prediction might be driven by:

```text
🔴 High water quality risk
🔴 High bacterial contamination
🔴 High historical cases
🔴 Low residual chlorine
```

The system displays these contributing factors to make the prediction more understandable.

### Implementation

```python
shap.TreeExplainer(random_forest_model)
```

was used because TreeExplainer is optimized for tree-based models.

---

# 📈 9️⃣ One-Week Forecasting

Instead of immediately using a complex LSTM, the project first tested whether the dataset actually contained enough temporal information to support forecasting.

### Features

Lag variables were created:

```text
cases_lag_1
cases_lag_2
cases_lag_3
```

The target was shifted to:

```text
future_risk_level = next week's risk
```

---

## 🧪 Naive Baseline Comparison

The forecasting model was compared against a simple baseline:

> **Assume next week's risk = this week's risk**

Results:

| Method            |  Accuracy |
| ----------------- | --------: |
| Naive baseline    |     60.3% |
| Forecasting model | **67.7%** |

### Improvement

```text
67.7% - 60.3%
= 7.4 percentage points
```

This provided evidence that the model captured some predictive information beyond simply repeating the current week's risk.

### Why No LSTM?

There are only:

```text
52 weeks × 20 villages
```

Using a deep learning time-series model such as LSTM would add complexity without enough temporal depth to justify it.

The project therefore used a simpler forecasting approach and validated it against a naive baseline.

---

# 🗺️ 🔟 GIS Risk Map

AquaWatch uses **Folium** to visualize village-level risk geographically.

Each village is displayed using a risk-based marker:

```text
🟢 Low
🟠 Medium
🔴 High
```

Popups contain information such as:

* Village name
* Risk level
* Population

The map is exported as:

```text
village_risk_map.html
```

This allows the map to work independently of the notebook and be embedded into the Streamlit application.

---

# 💡 1️⃣1️⃣ Recommendation Engine

The recommendation engine is intentionally **rule-based**.

For example:

```text
IF chlorine < threshold
        ↓
Recommend immediate chlorination
```

The system uses ordered threshold conditions through `np.select`.

### Why Not ML?

Recommendations need to be:

* Transparent
* Auditable
* Traceable
* Easy to debug

An ML-generated recommendation would introduce another black box on top of the prediction model.

Therefore:

> **ML predicts the risk. Rules explain what action should be considered.**

---

# 🖥️ 1️⃣2️⃣ Streamlit Dashboard

The entire project is integrated into an interactive Streamlit application.

### Dashboard Components

#### 🎯 Predict

Users can enter water/environmental information and receive:

* Risk level
* Prediction confidence
* SHAP explanation
* Recommendations

#### 📊 Overview

Displays:

* Dataset information
* Risk distribution
* Key statistics
* Visualizations

#### 🗺️ Map

Displays the village-level GIS risk map.

#### 🤖 Report

Generates a short plain-language report using the LLM.

---

## 🔄 Streamlit Session State

A Streamlit application reruns the script when users interact with widgets.

This created an important issue:

> Prediction results could disappear when another button was clicked.

This was solved using:

```python
st.session_state
```

Prediction results are persisted between interactions, allowing the user to generate an LLM report without losing the previous prediction.

---

# 🤖 1️⃣3️⃣ LLM-Powered Reports

AquaWatch integrates the **Groq API** to convert structured prediction results into a plain-language report.

### Input to LLM

The LLM receives structured information such as:

```text
Risk Level
Confidence
Top SHAP Factors
Relevant Environmental Conditions
Recommendations
```

### Output

The LLM generates a concise report intended for non-technical users such as health workers.

---

## 🛡️ LLM Safety Constraint

The prompt explicitly instructs the model:

> **Use only the facts provided. Do not invent new numbers, medical claims, or unsupported information.**

This is important because AquaWatch is a **health-adjacent application**.

The LLM is used for:

> 📝 Report generation

—not for making independent medical diagnoses.

---

# 🧰 Tech Stack

| Category               | Technology          |
| ---------------------- | ------------------- |
| 🐍 Programming         | Python              |
| 📊 Data Processing     | Pandas, NumPy       |
| 🤖 Machine Learning    | Scikit-learn        |
| 🌲 Primary Model       | Random Forest       |
| 📈 Baseline Model      | Logistic Regression |
| 🔎 Explainability      | SHAP                |
| 📊 Visualization       | Matplotlib          |
| 🗺️ GIS                | Folium              |
| 🖥️ Dashboard          | Streamlit           |
| 🤖 LLM                 | Groq API            |
| 💾 Storage             | CSV                 |
| 📦 Model Serialization | Joblib              |

---

# 🧠 Key Engineering Decisions

| Decision                      | Reason                                                                 |
| ----------------------------- | ---------------------------------------------------------------------- |
| 🧪 Simulated data             | No single public dataset combines all required village-level features  |
| 🎯 Classification             | Risk categories are more actionable than highly skewed raw case counts |
| 🚨 Drop `case_count`          | Prevents target leakage                                                |
| 🌲 Random Forest              | Handles non-linear relationships while remaining explainable           |
| ❌ No LSTM                     | Dataset has limited temporal depth                                     |
| 📏 Naive baseline             | Provides a meaningful forecasting comparison                           |
| 💡 Rule-based recommendations | Transparent and auditable                                              |
| 🔎 SHAP                       | Explains individual predictions                                        |
| 🤖 Constrained LLM            | Reduces hallucination risk                                             |
| 🔄 `st.session_state`         | Preserves prediction results across Streamlit interactions             |

---

# 📊 Results Summary

## Classification

```text
Random Forest
Accuracy:       68.3%
Weighted F1:    0.680
```

Compared with:

```text
Logistic Regression
Accuracy:       67.3%
Weighted F1:    0.675
```

---

## 📈 Forecasting

```text
Model:           67.7%
Naive baseline:  60.3%
Improvement:      7.4 percentage points
```

---

## ⚖️ Class Distribution

| Risk Level | Samples |
| ---------- | ------: |
| 🟢 Low     |     254 |
| 🟠 Medium  |     350 |
| 🔴 High    |     436 |

No class represents less than 10% of the dataset, so additional imbalance-handling techniques were not considered necessary for this prototype.

---

## 🔐 Data Leakage Check

```text
case_count included in features?
→ ❌ No

Leakage test
→ ✅ Passed
```

---

# ⚠️ Honest Limitations

AquaWatch is a **prototype**, not a clinically validated public-health surveillance system.

### Current limitations:

* 🧪 Dataset is simulated
* 📈 Forecasting currently focuses on 1-week-ahead prediction
* 📏 Recommendation engine uses fixed thresholds
* 💾 No persistent database yet
* 🤖 Hosted LLM model availability can change over time
* 🏥 No real-world medical or public-health validation has been performed

These limitations are intentionally documented rather than hidden.

---

# 🚀 Future Improvements

### 1️⃣ Real-World Data

Replace the simulated dataset with validated:

* Water-quality data
* Disease surveillance data
* Weather data
* Population data

The existing pipeline can be adapted because the modeling architecture is already separated from data generation.

### 2️⃣ Multi-Horizon Forecasting

Extend predictions to:

```text
1 week
2 weeks
3 weeks
4 weeks
```

Each horizon should be evaluated independently against its corresponding naive baseline.

### 3️⃣ Persistent Database

Add:

```text
SQLite / PostgreSQL
```

to store:

* Historical predictions
* Village risk trends
* Model outputs
* Alert history

### 4️⃣ Automated Alerts

Generate alerts when:

```text
Risk = 🔴 HIGH
```

Potential channels:

* 📧 Email
* 📱 SMS
* 🔔 Dashboard notifications

### 5️⃣ Automated Model Retraining

Future versions could retrain the model periodically as validated real-world observations become available.

---

# ▶️ How to Run

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd water_disease_ews
```

## 2. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## 3. Configure Groq API

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your-key-here
```

⚠️ **Never commit your `.env` file or API key to GitHub.**

Add this to `.gitignore`:

```text
.env
```

## 4. Run the Streamlit Application

```powershell
python -m streamlit run app.py
```

The dashboard should then open in your browser.

---

# 🧪 Optional Pipeline Test

Run:

```powershell
python test_pipeline.py
```

The test checks:

* ✅ Required files exist
* ✅ Model loads correctly
* ✅ Model predicts without errors
* ✅ `case_count` is excluded
* ✅ Target distribution is not degenerate
* ✅ Pipeline executes successfully

---

# 🎤 Interview Preparation

## Q1. Why did you use simulated data?

**Answer:**

> "There wasn't a single public dataset combining village-level water quality, disease cases, population, sanitation, and environmental data at the required weekly resolution. Instead of pretending scraped or manually created data was real, I generated a synthetic dataset using domain-inspired relationships and clearly labeled it as simulated. The pipeline was designed so real data with the same schema could replace it later."

---

## Q2. Why classification instead of regression?

**Answer:**

> "The raw case counts were highly skewed. Also, an early-warning system needs an actionable risk category more than an exact case-count prediction. So I classified villages into Low, Medium, and High risk."

---

## Q3. How did you prevent data leakage?

**Answer:**

> "`case_count` was excluded from the model features because it was directly involved in generating the target risk level. Including it would allow the model to indirectly see the answer and produce misleadingly high performance."

---

## Q4. Why Random Forest?

**Answer:**

> "I compared Random Forest with Logistic Regression. Logistic Regression provided a simple baseline, while Random Forest could capture non-linear relationships and feature interactions. Random Forest performed slightly better, with 68.3% accuracy compared with 67.3% for Logistic Regression."

---

## Q5. Why didn't you use an LSTM?

**Answer:**

> "I only had 52 weekly observations per village, so the temporal depth wasn't sufficient to justify a deep learning time-series model. Instead, I created lag features and compared the forecasting model against a naive baseline. The model achieved 67.7% accuracy compared with 60.3% for the naive approach."

---

## Q6. What is the difference between feature importance and SHAP?

**Answer:**

> "Random Forest feature importance gives a global view of which features are generally important across the model. SHAP can explain an individual prediction, so I can tell why a specific village was classified as High Risk during a particular week."

---

## Q7. Why is the recommendation engine rule-based?

**Answer:**

> "Recommendations need to be transparent and auditable. A threshold-based rule lets a health worker understand exactly why a recommendation was triggered. I didn't want to put another black-box ML model between the prediction and the recommended action."

---

## Q8. Why did you use `st.session_state`?

**Answer:**

> "Streamlit reruns the entire script when users interact with widgets. Without session state, a prediction could disappear when the user clicked another button, such as Generate Report. I used `st.session_state` to persist the prediction and explanation across interactions."

---

## Q9. How do you know the forecasting model actually learned something?

**Answer:**

> "I compared it against a naive baseline where the next week's risk is assumed to be the same as the current week's risk. The baseline achieved 60.3%, while my model achieved 67.7%, giving a 7.4 percentage-point improvement."

---

## Q10. What would you improve with more time?

**Answer:**

> "My priorities would be replacing the simulated data with validated real-world data, extending forecasting to multiple horizons, adding a persistent database for village-level trends, implementing automated alerts, and establishing a proper model-retraining pipeline."

---

# ⭐ What Makes AquaWatch Interesting?

AquaWatch is not just a classification model.

It combines several components into one end-to-end system:

```text
📊 Data Engineering
       +
🤖 Machine Learning
       +
🔎 Explainable AI
       +
📈 Forecasting
       +
🗺️ GIS
       +
💡 Rule-Based Recommendations
       +
🤖 LLM Report Generation
       +
🖥️ Streamlit
```

The focus is on building a **complete ML application**, rather than only training a model inside a notebook.

---

# 🧑‍💻 Skills Demonstrated

Through this project, I worked with:

* Python
* Pandas
* NumPy
* Scikit-learn
* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Classification
* Time-Series Features
* Model Evaluation
* SHAP
* Explainable AI
* Folium
* GIS Visualization
* Streamlit
* Session State
* LLM Integration
* Prompt Engineering
* API Integration
* Model Serialization
* Testing
* Documentation

---

# ⚠️ Important Disclaimer

> AquaWatch is an **educational/prototype project using simulated data**. It is not a medical diagnostic system and should not be used for real-world public-health decisions without validated surveillance data, expert review, regulatory consideration, and appropriate clinical/public-health validation.

---

# 👨‍💻 Project Status

```text
🟢 Core ML Pipeline       → Completed
🟢 Feature Engineering    → Completed
🟢 Model Evaluation       → Completed
🟢 SHAP Explainability    → Completed
🟢 Forecasting            → Completed
🟢 GIS Visualization      → Completed
🟢 Recommendation Engine  → Completed
🟢 Streamlit Dashboard    → Completed
🟢 LLM Report Generation  → Completed
🟢 Pipeline Testing       → Completed

🔵 Real-world Dataset      → Future Work
🔵 Persistent Database     → Future Work
🔵 Automated Alerts        → Future Work
🔵 Multi-week Forecasting  → Future Work
```

---

# 🌊 AquaWatch

### **From water-quality signals to explainable early warnings.**

> **Detect → Explain → Forecast → Recommend → Report**
