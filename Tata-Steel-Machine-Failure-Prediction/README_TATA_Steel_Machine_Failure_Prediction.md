# 🏭 TATA Steel Machine Failure Prediction

> **End-to-end Exploratory Data Analysis and Machine Learning project for predictive maintenance and machine failure prediction.**

## 📌 Project Overview

This project analyzes industrial machine sensor data and develops a machine learning pipeline to predict whether a machine is likely to fail.

The project is divided into two complementary stages:

1. **Exploratory Data Analysis (EDA)** — understanding machine operating conditions, failure patterns, feature relationships, and business-relevant insights.
2. **Machine Learning (ML)** — preparing the data, engineering features, handling severe class imbalance, training multiple classification models, tuning the models, explaining predictions, and generating predictions for unseen test data.

The project uses a synthetic TATA Steel manufacturing sensor dataset containing operating measurements such as temperature, rotational speed, torque, tool wear, machine type, and recorded failure-mode indicators.

---

## 🎯 Business Problem

Unexpected machine failures can lead to unplanned downtime, maintenance costs, and operational disruption.

The objective of this project is to identify operating conditions associated with machine failure and build a predictive model that can act as an **early-warning layer for predictive maintenance**.

The target variable is:

- `Machine failure = 1` → machine failure
- `Machine failure = 0` → normal operation

A major challenge is the severe class imbalance: only **2,148 of 136,429 training records (1.57%)** represent failures. Therefore, model evaluation cannot rely on accuracy alone.

---

## 📊 Dataset Overview

### Training Dataset

- **Rows:** 136,429
- **Columns:** 14
- **Missing values:** 0
- **Duplicate rows:** 0
- **Failure records:** 2,148
- **Failure rate:** 1.57%
- **Healthy records:** 134,281
- **Healthy rate:** 98.43%

### Test Dataset

- **Rows:** 90,954
- **Columns:** 13
- The target column `Machine failure` is not present because it is the variable to be predicted.

### Machine Types

| Type | Description | Records |
|---|---|---:|
| L | Low tier | 95,354 |
| M | Medium tier | 32,152 |
| H | High tier | 8,923 |

---

## 🔍 Dataset Features

| Feature | Description |
|---|---|
| `id` | Row identifier |
| `Product ID` | Per-unit identifier; redundant with `Type` |
| `Type` | Machine quality tier: L, M, H |
| `Air temperature [K]` | Air temperature in Kelvin |
| `Process temperature [K]` | Process temperature in Kelvin |
| `Rotational speed [rpm]` | Machine rotational speed |
| `Torque [Nm]` | Mechanical torque |
| `Tool wear [min]` | Cumulative tool usage time |
| `TWF` | Tool Wear Failure flag |
| `HDF` | Heat Dissipation Failure flag |
| `PWF` | Power Failure flag |
| `OSF` | Overstrain Failure flag |
| `RNF` | Random Failure flag |
| `Machine failure` | Target variable |

---

# 📈 Part 1 — Exploratory Data Analysis

The EDA notebook investigates the dataset using structured **Univariate, Bivariate, and Multivariate (UBM)** analysis.

The analysis includes:

- Dataset structure and quality checks
- Missing-value analysis
- Duplicate-value analysis
- Statistical summaries
- Numerical feature distributions
- Machine type distribution
- Failure class distribution
- Sensor-to-failure relationships
- Failure rate by machine type
- Failure-mode frequency
- Correlation analysis
- Pairwise feature relationships
- Business-oriented interpretation of findings

The EDA notebook contains meaningful visualizations covering sensor distributions, failure patterns, categorical relationships, correlations, and multivariate relationships.

### Key EDA Findings

#### ⚠️ Severe Class Imbalance

Only **1.57%** of observations represent machine failures. This makes the problem a rare-event classification task and means that a model predicting only the majority class could achieve high accuracy while failing to identify actual failures.

#### ⚙️ Torque

Torque showed one of the clearest visual differences between failed and healthy machines. Failed machines displayed a higher and wider Torque distribution.

#### 🔧 Tool Wear

Failed machines tended to have higher tool-wear values, supporting the use of Tool Wear as an important predictive feature.

#### 🌡️ Temperature Relationship

Air temperature and Process temperature are very strongly correlated. The EDA identified substantial redundancy between these two measurements.

#### 🏭 Machine Type

Low-tier (`L`) machines represent the largest portion of the dataset and show a generally higher failure rate than Medium (`M`) and High (`H`) machines.

#### 🔥 Failure Modes

Heat Dissipation Failure (`HDF`) is the most frequent individual failure subtype in the dataset.

Failure-mode counts identified in the EDA:

| Failure Mode | Count |
|---|---:|
| HDF | 704 |
| OSF | 540 |
| PWF | 327 |
| RNF | 308 |
| TWF | 212 |

---

# 🤖 Part 2 — Machine Learning

The ML pipeline builds on the EDA findings and prepares the data for binary classification.

## 🧹 Data Cleaning

The following preprocessing decisions were applied:

- `id` was removed because it is only a row identifier.
- `Product ID` was removed because its first letter duplicates `Type` and its numeric suffix acts as a serial identifier.
- No missing-value imputation was required.
- No duplicate removal was required.
- Outliers were identified but not automatically removed because extreme sensor values may represent genuine early warning signals of machine failure.

---

## 🔤 Categorical Encoding

`Type` was encoded as:

```text
L → 0
M → 1
H → 2
```

The project treats these machine categories as an ordered quality/tier variable.

---

## 🧠 Feature Engineering

Two important engineered features were created.

### 1. Temperature Difference

```text
Temp_diff = Process_temperature_K - Air_temperature_K
```

This captures the difference between process and air temperature while reducing the redundancy caused by their strong correlation.

### 2. Power

```text
Power = Torque_Nm × Rotational_speed_rpm
```

This represents a physically meaningful interaction between Torque and Rotational Speed.

---

## 🔎 Feature Selection & Multicollinearity

Variance Inflation Factor (VIF) was used to investigate multicollinearity.

Because:

```text
Temp_diff = Process temperature - Air temperature
```

keeping all three temperature-related variables would introduce perfect linear dependency.

Therefore, the original Air and Process temperature columns were removed after creating `Temp_diff`.

Dimensionality reduction such as PCA was not used because the final feature space is small and maintaining feature interpretability is important for a predictive-maintenance use case.

---

# ⚖️ Handling Class Imbalance

The failure class represents only **1.57%** of the training dataset.

To address this:

### SMOTE

**Synthetic Minority Over-sampling Technique (SMOTE)** was applied to the **training fold only**.

This is important because applying SMOTE before the train-validation split could introduce information leakage into the validation data.

The validation set remains untouched so that model performance can be evaluated on data that was not synthetically oversampled.

---

# ✂️ Train-Validation Split

An **80/20 stratified split** was used:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

Stratification preserves the rare failure-class proportion across the training and validation sets.

---

# 🤖 Models Implemented

Three classification algorithms were developed and compared.

## 1. Logistic Regression

Logistic Regression provides a strong interpretable baseline for binary classification.

StandardScaler was used because Logistic Regression is sensitive to feature scale.

Hyperparameter tuning was performed using `GridSearchCV` with F1-score as the optimization metric.

---

## 2. Random Forest

Random Forest was used because it can:

- Capture non-linear relationships
- Model feature interactions
- Handle complex decision boundaries
- Provide feature importance
- Work effectively with mixed feature behavior

Hyperparameters such as:

- `n_estimators`
- `max_depth`
- `min_samples_leaf`

were tuned using `GridSearchCV`.

---

## 3. XGBoost

XGBoost was used as the final gradient-boosting approach.

The model can capture non-linear relationships and interactions between machine operating parameters.

The following hyperparameters were tuned:

- `max_depth`
- `n_estimators`
- `learning_rate`

`GridSearchCV` with F1-score was used for model selection.

---

# 📏 Evaluation Metrics

Because failures are rare, **accuracy is not treated as the primary metric**.

The project focuses on:

### Recall

Measures how many actual machine failures were successfully detected.

High recall is important because missed failures can result in unexpected downtime.

### Precision

Measures how many predicted failures were actually failures.

This helps control unnecessary maintenance alerts.

### F1-Score

The harmonic mean of Precision and Recall.

F1-score is used as the primary model-selection metric because it balances missed failures and false alarms.

### ROC-AUC

Used as a supporting metric to evaluate the model's ability to distinguish between failure and non-failure observations across classification thresholds.

---

# 🎯 Final Model

The notebook selects **tuned XGBoost** as the final prediction model based on the expected F1/ROC-AUC balance and its ability to model non-linear relationships and engineered interactions.

> **Important:** The uploaded notebook contains the model-selection logic, but the saved notebook source does not contain the final printed evaluation values. Therefore, this README intentionally does not invent accuracy, precision, recall, F1, or ROC-AUC numbers. Add the actual values after executing the notebook.

Recommended results table:

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | Add result | Add result | Add result | Add result |
| Random Forest | Add result | Add result | Add result | Add result |
| XGBoost | Add result | Add result | Add result | Add result |
| Tuned XGBoost | Add result | Add result | Add result | Add result |

---

# 🔬 Model Explainability

The project includes model explainability using:

- XGBoost built-in feature importance
- SHAP-based explainability support

The analysis expects important predictive drivers to include:

- Torque
- Tool Wear
- HDF
- Temperature Difference

These findings are consistent with the EDA results, providing a connection between the exploratory analysis and the machine learning model.

---

# 💾 Model Saving & Prediction

The final model is saved using Joblib:

```python
joblib.dump(best_model, 'best_model.pkl')
```

The saved model is then loaded again to verify that it can make predictions on unseen test data.

Predictions are exported as:

```text
submission.csv
```

with the structure:

```text
id,Machine failure
```

---

# 🔄 End-to-End Workflow

```text
                    Raw Machine Data
                           │
                           ▼
                 Data Quality Analysis
                           │
                           ▼
                 Exploratory Data Analysis
                           │
                           ▼
                  Feature Understanding
                           │
                           ▼
                 Data Cleaning/Wrangling
                           │
                           ▼
              Feature Engineering & Selection
                           │
                           ▼
                Train/Validation Split
                           │
                           ▼
                      SMOTE
                           │
                           ▼
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         Logistic       Random       XGBoost
        Regression      Forest
              │            │            │
              └────────────┼────────────┘
                           ▼
                    Model Evaluation
                           │
                           ▼
                 GridSearchCV Tuning
                           │
                           ▼
                 Final XGBoost Model
                           │
                           ▼
                Feature Explainability
                           │
                           ▼
                 Save Model with Joblib
                           │
                           ▼
              Predict on Unseen Test Data
                           │
                           ▼
                    submission.csv
```

---

# 💼 Business Insights

The analysis suggests several areas that can support a predictive-maintenance strategy:

1. **Monitor Torque and Tool Wear closely** because they showed strong separation between failed and healthy machines.
2. **Monitor heat-dissipation conditions** because HDF was the most frequent individual failure mode.
3. **Pay particular attention to Low-tier machines**, which showed a generally higher failure rate.
4. Use **recall, precision and F1-score** rather than accuracy alone when evaluating a failure-detection system.
5. Use model explainability to help maintenance teams understand which operating conditions contribute to predicted failures.
6. Retrain the model periodically as additional machine operating data becomes available.

---

# 🗂️ Recommended Repository Structure

```text
Tata-Steel-Machine-Failure-Prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   ├── TATA_Steel_EDA_Capstone.ipynb
│   └── TATA_Steel_ML_Capstone.ipynb
│
├── models/
│   └── best_model.pkl
│
├── outputs/
│   └── submission.csv
│
├── images/
│   ├── eda_visualizations.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

> Do not upload large or confidential datasets to a public repository. If the dataset is not intended for public distribution, keep it excluded through `.gitignore` and document how it should be obtained.

---

# 🚀 How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/tata-steel-machine-failure-prediction.git
```

## 2. Open the Project

```bash
cd tata-steel-machine-failure-prediction
```

## 3. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Run the notebooks in this order:

```text
1. TATA_Steel_EDA_Capstone.ipynb
2. TATA_Steel_ML_Capstone.ipynb
```

Make sure `train.csv` and `test.csv` are available at the paths expected by the notebooks.

---

# 📦 Main Python Libraries

The notebooks use the following major libraries:

```text
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
imbalanced-learn
xgboost
statsmodels
joblib
```

Optional explainability:

```text
shap
```

A `requirements.txt` file should contain the exact versions used in the final environment if reproducibility is required.

---

# 🔮 Future Improvements

Potential improvements include:

- Threshold optimization based on maintenance cost.
- More extensive cross-validation.
- Cost-sensitive learning.
- Advanced hyperparameter optimization.
- Model calibration.
- SHAP-based local and global explanations.
- API deployment using Flask or FastAPI.
- Real-time monitoring dashboard.
- Integration with an industrial alerting system.
- Continuous model retraining with new sensor data.
- Monitoring model drift after deployment.

---

# ⚠️ Disclaimer

This project is an **educational and portfolio project** using a synthetic manufacturing sensor dataset. It demonstrates an approach to predictive maintenance and machine failure prediction and should not be interpreted as an official TATA Steel production system.

---

# 👨‍💻 Author

**Tushar**

---

## ⭐ Project Highlights

- End-to-end EDA + Machine Learning workflow
- 136K+ training records
- Severe class-imbalance handling with SMOTE
- Feature engineering with `Temp_diff` and `Power`
- Multicollinearity analysis using VIF
- Logistic Regression, Random Forest and XGBoost
- GridSearchCV hyperparameter tuning
- F1/Recall-focused evaluation
- XGBoost model explainability
- Joblib model persistence
- Unseen-data prediction and submission generation
