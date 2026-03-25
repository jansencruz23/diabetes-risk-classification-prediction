# Diabetes Risk Classification & Prediction

A machine learning project that classifies individuals into diabetes risk categories using clinical and lifestyle risk factors. Three classifiers — Logistic Regression, Decision Tree, and Random Forest — were trained, tuned via Grid Search, and evaluated on a real-world health dataset.

---

## 📁 Project Structure

```
HCU/
├── Diabetes Risk Classification.ipynb   # Main notebook
├── datasets/
│   ├── diabetesrisk (unclean dataset).csv  # Raw data
│   └── diabetes_risk.csv                   # Cleaned & processed data
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | Health & Community Unit (HCU) internal records |
| **Records** | 460 samples |
| **Target Variable** | `RiskCategory` |
| **Train / Test Split** | 80% / 20% (stratified) |

### Features

| Feature | Description |
|---|---|
| `BmiRiskPoints` | BMI-derived risk score |
| `IsPhysicallyActive` | Physical activity status (mapped to risk points) |
| `EatsVegetablesEveryDay` | Vegetable consumption habit |
| `TakingHighBloodPressureMedication` | BP medication usage (mapped to risk points) |
| `HasHighBloodGlucose` | Blood glucose status (mapped to risk points) |
| `FamilyWithDiabetesRiskPoints` | Family history of diabetes risk score |
| `WaistCircumferenceMenRiskPoints` | Waist circumference risk (male) |
| `WaistCircumferenceWomenRiskPoints` | Waist circumference risk (female) |
| `AgeRiskPoints` | Age-derived risk score |

### Target Classes (RiskCategory)

| Category | Risk Score Range |
|---|---|
| Low | < 7 |
| Slightly Elevated | 7 – 11 |
| Moderate | 12 – 14 |
| High | 15 – 20 |
| Very High | > 20 |

---

## ⚙️ Methodology

1. **Data Cleaning & Preprocessing**
   - Dropped non-predictive columns (IDs, timestamps, metadata)
   - Computed `Age` from `Birthday` using current date
   - Mapped binary features to scaled risk point values
   - Filled missing waist circumference values with `0` (zero is meaningful)
   - Engineered `RiskCategory` from cumulative risk score

2. **Model Training**
   - Logistic Regression (multinomial, L2 regularization)
   - Decision Tree (Gini criterion, `max_depth=7`)
   - Random Forest (Gini criterion, `max_depth=30`, bootstrap)
   - Support Vector Machine (SVM)

3. **Hyperparameter Tuning**
   - Grid Search CV (5-fold) on all three models
   - Scoring metric: `accuracy`

4. **Feature Scaling**
   - `StandardScaler` applied to all features before training

---

## 📈 Model Performance

### Logistic Regression

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Untuned | 88% | 0.82 | 0.88 |
| **Tuned** | **95%** | **0.90** | **0.94** |

**Tuned Logistic Regression — Per-Class Metrics:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| High | 0.89 | 0.67 | 0.76 | 12 |
| Low | 1.00 | 1.00 | 1.00 | 31 |
| Moderate | 0.77 | 1.00 | 0.87 | 10 |
| Slightly Elevated | 1.00 | 1.00 | 1.00 | 30 |
| Very High | 0.89 | 0.89 | 0.89 | 9 |

---

### Decision Tree

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Untuned | 74% | 0.68 | 0.74 |
| **Tuned** | **78%** | **0.72** | **0.78** |

**Tuned Decision Tree — Per-Class Metrics:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| High | 0.62 | 0.42 | 0.50 | 12 |
| Low | 0.97 | 0.97 | 0.97 | 31 |
| Moderate | 0.50 | 0.60 | 0.55 | 10 |
| Slightly Elevated | 0.73 | 0.80 | 0.76 | 30 |
| Very High | 0.88 | 0.78 | 0.82 | 9 |

---

### Random Forest

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Untuned | 85% | 0.80 | 0.85 |
| **Tuned** | **85%** | **0.79** | **0.85** |

**Tuned Random Forest — Per-Class Metrics** (`criterion='entropy'`, `max_depth=10`, `n_estimators=200`)**:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| High | 0.70 | 0.58 | 0.64 | 12 |
| Low | 1.00 | 0.94 | 0.97 | 31 |
| Moderate | 0.46 | 0.60 | 0.52 | 10 |
| Slightly Elevated | 0.88 | 0.93 | 0.90 | 30 |
| Very High | 1.00 | 0.89 | 0.94 | 9 |

---

### Support Vector Machine (SVM)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Tuned** | **96%** | — | — |

---

## 🏆 Best Model

> **Support Vector Machine (SVM)** achieved the highest overall accuracy of **96%**, narrowly outperforming the Tuned Logistic Regression (95%) and making it the best-performing model for this classification task.

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- **Environment:** Jupyter Notebook (Anaconda)

---

## 🚀 How to Run

1. Clone the repository
2. Open `Diabetes Risk Classification.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells sequentially (cells are organized in sections: Data Cleaning → Model Training → Hyperparameter Tuning → Evaluation)

> **Note:** The hyperparameter tuning cells (Section 3.1 and 3.3) use Grid Search and may take several minutes to complete. The optimal parameters have already been applied in the subsequent cells.
