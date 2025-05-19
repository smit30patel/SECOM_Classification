# 🏭 SECOM Sensor Failure Prediction (Imbalanced Classification Pipeline)

This project focuses on building a robust machine learning pipeline to predict product failures in a semiconductor manufacturing process using the SECOM dataset. The dataset includes over 590 sensor signals per unit and is heavily imbalanced with only ~6.6% failure cases.

The pipeline tackles key real-world challenges including high-dimensional noisy data and class imbalance. It incorporates feature selection using F-Test (SelectKBest), handles imbalance via both scale_pos_weight and SMOTE oversampling, and leverages XGBoost for classification. The model was evaluated using F1-score, classification reports, and confusion matrix visualizations to ensure interpretability and performance.

The final solution is production-ready and structured using sklearn and imblearn pipelines, making it ideal for demonstration in ML engineering interviews or real-time applications. Feature importance plots and evaluation metrics help communicate the model’s decision-making process clearly.

---

## 🚀 Project Highlights

- ✅ Feature Selection using F-Test (SelectKBest)
- ✅ Imbalanced classification handled via `scale_pos_weight` and SMOTE
- ✅ Model: XGBoost Classifier
- ✅ Evaluation with F1-score, classification report, and confusion matrix
- ✅ Visual insights into feature importance and model performance
- ✅ Built using modular `sklearn` + `imblearn` pipelines

---

## 📂 Dataset

- **Source**: [UCI SECOM Dataset](https://archive.ics.uci.edu/ml/datasets/SECOM)
- 1567 observations × 591 sensor features
- Binary target: `label = 1` (fail), `label = -1` (pass)

---

## 📊 Pipeline Overview

1. **Data Loading & Preprocessing**
   - Concatenated `secom.data` and `secom_labels.data`
   - Renamed feature columns (`f1` to `f590`)
   - Imputed missing values using column mean
   - Standardized features using `StandardScaler`

2. **Feature Selection**
   - Used `SelectKBest` with F-test to select top 40 features
   - Applied only to training data (no leakage)

3. **Handling Imbalance**
   - Tried both:
     - `scale_pos_weight` in XGBoost
     - `SMOTE` oversampling (via `imblearn.Pipeline`)

4. **Modeling**
   - Trained an `XGBClassifier`
   - Evaluated using:
     - F1-score
     - Confusion matrix
     - Classification report

5. **Visualization**
   - Confusion matrix with `seaborn`
   - Feature importances with horizontal bar chart

---

## ✅ Final Performance

> _(After fixing leakage and applying imbalance techniques)_

| Metric        | Value   |
|---------------|---------|
| F1 Score (Fail class) | ~0.60+ (with SMOTE / weight tuning) |
| Accuracy      | ~88%+   |
| Recall (Fail) | Significantly improved over baseline |

---

---

## 📈 Visual Output

Confusion Matrix:

![confusion-matrix](Unknown.png)

Feature Importances:

![feature-importances](Unknown-2.png)

---

## 📦 Requirements

```bash
pip install pandas scikit-learn seaborn matplotlib xgboost imbalanced-learn
```

## ✍️ Author

Smit Patel


