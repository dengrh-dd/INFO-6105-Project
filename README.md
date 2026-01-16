# Heart Disease Prediction with Interpretable Machine Learning

This project develops and evaluates multiple machine learning models to predict the presence of heart disease, with a strong emphasis on **robust preprocessing**, **fair model comparison**, and **model interpretability using SHAP**.

---

## Project Overview

- **Task**: Binary classification (Heart Disease: Yes / No)
- **Dataset**: Public heart disease dataset  
  ([Kaggle – Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction))

**Goals**:
- Build accurate predictive models
- Compare different missing-value handling strategies
- Provide transparent, clinically interpretable explanations using SHAP

---

## Models Implemented

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Model selection and hyperparameter tuning are conducted using **cross-validation**, with **ROC-AUC** as the primary evaluation metric.

---

## Exploratory Data Analysis

### Missing Value Check
- No missing values were found in the original dataset.

### Distribution Analysis
- Abnormally recorded cholesterol values (value = 0) were identified and treated as missing.

### Correlation Analysis
- A correlation heatmap was generated to assess multicollinearity.
- No highly correlated numerical features were identified.

---

## Feature Engineering

### Missing Cholesterol Handling (Three Strategies)
1. **Median Imputation** (baseline)
2. **Regression Imputation**
   - Random Forest Regressor trained on observed samples
   - Evaluated using 5-fold cross-validation (RMSE & R²)
3. **Dropping Missing Samples** (for robustness comparison)

### Feature Construction
- Categorical features encoded using One-Hot Encoding
- Numerical features preserved or standardized depending on model type
- Additional binned features (Age, MaxHR, Oldpeak) created for linear models
- Explicit prevention of data leakage between training and test sets

---

## Model Evaluation

Evaluation is performed on a held-out test set using:

- ROC-AUC
- Accuracy
- F1 Score
- Confusion Matrix
- ROC curve comparison (Random Forest vs XGBoost)
- Train–test performance comparison to assess overfitting

---

## Model Interpretability

To ensure interpretability and practical insight:

- **Coefficients** for linear models
- **Feature importance** for tree-based models
- **SHAP summary plots** (beeswarm & bar) for tree-based models
- **Aggregated SHAP analysis**:
  - One-hot encoded categorical variables are grouped back to their original features
  - Enables intuitive interpretation of variables such as `ChestPainType` and `ST_Slope`
- Feature interactions explored using SHAP dependence plots


