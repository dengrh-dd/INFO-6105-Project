# Heart Disease Prediction with Interpretable Machine Learning

This project develops and evaluates multiple machine learning models to predict the presence of heart disease, with a strong emphasis on **robust preprocessing**, **fair model comparison**, and **model interpretability using SHAP**.

## Project Overview

- **Task**: Binary classification (Heart Disease: Yes / No)
- **Dataset**: Public heart disease dataset ([Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction))
- **Goal**:
1. Build accurate predictive models
2. Compare different missing-value handling strategies
3. Provide transparent, clinically interpretable explanations using SHAP

## Models Implemented

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

- Model selection and hyperparameter tuning are conducted using **cross-validation** with ROC-AUC as the primary metric.

## Exploratory Data Analysis 

### Missing Value Check
- No Missing found

### Distribution Plot
- Found cholestrol level abnormal(value = 0), treated as missing value

### Correlation Check
- Generate Heatmap to check Multicolinearity, no high-correlated features found.

## Feature Engineering

### Missing Cholesterol Handling (Three Strategies)
1. **Median Imputation** (baseline)
2. **Regression Imputation**  
   - Random Forest Regressor trained
   - Evaluated with 5-fold cross-validation (RMSE & R²)
3. **Dropping Missing Samples** (for robustness comparison)

### Feature Engineering
- Categorical features encoded via One-Hot Encoding
- Numerical features preserved or standardized depending on model type
- Additional binned features (Age, MaxHR, Oldpeak) for linear models
- Explicit prevention of data leakage between train/test splits

## Model Evaluation

Evaluation is performed on a held-out test set:

- ROC-AUC
- Accuracy
- F1 Score
- Confusion Matrix
- ROC Curve comparison (RF vs XGBoost)
- Train-Test performance comparison (Check Overfitting)

## Model Interpretability 

To ensure interpretability and practical insight:

- Coefficents for Linear Models
- Feature Importance for Tree-Based Models
- **SHAP summary plots (beeswarm & bar)** for Tree-Based Models
- **Aggregated SHAP analysis**:
  - One-hot encoded categorical variables are grouped back to their original features
  - Enables intuitive interpretation of variables such as `ChestPainType`, `ST_Slope`, etc.

