# Extended Analysis Summary  
Heart Disease Prediction with Interpretable Machine Learning

---

## 1. Problem Definition

This project addresses a **binary classification problem**: predicting whether a patient has heart disease based on demographic, clinical, and test-related features.

Given the medical context, model evaluation emphasizes **discriminative ability** (ROC-AUC) and **balanced performance**, rather than accuracy alone.

---

## 2. Data Overview & Key Challenges

The dataset is a publicly available heart disease dataset containing both numerical and categorical clinical variables.

A key data challenge identified during exploratory analysis was the presence of **abnormally recorded cholesterol values (value = 0)**, which are not physiologically meaningful and were therefore treated as missing values.

This issue motivated a systematic comparison of different missing-value handling strategies and their downstream impact on model performance.

---

## 3. Methodology

### Missing-Value Handling Strategies

Three strategies were evaluated for handling missing cholesterol values:

1. **Median Imputation**  
   - Serves as a simple, leakage-safe baseline.

2. **Regression-Based Imputation**  
   - A Random Forest Regressor was trained **only on observed training samples**.
   - Performance was evaluated using 5-fold cross-validation (RMSE & R²).
   - The fitted regressor was then used to impute missing values in both training and test sets, avoiding data leakage.

3. **Dropping Missing Samples**  
   - Used as a robustness check to assess the sensitivity of model performance to sample removal.

---

### Models Evaluated

- Logistic Regression (with L1 / L2 regularization)
- Random Forest Classifier
- XGBoost Classifier

Hyperparameters were tuned using **cross-validation**, with **ROC-AUC** as the primary optimization metric.

---

## 4. Model Evaluation & Results

Models were evaluated on a held-out test set using:

- ROC-AUC
- Accuracy
- F1 Score
- Confusion Matrix
- ROC curve comparisons across models

Key observations include:

- **Tree-based models (Random Forest and XGBoost) consistently outperformed Logistic Regression** in terms of ROC-AUC.
- Missing-value handling strategies materially affected downstream model performance.
- Regression-based imputation provided a more stable alternative compared to simple median imputation in several settings.
- Train–test performance gaps were monitored to assess potential overfitting.

---

## 5. Model Interpretability

Interpretability was a core focus of this project.

- For **linear models**, coefficient magnitudes were examined to assess feature influence.
- For **tree-based models**, feature importance scores were analyzed.
- **SHAP (SHapley Additive exPlanations)** was used to provide local and global explanations:
  - SHAP summary plots (beeswarm and bar) were generated based on predicted probabilities.
  - Feature interactions were explored using SHAP dependence plots.

### Aggregated SHAP Analysis

To improve interpretability for categorical variables encoded via One-Hot Encoding:

- SHAP values for dummy variables were **aggregated back to their original categorical features**.
- This aggregation enables more intuitive interpretation of clinically meaningful variables such as `ChestPainType` and `ST_Slope`.

---

## 6. Limitations

- Analysis is based on a single public dataset.
- No external validation dataset was available.
- The models are not calibrated for clinical decision thresholds.

---

## 7. Future Work

Potential extensions include:

- Probability calibration (e.g., Platt scaling or isotonic regression)
- Validation on independent datasets
- Cost-sensitive learning with greater emphasis on recall for high-risk patients
- Incorporation of domain-specific clinical constraints

---

## 8. Conclusion

This project demonstrates how **robust preprocessing**, **careful experimental design**, and **interpretable machine learning techniques** can be combined to build transparent and reliable predictive models in a healthcare context.

The results highlight the importance of data quality decisions and interpretability when applying machine learning to real-world clinical problems.
