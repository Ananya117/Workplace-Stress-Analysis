# 💼 Workplace Stress Analysis using Ensemble Machine Learning

**📍 Research Internship – CEAT, VIT Chennai**

## 📌 Overview

This project aims to predict psychological stress levels in employees using ensemble-based machine learning techniques on a custom stress dataset. It incorporates feature selection, explainability, and model optimization for reliable workplace stress assessment.

## 🔍 Feature Selection

Applied a combination of:

* Recursive Feature Elimination (RFE)
* Correlation matrix
* Lasso Regression
* Mutual Information Gain
* XGBoost Feature Importance

These helped narrow down the most relevant predictors from 25+ features to the top 18.

## 🧠 Model Architecture

A stacked ensemble model was trained using:

* **Base Learners**: CatBoost, XGBoost, Random Forest
* **Meta Learner**: Logistic Regression
* **Evaluation Metrics**: Accuracy, F1 Score, AUC, Confusion Matrix

## 📈 Explainability & Optimization

* Used **SHAP (SHapley Additive exPlanations)** for model interpretability.
* Applied **model quantization** and **pruning** techniques to enhance deployment scalability.
* **Optuna** used for hyperparameter optimization.

## ⚙️ Tech Stack

* Python (Pandas, Scikit-learn, CatBoost, XGBoost, SHAP, Optuna)
* Google Colab
* GitHub for version control

## 📁 Dataset

Base dataset: https://data.mendeley.com/datasets/6nzhv6vtxr/1
Includes 497 samples with categorical-numerical features (scaled 1–5).
Target: Stress level classification (multi-class).

Cross-validation datasets:
1. https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
2. https://www.kaggle.com/datasets/adilshamim8/global-remote-work-and-wellbeing-dataset
3. https://www.kaggle.com/datasets/atharvasoundankar/mental-health-and-lifestyle-habits-2019-2024/data

