### 1. **CatBoost (Baseline Model)**

> 🎯 Initial benchmark using gradient boosting.

* Used `CatBoostClassifier` on the base dataset (`adjusted_stress_dataset_normalized.csv`) with native handling of categorical variables.
* Applied minimal tuning to get a performance baseline.

**Accuracy**: \~81%
**Issue**: Plateaued performance, struggled with minority stress classes.

---

### 2. **Stacked Ensemble: CatBoost + TabNet → Logistic Regression**

> ⚡ Hybrid deep + tree ensemble for improved generalization.

* Trained CatBoost and TabNet separately.
* Combined predictions using `LogisticRegression` as a meta-learner via `StackingClassifier`.
* Aimed to merge CatBoost’s interpretability with TabNet’s deep-sequential modeling.

**Accuracies**:

* Base Dataset → 90%
* Dataset 1 → 98%
* Dataset 2 → 34%
* Dataset 3 → \~10%

**Issue**: Extreme performance drop on Datasets 2 and 3. Sensitive to input distribution shifts — revealed generalization and robustness concerns.

---

### 🧪 Feature Selection Step (Intermediate)

> Applied multiple methods to reduce noise and focus on informative features:

* **Techniques used**:

  * Recursive Feature Elimination (RFE)
  * Correlation Matrix filtering
  * Lasso (L1) Regularization
  * Mutual Information
  * XGBoost Feature Importance

* Took the **common top features** across all methods.

* Helped improve signal quality for final modeling.

**Challenge**: Feature ranks varied by method — needed careful overlap logic to avoid losing useful features.

---

### 3. **Stacked Ensemble: CatBoost + XGBoost + Random Forest → Logistic Regression**

> 🌳 Tree-only ensemble using selected features.

* Trained CatBoost, XGBoost, and Random Forest individually.
* Combined predictions using `LogisticRegression` as a meta-model.
* Used only the features selected from the previous step.

**Accuracies**:

* Base Dataset → 100%
* Dataset 1 → 98%
* Dataset 2 → 34%
* Dataset 3 → 50%

**Issue**: Overfitting on Base Dataset confirmed via cross-validation. Generalization still weak.

---

### 4. **CatBoost + TabNet (Reduced Dataset 2)**

> 🔍 Evaluation on minimal-feature version of Dataset 2.

* Reduced Dataset 2 to just **4 common features** (lowest correlation but retained across datasets).
* Used CatBoost and TabNet independently.
* Optuna used for tuning; SHAP used for CatBoost feature analysis.

**Accuracy**: \~32%
**Insight**: Better than full-feature version, but still low.

---

### 5. **Additional Attempt: Full Dataset 2 with All Features**

* Re-ran CatBoost and TabNet on full Dataset 2 using:

  * Optuna for hyperparameter tuning
  * SHAP for explainability

**Accuracies**:

* CatBoost → \~12%
* TabNet → \~11%
  **Observation**:
* Highly non-linear, weak correlation between input features and `Stress_Level`.
* Models failed to learn meaningful decision boundaries → deeper modeling required.

---

### 6. **Additional Attempt: Full Dataset 2 with All Features**

* Re-ran multiple models on full Dataset 2 using:

  * Advanced preprocessing techniques
  * 8 diverse models (tree-based, DL, ensemble)
  * Unified evaluation pipeline

---

**🛠️ Preprocessing Applied:**

* Dropped identifier column (`Employee_ID`)
* Imputed missing values using:

  * **KNNImputer** for numerical features
  * **SimpleImputer** (mode) for categorical
* Normalized numerical columns using **MinMaxScaler**
* One-hot encoded categorical columns (if any)
* Addressed class imbalance with **SMOTE** oversampling
* Converted sparse matrices to dense arrays where required

---

**🧠 Models Evaluated (Classification):**

| No. | Model Name                             | Accuracy |
| --- | -------------------------------------- | -------- |
| 1   | CatBoostClassifier                     | \~11.2%  |
| 2   | TabNetClassifier                       | \~10.8%  |
| 3   | StackingClassifier (CatBoost + TabNet) | \~10.5%  |
| 4   | XGBoostClassifier                      | \~11.0%  |
| 5   | LightGBMClassifier                     | \~11.4%  |
| 6   | ExtraTreesClassifier                   | \~10.9%  |
| 7   | MLPClassifier (Neural Net)             | \~10.3%  |
| 8   | VotingClassifier (XGB + LGBM + ET)     | \~11.1%  |

---

**📉 Observation:**

* All models performed near **random guess baseline** (10-class problem → \~10%).
* Strong signs of **non-linearity** and **weak statistical correlation** between input features and target (`Stress_Level`).
* Feature representations appear **insufficient for learning meaningful patterns**.
* Indicates the need for:

  * Additional or richer features (e.g., survey data, temporal logs, external context)
  * Possibly using **unsupervised pretraining**, **self-supervised learning**, or **deep feature engineering**
  * Alternative targets (e.g., binary or ordinal stress levels)
