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
