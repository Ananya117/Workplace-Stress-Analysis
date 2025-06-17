### 1. CatBoost (Baseline)

**→ Single Gradient Boosting Model on Base Dataset**

**What Was Done:**

* Used the `CatBoostClassifier` with default and light hyperparameter tuning.
* Handled categorical features natively with CatBoost (no manual encoding needed).
* Trained on the **base normalized dataset** for initial benchmark performance.

**Dataset Used:**

* Base dataset (adjusted\_stress\_dataset\_normalized.csv)

**Accuracy:**

* **\~81%**

**Problem Faced:**

* Accuracy plateaued around 80–81%.
* Needed better generalization on minority stress level classes.
* Insufficient performance for deployment or comparison across datasets.

---

### 2. Stacked Ensemble Model 1 (CatBoost + TabNet → Logistic Regression)

**→ Deep+Boost Hybrid Stacking Model**

**What Was Done:**

* Trained **CatBoost** and **TabNet** separately.
* Used a `LogisticRegression` as meta-learner in a `StackingClassifier`.
* Designed to leverage both tabular interpretability (CatBoost) and deep learning sequential reasoning (TabNet).

**Datasets Used:**

* Base Dataset → **90%**
* Dataset 1 → **98%**
* Dataset 2 → **34%**
* Dataset 3 → **\~10%**

**Problem Faced:**

* Massive accuracy drop across datasets 2 and 3.
* High dataset variance indicates sensitivity to input distribution shifts.
* Required deeper understanding of feature importance and generalization issues.

---

### 3. Feature Selection (Across 5 Techniques)

**→ Combined Filter, Wrapper, and Embedded Selection**

**What Was Done:**

* Applied **5 different selection methods** to identify the most relevant features:

  * Recursive Feature Elimination (RFE)
  * Correlation Matrix (removing highly collinear variables)
  * Lasso Regression (embedded L1 penalty)
  * Mutual Information Gain
  * XGBoost Feature Importance
* Extracted the **common top features** identified across all five methods for final modeling.

**Dataset Used:**

* Base dataset

**Outcome:**

* Reduced dimensionality
* Improved signal-to-noise ratio before final ensemble training

**Problem Faced:**

* Feature rankings varied across methods; union/intersection logic had to be carefully applied
* Risk of losing informative features during overlap trimming

---

### 4. Stacked Ensemble Model 2 (CatBoost + XGBoost + Random Forest → Logistic Regression)

**→ All-Tree Stacking Model for Final Performance**

**What Was Done:**

* Trained three tree-based models: **CatBoost**, **XGBoost**, and **Random Forest**.
* Combined their outputs in a `StackingClassifier` with **Logistic Regression** as meta-learner.
* Used only the final selected features (from step 3) as inputs.

**Dataset Used:**

* all

**Accuracy:**

* Base Dataset → **100%**
* Dataset 1 → **98%**
* Dataset 2 → **34%**
* Dataset 3 → **50%**

**Problem Faced:**

* Final model gave perfect results on test split for base dataset alone - suspected overfitting.
* overfitting confirmed with cross validation with other three datasets


