# Ethereum Fraud Detection

**Author:** Dominic Rueck · **Course project (Spring 2026)**  
**Notebook:** [`Ethereum_Fraud_Detection.ipynb`](Ethereum Fraud Detection.ipynb) (written for **Google Colab** and local Jupyter).  
**Data:** [`transaction_dataset.csv`](transaction_dataset.csv) (Kaggle: [Ethereum Transaction Dataset for Fraud Detection](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)).

---

## Project workflow

The notebook follows an end-to-end analytics workflow:

1. **Problem framing and data acquisition**
2. **Exploratory data analysis (EDA) and data preparation**
3. **Model development, evaluation, and interpretation**

---

## Problem and target

The goal is a **binary classifier** at the **wallet (address) level**: predict whether an Ethereum address is associated with fraudulent behavior using aggregated transaction statistics. The supervised label is **`FLAG`** (0 = legitimate, 1 = fraud).

---

## Dataset summary

- **Size:** 9,841 rows × 51 columns in the raw CSV (per `df.info()` in the notebook).
- **Class imbalance:** About **22%** fraud vs **78%** legitimate (roughly **3.5:1** majority to minority). The notebook reports fractions after cleaning; metrics emphasize **precision, recall, F1**, **Cohen’s kappa**, and **ROC-AUC**, not accuracy alone.
- **Missing values:** Common in **ERC-20–related** columns when a wallet has little or no token activity. The modeling pipeline uses **imputation** rather than dropping those rows.

---

## EDA and visualizations (in the notebook)

The notebook generates plots with **matplotlib** and **seaborn**, including:

- Missingness summary (bar-style figure when implemented in the notebook).
- **Class distribution** for `FLAG`.
- **Correlation heatmap** for numeric features.
- Box and violin plots for top features **by class** (before and after preprocessing steps as coded).

| Topic | Preview |
|--------|--------|
| Class balance | ![FLAG class distribution placeholder](images/baseline_rf_conf_matrix.png) |
| Feature relationships | ![Correlation heatmap placeholder](images/feature_correlation.png) |

---

## Data preparation (current notebook logic)

- **Identifier columns removed:** `Unnamed: 0`, `Index`, and **`Address`** so the model does not memorize row keys or raw string IDs.
- **Numeric outliers:** Values (excluding the target) are **soft-capped** with **1.5× IQR**; when IQR is ~0, the notebook falls back to clipping at the **1st–99th percentiles** and prints how many cells were clipped.
- **Zero-variance numerics:** Columns with **≤1 unique value** are dropped after outlier treatment.
- **High-cardinality categoricals:** **`ERC20 most sent token type`** and **`ERC20_most_rec_token_type`** are dropped before modeling (very high cardinality; noted in the notebook).
- **Train / test split:** **Stratified 80/20** (`test_size=0.20`, `random_state=123`) so train and test keep a similar fraud rate.

---

## Modeling approach

- **Libraries:** **pandas**, **NumPy**, **scikit-learn** (`Pipeline`, `ColumnTransformer`, imputers, `OneHotEncoder`, **RandomForestClassifier**, **`GridSearchCV`**), **matplotlib**, **seaborn**. Additional classifiers are imported in the first code cell for extension but the delivered analysis centers on **Random Forest** (course requirement).
- **Preprocessing pipeline:**
  - Numeric features: **median** imputation (no scaling required for tree splits in this setup).
  - Categorical features: **most frequent** imputation + **one-hot** encoding (`handle_unknown='ignore'`).
- **Class imbalance:** After the split, the **minority (fraud) class is oversampled in the training set only** using `sklearn.utils.resample` so the **test set stays at the real class distribution**. Models are fit on `X_train_balanced` / `y_train_balanced`; evaluation uses `X_test` / `y_test`.
- **Baseline:** Random Forest with a **shallower** `max_depth` (as in the notebook) for a simple reference.
- **Tuning:** `GridSearchCV` with **5-fold CV**, primary scoring **`f1`** (binary **positive class = fraud**), and a grid over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and **`class_weight`** (`None` vs `'balanced'`).
- **Evaluation:** Classification report, **confusion matrices** (baseline and tuned), **ROC curve** with **AUC**, **Cohen’s kappa**, and **Random Forest feature importances** (`feature_importances_` on the preprocessor’s combined feature names, plotted as a horizontal bar chart for the top predictors).

### Model outputs

![Baseline RF confusion matrix placeholder](images/baseline_rf_conf_matrix.png)

![Tuned RF confusion matrix placeholder](images/tuned_rf_conf_matrix.png)

![ROC curve placeholder](images/roc_curve.png)

![Feature importance placeholder](images/feature_importance.png)

---

## How to run

1. Place **`transaction_dataset.csv`** next to the notebook (or adjust `data_path` in the loading cell).
2. Open the notebook in **Jupyter**, **VS Code**, or **Google Colab** (upload the CSV if using Colab).
3. Run **all cells** from top to bottom so derived `df`, `X_train_balanced`, and models stay consistent.

---

## Repository layout (main artifacts)

| File | Role |
|------|------|
| `Ethereum Fraud Detecion.ipynb` | Full analysis and figures |
| `transaction_dataset.csv` | Kaggle dataset (not always committed; obtain from Kaggle if missing) |
| `images/` | exported figures from the notebook|
