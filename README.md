# Ethereum-Fraud-Detection

**Author:** Dominic Rueck · **Course project (Spring 2026)** · Analysis in [`DataMining_Project_Template_Spring_2026 (1).ipynb`](DataMining_Project_Template_Spring_2026%20(1).ipynb) (designed for **Google Colab**).

## Project workflow

The notebook follows an end-to-end analytics workflow:

1. **Problem framing and data acquisition**
2. **Exploratory data analysis (EDA) and data preparation**
3. **Model development, evaluation, and interpretation**

## Project overview

### 1. Introduction and problem statement

As blockchain technology gains mainstream adoption, the Ethereum network has become a primary hub for global decentralized finance. However, as with all financial systems, people try to exploit vulnerabilities for fraudulent gain. Detecting and preventing fraud on the Ethereum blockchain is crucial for maintaining trust and security in the ecosystem.

The core objective of this project is to build a robust classification model capable of differentiating legitimate from fraudulent activity using wallet-level behavior. The analysis uses historical transaction aggregates per Ethereum address and predicts the binary label **`FLAG`** (fraud vs. legitimate) in the dataset.

### 2. Business use case and impact

This tool could serve as a critical tool for:

- **Financial institutions and exchanges**: To automatically flag suspicious patterns tied to wallets and to protect users from potential losses.
- **Regulatory bodies**: To monitor and investigate fraudulent activity on the blockchain, supporting compliance with financial regulations.
- **Individual users**: To safeguard assets by surfacing higher-risk addresses before irreversible actions.

### 3. Data acquisition

The data is sourced from Kaggle: [Ethereum Transaction Dataset for Fraud Detection](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset). In the notebook, the working file is `transaction_dataset.csv`. The table has **9,841 rows and 51 columns**: each row is an Ethereum **address**, with numeric features describing timing, counts, Ether flows, and ERC-20 token activity, plus **`FLAG`** as the supervised label.

### Motivation

I selected this dataset and problem because of the growing importance of blockchain technology and the critical need for security in decentralized finance. Fraud detection on Ethereum is a real-world problem with significant implications for users and institutions alike. By applying data mining techniques to this dataset, I hope to contribute to more secure blockchain systems and to gain practical experience with complex, real-world tabular data.

## Dataset highlights (from EDA)

- **Class imbalance**: Roughly **22%** of rows are labeled fraud (about **3.5:1** majority-to-minority). Accuracy alone can look optimistic while missing fraud, so modeling emphasizes **precision, recall, F1, and Cohen’s kappa** (and class weights or resampling can be added if needed).
- **Missing values**: Much of the missingness appears in **ERC-20–related** columns when a wallet has little or no token activity. Rows are retained; gaps are handled with **imputation** in the modeling pipeline.

## Data preparation (summary)

- **Identifiers removed**: `Unnamed: 0`, `Index`, and `Address` are dropped so the model does not rely on row keys or high-cardinality raw strings.
- **Imputation**: Numeric features use **median** imputation; categorical features use **mode** (or a `(missing)` placeholder if needed). The sklearn `Pipeline` keeps the same steps if values shift.
- **Outliers**: Numeric features (excluding the target) are **soft-capped** with the **1.5×IQR** rule; when IQR is near zero, values are clipped to the **1st–99th percentiles** to limit the influence of extreme sends, receives, or timing without deleting rare fraud rows.
- **Train / test split**: **Stratified 80/20** split (`random_state=123`) so train and test keep a similar fraud rate.

## Methods and tools

- **Languages and libraries**: Python with **pandas** and **NumPy**, **scikit-learn** (pipelines, `GridSearchCV`, imputers, encoders), **matplotlib** and **seaborn** for plots; **AutoViz** is available in the notebook for automated EDA-style charts.
- **Primary model**: **Random Forest** classifier inside a **sklearn `Pipeline`** (course requirement), with **hyperparameter tuning** via `GridSearchCV` (e.g., weighted F1 as the scoring metric in the template).
- **Evaluation**: Classification report, **confusion matrix**, **Cohen’s kappa**, and **feature importance** (interpreted as model reliance, not proof of causation).

## Data loading

For reproducibility, load the CSV from the repo, upload it in Colab, or read from a URL; keeping the raw file in version control when possible is recommended.
