# 💳 Credit Card Fraud Detection — Deep Learning & Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF) ![License](https://img.shields.io/badge/License-MIT-green)

> Detecting financial fraud using unsupervised and supervised deep learning on the most referenced fraud dataset on Kaggle (13,000+ votes).

## Problem Statement

Credit card fraud costs the global economy over **$32 billion/year**. With only **0.17% fraud rate**, traditional ML struggles with extreme class imbalance. This project tackles both the technical challenge (imbalance, threshold selection) and the business challenge (minimizing false negatives while controlling false positive rate).

## Approach

| Method | Type | Key Metric |
|--------|------|------------|
| Autoencoder | Unsupervised anomaly detection | Reconstruction error threshold |
| LightGBM + SMOTE | Supervised + resampling | PR-AUC |
| SHAP | Explainability | Feature attribution |

## Pipeline

```
Raw transactions → StandardScaler → Train/Test split (stratified)
    ├── Autoencoder (train on normal only) → reconstruction error → threshold → predictions
    └── LightGBM + SMOTE → calibrated probabilities → optimal threshold → predictions
                                ↓
                    SHAP explanations + business scorecard
```

## Key Results *(to fill after training)*

| Model | PR-AUC | F1 | Recall@1%FPR |
|-------|--------|----|--------------|
| Autoencoder | - | - | - |
| LightGBM + SMOTE | - | - | - |
| Ensemble | - | - | - |

## Kaggle Notebook

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ibrahimagabardiop/fraud-detection-deep-learning)

## Datasets

- [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284K transactions, 492 frauds (13K+ votes)
- [`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) — IBM synthetic AML data

## Tech Stack

```
Python · PyTorch · LightGBM · SHAP · imbalanced-learn · scikit-learn · Matplotlib
```

## Run Locally

```bash
pip install -r requirements.txt
jupyter notebook fraud-detection-deep-learning.ipynb
```

## Author

**Ibrahima Gabar Diop** — [Kaggle](https://www.kaggle.com/ibrahimagabardiop) · [GitHub](https://github.com/Gblack98) · [LinkedIn](https://www.linkedin.com/in/ibrahimagabardiop)
