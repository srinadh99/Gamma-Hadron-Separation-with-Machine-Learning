# Gamma-Hadron-Separation-with-Machine-Learning
Gamma-Hadron Separation with Machine Learning

# MAGIC Gamma–Hadron Classification (Low-FPR Machine Learning Suite)

This repository provides a **machine learning framework** to classify **gamma-ray (signal)** vs. **hadronic (background)** events from the **MAGIC dataset**, emphasizing **low false-positive rate (FPR)** performance.
It supports multiple models, feature-processing variants, and automatic evaluation with confusion matrices and ROC-based metrics.

---

## 🔍 Overview

Gamma–hadron separation is a key task in ground-based Cherenkov telescope analysis.
Simple accuracy is insufficient — **classifying a background event as signal is far worse** than misclassifying signal as background.
Thus, models here are compared using **ROC-based metrics**, particularly **TPR at low FPRs** (e.g., 1–10%) and **partial AUC (pAUC)**.

---

## ⚙️ Pipeline Summary

The full pipeline consists of standardized preprocessing, PCA-based feature compression, upsampling for class balance, and low-FPR model evaluation.

### Step 1 — Baseline: All Features → StandardScaler

* All original features (`fLength` → `fDist`) are standardized using `StandardScaler`.
* Models are trained directly on these standardized features.
* Evaluation focuses on:

  * **Partial AUC (pAUC@≤0.10)** as the CV selection metric
  * **TPR at FPR = 0.01, 0.02, 0.05, 0.10, 0.20**
  * **Full AUC**, **Confusion Matrix**, and **ROC plots**

### Step 2 — PCA Features (Top MI Feature + 95% Variance PCs)

* Compute **Mutual Information (MI)** between each feature and the target.
* Retain the **top MI feature (`fAlpha`)** explicitly.
* Apply `StandardScaler` to the remaining features, then fit **PCA** to keep components explaining **≈95% of variance**.
* Concatenate `[fAlpha (scaled)] + [PCA components]` to form the final training matrix.
* Train the same set of models with identical evaluation metrics.

### Step 3 — Model Training and Evaluation

* Upsample the minority class in the training data using `sklearn.utils.resample`.
* Perform **5-fold Stratified Cross-Validation** with **RandomizedSearchCV**.
* Optimize models for **pAUC@≤0.10**.
* Compute test-set metrics:

  * TPR@FPR thresholds (0.01–0.20)
  * Partial AUCs and Full AUC
  * Confusion Matrix and ROC plots (saved and/or displayed)
* Generate a **summary table** ranking all models by CV and test performance.

---

#### The confusion matrix.

![plot](./plot_rawfeatures/MLP_cm.png)

![plot](./plot_rawfeatures/ROC_allmodels.png)



