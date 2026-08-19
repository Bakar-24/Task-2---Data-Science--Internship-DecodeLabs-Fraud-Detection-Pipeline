# Task-2---Data-Science--Internship-DecodeLabs-Fraud-Detection-Pipeline
Data Science Project 2 for DecodeLabs Internship. A Supervised Learning pipeline using SMOTE and robust evaluation metrics (Precision, Recall, ROC-AUC) to detect credit card fraud in highly imbalanced datasets.
# 🛡️ Enterprise-Grade Fraud Detection Pipeline

**Author:** M.Abubakar UL Hassan  
**Education:** BS Computer Science, University of Gujrat  
**Role:** Data Science Intern at DecodeLabs 

## 📌 Project Overview
This repository contains a robust Supervised Learning pipeline designed to detect fraudulent credit card transactions[cite: 7]. The primary focus of this project is **Algorithmic Precision**—moving beyond misleading metrics to build a mathematically secure, zero-leakage machine learning architecture[cite: 6]. 

## 🚨 The Challenge: The Accuracy Trap
Financial datasets are notoriously skewed. In this specific dataset:
* **99.83%** of transactions are Legitimate.
* **0.17%** of transactions are Fraudulent.

A lazy model predicting "Legitimate" for every transaction would achieve 99.83% accuracy while catching zero fraud[cite: 5]. Therefore, the **"Accuracy" metric was completely discarded** in favor of strict, business-critical evaluation metrics[cite: 6, 7].

## 🛠️ Architectural Solutions & Key Milestones
To combat class imbalance and ensure absolute mathematical fidelity, the following protocols were implemented:

1. **Synthetic Minority Over-sampling (SMOTE):** Instead of merely duplicating data (which leads to severe overfitting), SMOTE was used to interpolate and create synthetic data points, helping the classifier learn a robust decision boundary[cite: 5, 6].
2. **The Zero-Leakage Protocol (`imblearn`):** Applying SMOTE before a Train/Test split causes a "Data Leakage Catastrophe"[cite: 5, 6]. To prevent this, `imblearn.pipeline.Pipeline` was utilized, ensuring that resampling and scaling occurred *strictly* within the Training Folds during Cross-Validation[cite: 6, 7].
3. **Robust Modeling via GridSearchCV:**
   * **Logistic Regression:** Tuned with `StandardScaler` as regularization penalties are highly sensitive to massive transaction amount variances[cite: 5, 6].
   * **Random Forest:** Implemented without a scaler, capitalizing on the scale-invariant nature of ensemble trees[cite: 5, 6].

## 📊 Strict Evaluation Metrics
The models were evaluated on the untouched test partition using the True Compass of imbalanced classification[cite: 5, 6]:
* **Precision:** Minimizing false declines to protect customer experience[cite: 5, 6].
* **Recall:** Ensuring the maximum amount of fraud is caught to prevent financial loss[cite: 5, 6].
* **ROC-AUC:** Measuring the model's overall capability to separate the legitimate distribution from the fraudulent distribution[cite: 5, 6].

## 💻 Technologies Used
* **Python** (Pandas, NumPy)
* **Scikit-Learn** (Logistic Regression, Random Forest, GridSearchCV, StratifiedKFold)
* **Imbalanced-Learn (`imblearn`)** (SMOTE, Pipeline)
* **Data Visualization** (Matplotlib, Seaborn)

## 📂 Dataset Reference
Due to GitHub's file size limitations, the `creditcard.csv` dataset is not hosted in this repository. 
You can download the dataset directly from Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
