# Fraud Detection with Logistic Regression | R

This project tackles the growing challenge of financial fraud in the era of digital banking. Using a synthetically generated dataset simulating transaction behavior from 500,000 customers, we built a fraud detection model to help a fictional "Bank A" identify high-risk transactions.

## 🧠 Business Context

With over $198 million lost to scams in New Zealand in 2023 alone (MBIE, 2024), effective fraud detection systems are essential. The project's goal: build a robust, interpretable model that flags fraudulent transactions before they wreak havoc on customers and banks.

## 🔍 Dataset Summary

- **Timeframe:** 2 Feb 2024 – 31 July 2024
- **Observations:** ~500,000 transactions
- **Features:** Age, Balance, Transaction Amount, Transaction Time, Joint Account Flag, Device Agent, Transaction Type, etc.
- **Target Variable:** `FraudLabel` (1 = Fraud, 0 = Not Fraud)

> 🚨 Disclaimer: This dataset is fully synthetic and NOT derived from real banking or ASB data.

## 🛠️ Methodology

1. **Data Wrangling**
   - Recoded agent types into categories (e.g., Android, iPhone, ATM).
   - Bucketed age and balance into meaningful groups.
   - Extracted hour from transaction timestamps.

2. **Feature Engineering**
   - Created a custom `RiskScore` based on domain rules (e.g., balance thresholds, time of day).
   - Labeled transactions as high, medium, or low risk.
   - 🚨 Introduced a custom **Risk Score** based on intuitive and data-driven fraud patterns.
     This custom risk score classifies transactions into high, medium, and low categories for fraud prevention,
     High risk transactions are a complete no-no in this system.
     PS: This system was designed by me under the influence of lot of caffeine.
3. **Model Building**
   - Upsampled minority class using the `themis` package.
   - Applied logistic regression with L1 regularization (`glmnet`).
   - Benchmarked against XGBoost (used as exploratory only).

4. **Evaluation Metrics**
   - **Accuracy:** 66.8%
   - **Sensitivity (Recall for Fraud):** 71.3%
   - **Specificity (Recall for Non-Fraud):** 78.2%
   - Not bad for a bank with no real customers 😄

## 📊 Key Insights

- Fraud spikes between **2 AM and 6 AM**—timing matters.
- **Senior citizens (Age > 70)** are at elevated risk.
- **Negative balance** and **high-value transactions** are major red flags.
- The model picked up on **2,000+ fraud cases**, simulating real-world value.

## 🎯 Why Logistic Regression?

It’s not flashy—but it's **interpretable**, fast, and perfect for showing decision-makers *why* a transaction was flagged. And in fraud detection, explainability can be just as important as accuracy.

## 📌 Visual Highlights

- Boxplots for age distribution by fraud label.
- Hour-wise heatmap of fraudulent transactions.
- Coefficient table showing variable importance.

## 🤔 What’s Next?

While the model shows promise, real-world implementation would require:
- Real-time transaction scoring infrastructure
- Dynamic model retraining as fraud evolves
- Integration with customer alerts and fraud teams

---
## ⚠️ But We Hit a Wall
This work was completed as part of the **BUSINFO 704** course at the **University of Auckland**. 

Made with 💻, ☕, and lots of `ggplot2`.

