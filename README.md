# LoanGuard AI - Loan Approval System 🏦📊

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

**LoanGuard AI** is a machine learning-powered credit risk prediction system designed to assist financial institutions in making fast, accurate, and data-driven loan approval decisions. The system analyzes an applicant's financial profile across key variables and predicts whether they are likely to fully repay a loan or default on it.

## 🚀 Live Application
We have deployed the final model on Streamlit for real-time risk assessment. 

🔗 **Access the Live Web App Here:** [Loan Approval System App](https://ai-based-loan-approval-system.streamlit.app/)

---

## 🌟 Key Features
- **Predictive Accuracy:** Uses an optimized **XGBoost Classifier** tuned via RandomizedSearchCV.
- **Risk Assessment:** Specifically engineered financial ratios such as Debt-to-Income (DTI), Credit Utilization, and Loan-to-Income (LTI) ratios.
- **Imbalance Handling:** Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance, ensuring fair and unbiased predictions.
- **Strict Approval Threshold:** Prioritizes lender protection by using a conservative `0.71` probability threshold for approvals.
- **Interactive Web UI:** Clean, responsive, and easy-to-use interface built with Streamlit.

## 🛠️ Technology Stack
- **Language:** Python
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
- **Deployment:** Streamlit
- **Model Serialization:** Joblib

## 📊 Dataset Overview
The dataset contains historical credit records of loan applicants.
- **Total Records:** 110,000 valid records (after preprocessing).
- **Target Variable:** `Loan Status` (Fully Paid vs. Charged Off).
- **Key Predictors:** Credit Score, Annual Income, Current Loan Amount, Term, Purpose, Home Ownership, and more.

## ⚙️ Machine Learning Pipeline
1. **Data Preprocessing:** Imputed missing values (median for numerical, mode for categorical), handled extreme outliers using IQR, and dropped redundant features.
2. **Feature Engineering:** Derived critical ratios (DTI, Credit Utilization, LTI) to better capture financial stress.
3. **Feature Selection:** Ranked importance using a Random Forest model, reducing the feature space down to the top 20 most impactful predictors.
4. **Model Selection:** Tested Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, and a Stacking Classifier before finalizing **XGBoost**.

## 💻 Installation & Setup

To run this project locally, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Loan-Approval-System.git
   cd Loan-Approval-System
   ```

2. **Install the required dependencies**
   Make sure you have Python installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure `streamlit`, `xgboost`, `scikit-learn`, `pandas`, and `numpy` are in your environment).*

3. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
```text
├── Dataset/                       # Contains the raw dataset (Loan Approval Dataset.csv)
├── Models/                        # Serialized, trained models (xgb_model.pkl, scaler.pkl)
├── Notebook/                      # Jupyter notebook for EDA and model training (Loan_Approval_System.ipynb)
├── Project Report/                # Final project documentation and PDF reports
├── Streamlit UI/                  # Streamlit web application script and assets
├── requirements.txt               # Dependencies required to run the app
└── README.md                      # Project overview and instructions
```

## 👥 Contributors
- **Muhammad Kaif**
- **Laiba Wazir**

---
*Developed as the Final Project for the People’s Information and Technology Programme (PITP) Phase – II Data Science Course.*
