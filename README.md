# 🏦 Loan Approval Predictor

A Machine Learning project that predicts whether a loan
application will be approved or rejected based on applicant
details using Random Forest Classification.

## 🔗 Live Demo

[Click here to try the app](your-streamlit-link-here)

## 📌 Project Overview

This project simulates a real bank loan approval system.
The model learns patterns from 1000 loan applications and
predicts approval based on salary, credit score, loan amount,
age and years employed with 99.5% accuracy.

## 🧠 How It Works

1. User enters their details
2. Random Forest model analyzes the data
3. Model checks patterns from 1000 loan applications
4. Predicts Approved or Rejected instantly

## 🛠️ Tech Stack

- Python
- Scikit-learn (Machine Learning)
- Pandas (Data handling)
- NumPy (Dataset creation)
- Streamlit (Web UI)
- Joblib (Model saving)

## 📊 Model Performance

- Decision Tree Accuracy: 99.5%
- Random Forest Accuracy: 99.5%
- Algorithm used: Random Forest (100 trees)

## 📁 Project Structure

loan_approval_ml/
├── dataset/
│ └── loan_data.csv
├── model/
│ └── loan_model.pkl
├── src/
│ └── main.py
│ └── app.py
└── requirements.txt

## ⚙️ How to Run Locally

1. Clone the repo
2. Install requirements
   pip install -r requirements.txt
3. Run the app
   streamlit run src/app.py

## 📦 Dataset

- 1000 synthetic loan applications
- Created using NumPy random generation
- Features: age, salary, credit score,
  loan amount, years employed

## 🔍 Approval Criteria

- Annual salary > 50,000
- Credit score > 650
- Loan amount < 5x salary
- Years employed > 2

## Author

Akshay Prakash
