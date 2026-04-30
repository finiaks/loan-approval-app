import streamlit as st
import joblib as job
import numpy as np

#Load Model
model = job.load("model/loan_model.pkl")

#Title
st.title("Loan Approval Predictor")
st.write("Enter your details to check if your loan will be approved!")

#Input Fields
age = st.number_input("Age", min_value = 18, max_value = 100, value = 30)
salary = st.number_input("Annual Salary", min_value = 0, value = 50000)
credit_score = st.number_input("Credit Score", min_value = 300, max_value = 850, value = 650)
loan_amount = st.number_input("Loan Amount", min_value = 0, value = 100000)
years_employed = st.number_input("Years Employed", min_value = 0, max_value = 50, value = 3)

#Predict Button
if st.button("Check Loan Approval"):
    input_data = np.array([[age,salary,credit_score,loan_amount,years_employed]])
    result = model.predict(input_data)

    if result[0] == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Rejected!")



