
import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("best_asd_model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("ASD Screening Demo")

age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", ["male", "female"])
gender_val = 1 if gender == "male" else 0

# Example input (⚠️ replace with your actual features order later)
x = np.array([[age, gender_val]])
x_scaled = scaler.transform(x)
pred = model.predict(x_scaled)[0]
prob = model.predict_proba(x_scaled)[0][1]

st.write("Prediction:", "ASD" if pred==1 else "Non-ASD")
st.write("Probability of ASD:", round(prob, 2))
    
