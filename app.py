import streamlit as st
import pandas as pd
import joblib
import json

st.title("🏦 MSME Default Risk Predictor")

# Load model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

with open('features.json') as f:
    features = json.load(f)

st.subheader("Enter Borrower Details")

user_input = {}

for feature in features:
    user_input[feature] = st.number_input(feature, value=0.0)

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"Default Probability: {prob:.2f}")

    if prob > 0.48:
        st.error("🔴 High Risk")
    else:
        st.success("🟢 Low Risk")
