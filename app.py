import streamlit as st
import pandas as pd
import joblib
import json

# Page config
st.set_page_config(page_title="MSME Risk Predictor", page_icon="🏦")

st.title("🏦 MSME Loan Default Predictor")
st.markdown("Enter borrower details to check risk level")

# Load model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

with open('features.json') as f:
    features = json.load(f)

# User-friendly labels
feature_labels = {
    'RevolvingUtilizationOfUnsecuredLines': "Credit Usage (%)",
    'age': "Age",
    'NumberOfTime30-59DaysPastDueNotWorse': "Late Payments (30-59 days)",
    'DebtRatio': "Debt Ratio",
    'MonthlyIncome': "Monthly Income (₹)",
    'NumberOfOpenCreditLinesAndLoans': "Total Loans",
    'NumberOfTimes90DaysLate': "Late Payments (90+ days)",
    'NumberRealEstateLoansOrLines': "Real Estate Loans",
    'NumberOfTime60-89DaysPastDueNotWorse': "Late Payments (60-89 days)",
    'NumberOfDependents': "Dependents"
}

st.subheader("📋 Enter Details")

user_input = {}

for feature in features:
    label = feature_labels.get(feature, feature)

    # SMART INPUT TYPES
    if feature == 'age':
        user_input[feature] = st.slider(label, 18, 100, 30)

    elif feature == 'MonthlyIncome':
        user_input[feature] = st.number_input(label, min_value=0, value=20000, step=1000)

    elif feature == 'RevolvingUtilizationOfUnsecuredLines':
        percent = st.slider(label, 0, 100, 30)
        user_input[feature] = percent / 100  # convert to decimal

    elif feature == 'DebtRatio':
        user_input[feature] = st.slider(label, 0.0, 2.0, 0.5)

    else:
        user_input[feature] = st.number_input(label, min_value=0, value=0, step=1)

# Convert to dataframe
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

st.divider()

# Prediction
if st.button("🔍 Predict Risk", use_container_width=True):

    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Result")

    # Display probability
    st.metric("Default Probability", f"{prob*100:.2f}%")

    # Risk output
    if prob > 0.48:
        st.error("🔴 HIGH RISK — Likely to Default")
    else:
        st.success("🟢 LOW RISK — Likely to Repay")

    # Simple explanation
    st.subheader("💡 Interpretation")

    if prob > 0.7:
        st.write("Very high risk due to poor repayment history or high debt.")
    elif prob > 0.4:
        st.write("Moderate risk. Bank should review before approval.")
    else:
        st.write("Low risk borrower with stable financial behavior.")
