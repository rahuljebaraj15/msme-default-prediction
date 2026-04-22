import streamlit as st
import pandas as pd
import joblib
import json

# Page config
st.set_page_config(
    page_title="MSME Risk Predictor",
    page_icon="🏦",
    layout="centered"
)

# Title
st.title("🏦 MSME Loan Default Predictor")
st.markdown("### Simple tool to assess loan risk")
st.write("Fill in borrower details to predict default probability.")

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

st.divider()

# Input section
st.subheader("📋 Enter Borrower Details")

user_input = {}

for feature in features:
    label = feature_labels.get(feature, feature)

    if feature == 'age':
        user_input[feature] = st.slider(label, 18, 100, 30)

    elif feature == 'MonthlyIncome':
        user_input[feature] = st.number_input(label, min_value=0, value=20000, step=1000)

    elif feature == 'RevolvingUtilizationOfUnsecuredLines':
        percent = st.slider(label, 0, 100, 30)
        user_input[feature] = percent / 100

    elif feature == 'DebtRatio':
        user_input[feature] = st.slider(label, 0.0, 2.0, 0.5)

    else:
        user_input[feature] = st.number_input(label, min_value=0, value=0, step=1)

# Convert input
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

st.divider()

# Prediction
if st.button("🔍 Predict Risk", use_container_width=True):

    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    # Progress bar
    st.progress(int(prob * 100))

    # Probability display
    st.metric("Default Probability", f"{prob*100:.2f}%")

    st.divider()

    # Risk classification
    if prob > 0.48:
        st.markdown("<h2 style='color:red;'>🔴 HIGH RISK</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:green;'>🟢 LOW RISK</h2>", unsafe_allow_html=True)

    st.divider()

    # Interpretation
    st.subheader("💡 Interpretation")

    if prob > 0.7:
        st.error("Very high risk. Strong chance of default due to poor repayment history and high debt.")
    elif prob > 0.4:
        st.warning("Moderate risk. Bank should review application carefully before approval.")
    else:
        st.success("Low risk borrower. Likely to repay loan on time.")

    st.divider()

    # Suggestions
    st.subheader("📌 Suggestions")

    if prob > 0.48:
        st.write("• Reduce outstanding debt")
        st.write("• Improve repayment history")
        st.write("• Avoid late payments")
        st.write("• Increase stable income sources")
    else:
        st.write("• Maintain good repayment habits")
        st.write("• Keep debt levels low")
        st.write("• Continue stable income flow")

    # Animation
    st.balloons()

st.divider()

st.caption("Model: Logistic Regression + SMOTE | Threshold = 0.48")
