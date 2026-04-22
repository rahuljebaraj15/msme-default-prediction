import streamlit as st
import pandas as pd
import joblib
import json
import time

# Page config
st.set_page_config(
    page_title="MSME Risk Predictor",
    page_icon="🏦",
    layout="centered"
)

# Title
st.title("🏦 MSME Loan Risk Assessment Tool")
st.markdown("### Decision Support System for Credit Risk Evaluation")

# Load model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

with open('features.json') as f:
    features = json.load(f)

# Labels
feature_labels = {
    'RevolvingUtilizationOfUnsecuredLines': "Credit Usage (%)",
    'age': "Age",
    'NumberOfTime30-59DaysPastDueNotWorse': "Late Payments (30–59 days)",
    'DebtRatio': "Debt Ratio",
    'MonthlyIncome': "Monthly Income (₹)",
    'NumberOfOpenCreditLinesAndLoans': "Total Loans",
    'NumberOfTimes90DaysLate': "Late Payments (90+ days)",
    'NumberRealEstateLoansOrLines': "Real Estate Loans",
    'NumberOfTime60-89DaysPastDueNotWorse': "Late Payments (60–89 days)",
    'NumberOfDependents': "Dependents"
}

st.divider()

# Input section
st.subheader("📋 Borrower Profile")

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

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

st.divider()

# Prediction
if st.button("🔍 Assess Risk", use_container_width=True):

    with st.spinner("Analyzing borrower profile..."):
        time.sleep(1.5)

    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Risk Assessment")

    # Progress bar
    st.progress(int(prob * 100))

    # Probability
    st.metric("Probability of Default", f"{prob*100:.2f}%")

    st.divider()

    # Risk category
    if prob > 0.6:
        st.error("🔴 HIGH RISK BORROWER")
    elif prob > 0.4:
        st.warning("🟡 MODERATE RISK BORROWER")
    else:
        st.success("🟢 LOW RISK BORROWER")

    st.divider()

    # Interpretation
    st.subheader("💡 Model Insight")

    if prob > 0.6:
        st.write("Borrower shows significant indicators of financial stress and repayment risk.")
    elif prob > 0.4:
        st.write("Borrower has moderate risk. Some indicators suggest caution.")
    else:
        st.write("Borrower demonstrates stable financial behavior with low default risk.")

    st.divider()

    # BANK DECISION RECOMMENDATIONS
    st.subheader("🏦 Bank Decision Recommendation")

    if prob > 0.6:
        st.markdown("""
        **Recommended Action:**
        - Reject loan OR require strong collateral  
        - Perform detailed credit investigation  
        - Consider reducing loan exposure  
        """)
    elif prob > 0.4:
        st.markdown("""
        **Recommended Action:**
        - Approve with conditions  
        - Require collateral or guarantor  
        - Offer reduced loan amount  
        """)
    else:
        st.markdown("""
        **Recommended Action:**
        - Safe to approve loan  
        - Standard lending terms applicable  
        - Low monitoring required  
        """)

    st.divider()

    # PROFESSIONAL VISUAL FEEDBACK
    if prob < 0.4:
        st.toast("Low risk profile detected", icon="✅")
    elif prob > 0.6:
        st.toast("High risk alert generated", icon="⚠️")

st.divider()
st.caption("Model: Logistic Regression + SMOTE | Decision Threshold = 0.48")
