import streamlit as st 
import pandas as pd 
import joblib
import os
print("App location:", os.path.abspath(__file__))
print("Current folder:", os.getcwd())
print("Files here:", os.listdir())

sex_encoder = joblib.load("Sex_encoder.pkl")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "extra_trees_credit_model.pkl"))

housing_encoder = joblib.load(os.path.join(BASE_DIR, "Housing_encoder.pkl"))
saving_encoder = joblib.load(os.path.join(BASE_DIR, "Saving accounts_encoder.pkl"))
checking_encoder = joblib.load(os.path.join(BASE_DIR, "Checking account_encoder.pkl"))

model = joblib. load("extra_trees_credit_model.pkl")
encoders = {col: joblib. load (f"{col}_encoder.pkl") 
            for col in ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]}

st. title("Credit Risk Prediction App")
st.write("Enter applicant information to predict if the credit risk is good or bad")

age = st. number_input("Age", min_value = 18, max_value= 80, value = 30)
sex = st. selectbox ("Sex", ["male", "female" ])
job = st. number_input("Job (0-3)", min_value = 0, max_value= 3, value = 1)
housing = st. selectbox ("Housing", ["own", "rent", "free"] )
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
checking_account = st. selectbox("Checking Account", ["little", "moderate", "rich"])
credit_amount = st. number_input("Credit Amount", min_value= 0, value = 1000)
duration = st. number_input ("Duration (months)", min_value= 1, value= 12)

purpose = st.selectbox(
    "Purpose",
    ["radio/TV", "education", "furniture/equipment", "car",
     "business", "domestic appliances", "repairs", "vacation/others"]
)

input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration],
    "Purpose": [encoders["Purpose"].transform([purpose])[0]]
})

if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    good_probability = probabilities[1]
    bad_probability = probabilities[0]

    # Risk score based on probability of bad credit
    risk_score = bad_probability * 100

    st.write("### Credit Risk Assessment")

    # Display risk score
    st.metric(
        "Credit Risk Score",
        f"{risk_score:.2f}/100"
    )

    # Classify risk level
    if risk_score < 30:
        risk_level = "LOW RISK"
        st.success(f"🟢 {risk_level}")

    elif risk_score < 60:
        risk_level = "MEDIUM RISK"
        st.warning(f"🟡 {risk_level}")

    else:
        risk_level = "HIGH RISK"
        st.error(f"🔴 {risk_level}")

    # Display prediction probabilities
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Probability of Good Credit",
            f"{good_probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Probability of Bad Credit",
            f"{bad_probability * 100:.2f}%"
        )
