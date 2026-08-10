import streamlit as st 
import pandas as pd 
import joblib
import os
import shap
import openai
client = openai.OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)
print("App location:", os.path.abspath(__file__))
print("Current folder:", os.getcwd())
print("Files here:", os.listdir())

sex_encoder = joblib.load("Sex_encoder.pkl")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "extra_trees_credit_model.pkl"))
explainer = shap.TreeExplainer(model)

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
                    # Calculate SHAP values
    shap_values = explainer.shap_values(input_df)

    # For binary classification, select the BAD credit class
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]

    feature_names = [
        "Age",
        "Sex",
        "Job",
        "Housing",
        "Saving accounts",
        "Checking account",
        "Credit amount",
        "Duration",
        "Purpose"
    ]

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0]
    })

    shap_df["Absolute Impact"] = shap_df["SHAP Value"].abs()

    shap_df = shap_df.sort_values(
        "Absolute Impact",
        ascending=False
    )

    st.write("### 🧠 Why this prediction?")

    for _, row in shap_df.iterrows():

        feature = row["Feature"]
        impact = row["SHAP Value"]

        if impact > 0:
            st.write(
                f"🔴 **{feature}** — increased predicted credit risk"
            )
        else:
            st.write(
                f"🟢 **{feature}** — reduced predicted credit risk"
            )
            top_factors = shap_df.head(5)

factor_text = ""

for _, row in top_factors.iterrows():
    factor_text += (
        f"{row['Feature']}: "
        f"{row['SHAP Value']:.4f}\n"
    )
    response = client.responses.create(
    model="openai/gpt-oss-20b",
    input=f"""
You are a credit risk explanation assistant.

Explain the machine-learning prediction below in simple,
professional language.

Do NOT make a new credit decision.
Do NOT invent information.
Only explain the provided model output and factors.

Risk score: {risk_score:.2f}/100
Risk level: {risk_level}
Probability of bad credit: {bad_probability * 100:.2f}%

The most important SHAP factors are:

{factor_text}

Write:
1. A short overall assessment.
2. The main factors increasing risk.
3. The main factors reducing risk.

Keep the explanation under 150 words.
"""
)
    ai_explanation = response.output_text

st.write("### 🤖 AI Risk Assessment")
st.write(ai_explanation)
