# AI-Powered Credit Risk Prediction & Analysis

An AI-powered credit risk assessment application that combines **Machine Learning, Explainable AI (SHAP), and Generative AI** to predict credit risk and provide understandable explanations for individual predictions.

The application uses an **Extra Trees Classifier** to predict credit risk, **SHAP** to identify the factors influencing each prediction, and the **Groq API** to generate a natural-language AI assessment based on the model's results.

## 🚀 Live Demo

🔗 **Streamlit App:https://credit-risk-prediction-app-o2c9xd6xjcjxtz3fmxqcfu.streamlit.app/

## 📌 Project Overview

Credit risk assessment is an important task in financial institutions, where the goal is to determine whether an applicant represents a higher or lower credit risk.

Traditional machine learning models can make accurate predictions but are often difficult to interpret.

This project addresses that problem by combining:

- **Machine Learning** → Predicts credit risk
- **SHAP Explainability** → Shows which features influenced the prediction
- **Generative AI** → Converts model insights into a human-readable explanation
- **Streamlit** → Provides an interactive web application

### Workflow

```text
Applicant Information
        ↓
Data Preprocessing
        ↓
Extra Trees Classifier
        ↓
Credit Risk Prediction
        ↓
Risk Probability / Score
        ↓
SHAP Explainability
        ↓
Important Risk Factors
        ↓
Groq Generative AI
        ↓
Natural-Language Risk Assessment
