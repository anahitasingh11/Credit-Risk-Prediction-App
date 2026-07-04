# Credit Risk Prediction App

## Overview

The Credit Risk Prediction App is a Machine Learning web application built using Streamlit. It predicts whether a loan applicant has a **Good** or **Bad** credit risk based on personal and financial information.

The application uses an Extra Trees Classifier trained on the German Credit Risk dataset and provides instant predictions through a simple and user-friendly interface.

---

## Features

- Predicts credit risk as **Good** or **Bad**
- Interactive Streamlit web application
- Uses trained Machine Learning model
- Encodes categorical features automatically
- Easy-to-use interface for entering applicant details

---

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- NumPy

---

## Machine Learning Model

**Algorithm Used:**
- Extra Trees Classifier

**Input Features:**
- Age
- Sex
- Job
- Housing
- Saving Accounts
- Checking Account
- Credit Amount
- Duration
- Purpose

**Output:**
- Good Credit Risk
- Bad Credit Risk

---

## Project Structure

```
credit-risk-prediction-app/
│
├── app.py
├── analysis_model.ipynb
├── extra_trees_credit_model.pkl
├── Sex_encoder.pkl
├── Housing_encoder.pkl
├── Saving accounts_encoder.pkl
├── Checking account_encoder.pkl
├── Purpose_encoder.pkl
└── README.md



## How to Use

1. Open the application.
2. Enter the applicant's details.
3. Click **Predict Risk**.
4. The application displays whether the predicted credit risk is **Good** or **Bad**.

---

## Dataset

This project is based on the **German Credit Risk Dataset**, which contains customer demographic and financial information used to predict creditworthiness.

---

## Future Improvements

- Display prediction confidence score
- Improve UI design and responsiveness
- Add data visualization dashboard
- Deploy using Streamlit Community Cloud
- Compare multiple machine learning models

---

## Author

**Anahita Singh**

Bachelor of Technology (KIIT DU)

---

## License

This project is developed for educational and learning purposes.
