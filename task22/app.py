import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Saved Model
# ==========================

model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")

# ==========================
# Streamlit Page
# ==========================

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details below.")

# ==========================
# User Inputs
# ==========================

age = st.number_input("Age", min_value=1, max_value=120, value=40)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

bp = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=0,
    max_value=700,
    value=200
)

fast = st.selectbox(
    "Fasting Blood Sugar >120 mg/dl",
    [0, 1]
)

ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "LVH", "ST"]
)

maxhr = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

angina = st.selectbox(
    "Exercise Induced Angina",
    ["Y", "N"]
)

oldpeak = st.number_input(
    "Old Peak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# ==========================
# Prediction
# ==========================

if st.button("Predict"):

    sample = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest,
        "RestingBP": bp,
        "Cholesterol": chol,
        "FastingBS": fast,
        "RestingECG": ecg,
        "MaxHR": maxhr,
        "ExerciseAngina": angina,
        "Oldpeak": oldpeak,
        "ST_Slope": slope
    }

    sample_df = pd.DataFrame([sample])

    # One-Hot Encoding
    sample_df = pd.get_dummies(sample_df)

    # Match Training Columns
    sample_df = sample_df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(sample_df)

    if prediction[0] == 1:
        st.error("❤️ Prediction: Heart Disease Detected")
    else:
        st.success("💚 Prediction: No Heart Disease")



        #to run the app used this command on terminal
        #python -m streamlit run app.py
        