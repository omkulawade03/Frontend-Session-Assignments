# ==========================================
# Q10. Complete Streamlit Web App
# ==========================================

# Import Required Libraries
import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------
# Load Saved Model and Objects
# ------------------------------------------

model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------------------
# Streamlit Page
# ------------------------------------------

st.set_page_config(page_title="Heart Disease Prediction")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient details below.")

# ------------------------------------------
# Input Fields
# ------------------------------------------

age = st.number_input("Age", min_value=1, max_value=120, value=45)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure",
    value=120
)

cholesterol = st.number_input(
    "Cholesterol",
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.number_input(
    "Maximum Heart Rate",
    value=150
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["Y", "N"]
)

oldpeak = st.number_input(
    "Oldpeak",
    value=1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# ------------------------------------------
# Predict Button
# ------------------------------------------

if st.button("Predict"):

    # Create patient record
    sample = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    # One-Hot Encoding
    sample = pd.get_dummies(sample)

    # Match columns with training data
    sample = sample.reindex(columns=columns, fill_value=0)

    # Apply scaler (if used)
    if scaler is not None:
        sample = scaler.transform(sample)

    # Predict
    prediction = model.predict(sample)

    # Display Result
    if prediction[0] == 1:
        st.error("❤️ Heart Disease: Yes")
    else:
        st.success("💚 Heart Disease: No")



        #to run this app used this command 
        #python -m streamlit run Q10app.py