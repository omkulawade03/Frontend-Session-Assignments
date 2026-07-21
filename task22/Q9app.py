# ==========================================
# Q9. Streamlit Input Interface
# ==========================================

import streamlit as st

# Page Configuration
st.set_page_config(page_title="Heart Disease Prediction")

# Title
st.title("❤️ Heart Disease Prediction System")

st.write("Enter the patient details below.")

# ------------------------------
# Input Fields
# ------------------------------

age = st.number_input("Age", min_value=1, max_value=120, value=45)

sex = st.selectbox("Sex", ["M", "F"])

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

# ------------------------------
# Predict Button
# ------------------------------

if st.button("Predict"):
    st.success("Input received successfully!")



    
        #to run this app used this command 
        #python -m streamlit run Q9app.py