import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Risk Prediction App")
st.write("Enter the patient's health parameters below to assess risk using our trained machine learning pipeline.")

st.divider()

# Input Form
with st.form("prediction_form"):
    st.subheader("Patient Clinical Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=120.0)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0)
        
    with col2:
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=33)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        
    submit_button = st.form_submit_button(label="Predict Health Status")

# Prediction Execution
if submit_button:
    try:
        # Load saved artifacts
        model = joblib.load("best_diabetes_model.pkl")
        scaler = joblib.load("scaler.pkl")
        
        # Prepare input array
        input_data = np.array([[glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age, pregnancies]])
        
        # Apply scaling if the model is distance-based (SVM/KNN)
        model_type = type(model).__name__
        if "SVC" in model_type or "KNeighbors" in model_type:
            input_data = scaler.transform(input_data)
            
        prediction = model.predict(input_data)[0]
        
        # Calculate confidence score if model supports probability output
        prob_str = ""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            prob_str = f" (Confidence: {probs[prediction]*100:.1f}%)"
        
        st.divider()
        st.subheader("Prediction Outcome")
        
        if prediction == 1:
            st.error(f"⚠️ **High Risk:** The model indicates a high likelihood of Diabetes{prob_str}.")
        else:
            st.success(f"✅ **Low Risk:** The model indicates low risk for Diabetes{prob_str}.")
            
        st.caption(f"Active Model: **{model_type}**")
        
    except FileNotFoundError:
        st.error("Model artifacts not found! Run `python train_and_eval.py` first to train the pipeline and generate `best_diabetes_model.pkl` and `scaler.pkl`.")