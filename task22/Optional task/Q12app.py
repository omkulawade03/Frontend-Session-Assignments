# ==========================================================
# Q12. Complete Streamlit Web App
# ==========================================================

import streamlit as st
import pandas as pd
import joblib

# Load Saved Files
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Breast Cancer Prediction")

st.title("🎗️ Breast Cancer Prediction System")

# Input Fields
radius_mean = st.number_input("Radius Mean", value=17.99)
texture_mean = st.number_input("Texture Mean", value=10.38)
perimeter_mean = st.number_input("Perimeter Mean", value=122.8)
area_mean = st.number_input("Area Mean", value=1001.0)
smoothness_mean = st.number_input("Smoothness Mean", value=0.1184)
compactness_mean = st.number_input("Compactness Mean", value=0.2776)
concavity_mean = st.number_input("Concavity Mean", value=0.3001)
concave_points_mean = st.number_input("Concave Points Mean", value=0.1471)
symmetry_mean = st.number_input("Symmetry Mean", value=0.2419)
fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.07871)

# Predict Button
if st.button("Predict"):

    sample = pd.DataFrame([{
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "perimeter_mean": perimeter_mean,
        "area_mean": area_mean,
        "smoothness_mean": smoothness_mean,
        "compactness_mean": compactness_mean,
        "concavity_mean": concavity_mean,
        "concave points_mean": concave_points_mean,
        "symmetry_mean": symmetry_mean,
        "fractal_dimension_mean": fractal_dimension_mean
    }])

    # Add missing columns
    for col in columns:
        if col not in sample.columns:
            sample[col] = 0

    sample = sample[columns]

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("🔴 Prediction: Malignant Tumor")
    else:
        st.success("🟢 Prediction: Benign Tumor")



#to run this code used command 
#python -m streamlit run Q12app.py
        