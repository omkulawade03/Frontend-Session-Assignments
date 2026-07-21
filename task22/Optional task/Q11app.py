# ==========================================================
# Q11. Streamlit Input Interface
# ==========================================================

import streamlit as st

# Page Title
st.set_page_config(page_title="Breast Cancer Prediction")

# Heading
st.title("🎗️ Breast Cancer Prediction System")

st.write("Enter the patient details below.")

# -----------------------------
# Input Fields
# -----------------------------

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
    st.success("Input Received Successfully")


    