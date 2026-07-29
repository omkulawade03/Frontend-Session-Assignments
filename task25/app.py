# app.py

import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("🔬 Breast Cancer Prediction App")
st.write("This application uses the best model saved from our ML Pipeline (`best_model.pkl`).")

# Load model safely
@st.cache_resource
def load_trained_model():
    try:
        return joblib.load('best_model.pkl')
    except FileNotFoundError:
        return None

model = load_trained_model()

if model is None:
    st.error("⚠️ `best_model.pkl` not found! Please run `python mini_project_pipeline.py` first.")
    st.stop()

st.success("✅ Model loaded successfully!")

# Input Form
st.header("Input Feature Sample")
st.write("Adjust feature parameters to predict patient diagnosis:")

col1, col2 = st.columns(2)

with col1:
    mean_radius = st.slider("Mean Radius", 6.0, 30.0, 14.0)
    mean_texture = st.slider("Mean Texture", 9.0, 40.0, 19.0)

with col2:
    mean_perimeter = st.slider("Mean Perimeter", 43.0, 190.0, 90.0)
    mean_area = st.slider("Mean Area", 143.0, 2500.0, 650.0)

# Prediction Button
if st.button("Predict Diagnosis", type="primary"):
    # Construct input array for 30 features (setting top features, remaining default to dataset averages)
    input_features = np.zeros((1, 30))
    input_features[0, 0] = mean_radius
    input_features[0, 1] = mean_texture
    input_features[0, 2] = mean_perimeter
    input_features[0, 3] = mean_area
    
    prediction = model.predict(input_features)[0]
    classes = ['Malignant (Cancerous)', 'Benign (Non-Cancerous)']
    
    st.markdown("---")
    st.subheader("Prediction Result:")
    if prediction == 0:
        st.error(f"Predicted Diagnosis: **{classes[prediction]}**")
    else:
        st.success(f"Predicted Diagnosis: **{classes[prediction]}**")