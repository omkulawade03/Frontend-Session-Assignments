import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AIML Multi-Model Inference App",
    page_icon="🤖",
    layout="wide"
)

# Load saved models artifact
@st.cache_resource
def load_artifacts():
    return joblib.load('all_trained_models.pkl')

try:
    artifacts = load_artifacts()
except Exception as e:
    st.error("Could not find 'all_trained_models.pkl'. Please run 'train_and_save.py' first!")
    st.stop()

st.title("🤖 Multi-Model ML Prediction App")
st.markdown("Select a problem type, choose a model, adjust feature inputs, and get immediate predictions.")

# Sidebar Selection
st.sidebar.header("Configuration")
problem_type = st.sidebar.radio("1. Select Problem Type", ["Classification", "Regression"])

if problem_type == "Classification":
    clf_data = artifacts["classification"]
    selected_model_name = st.sidebar.selectbox("2. Select Classification Model", list(clf_data["models"].keys()))
    
    st.header(f"Iris Flower Classification — `{selected_model_name}`")
    st.markdown("### Enter Input Feature Values:")
    
    col1, col2 = st.columns(2)
    inputs = {}
    
    # Feature inputs (Iris defaults)
    feature_defaults = {
        "sepal length (cm)": 5.8,
        "sepal width (cm)": 3.0,
        "petal length (cm)": 3.8,
        "petal width (cm)": 1.2
    }
    
    for i, feature in enumerate(clf_data["features"]):
        target_col = col1 if i % 2 == 0 else col2
        inputs[feature] = target_col.number_input(
            f"**{feature.title()}**", 
            value=float(feature_defaults.get(feature, 1.0)), 
            step=0.1
        )
    
    if st.button("Predict Species", type="primary"):
        input_df = pd.DataFrame([inputs])
        
        # Check scaling requirements
        if clf_data["scaling_reqs"][selected_model_name]:
            input_processed = clf_data["scaler"].transform(input_df)
        else:
            input_processed = input_df
            
        model = clf_data["models"][selected_model_name]
        prediction = model.predict(input_processed)[0]
        predicted_class = clf_data["target_names"][prediction]
        
        st.success(f"**Predicted Target Class:** `{predicted_class.title()}` (Class ID: {prediction})")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_processed)[0]
            prob_df = pd.DataFrame({
                "Species": clf_data["target_names"],
                "Probability": probs
            })
            st.bar_chart(prob_df.set_index("Species"))

else:
    reg_data = artifacts["regression"]
    selected_model_name = st.sidebar.selectbox("2. Select Regression Model", list(reg_data["models"].keys()))
    
    st.header(f"California Housing Price Prediction — `{selected_model_name}`")
    st.markdown("### Enter Input Feature Values:")
    
    col1, col2 = st.columns(2)
    inputs = {}
    
    # Feature defaults for California Housing
    feature_defaults = {
        "MedInc": 3.87,
        "HouseAge": 28.0,
        "AveRooms": 5.4,
        "AveBedrms": 1.1,
        "Population": 1425.0,
        "AveOccup": 3.0,
        "Latitude": 35.6,
        "Longitude": -119.5
    }
    
    for i, feature in enumerate(reg_data["features"]):
        target_col = col1 if i % 2 == 0 else col2
        inputs[feature] = target_col.number_input(
            f"**{feature}**", 
            value=float(feature_defaults.get(feature, 1.0)),
            step=0.1
        )
        
    if st.button("Predict Median House Value", type="primary"):
        input_df = pd.DataFrame([inputs])
        
        # Check scaling requirements
        if reg_data["scaling_reqs"][selected_model_name]:
            input_processed = reg_data["scaler"].transform(input_df)
        else:
            input_processed = input_df
            
        model = reg_data["models"][selected_model_name]
        prediction = model.predict(input_processed)[0]
        
        # Target variable MedHouseVal is in $100,000s
        actual_value = prediction * 100000
        
        st.success(f"**Predicted Median House Value:** `${actual_value:,.2f}` (`{prediction:.4f}` in $100k units)")