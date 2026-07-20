import streamlit as st
import pandas as pd
import joblib

model = joblib.load("cancer_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Breast Cancer Prediction")

user_input = {}

for col in columns:
    user_input[col] = st.number_input(col, value=0.0)

if st.button("Predict"):

    sample = pd.DataFrame([user_input])

    sample = sample[columns]

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("Prediction: Malignant Cancer")
    else:
        st.success("Prediction: Benign")

        #for runnig this app used this command
        #python -m streamlit run app.py