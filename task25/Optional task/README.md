# Session 25 - Machine Learning Hyperparameter Tuning & Model Comparison

## Student Details
* **Name:** Om Prashant Kulawade
* **College:** Zeal Polytechnic, Narhe, Pune
* **Branch:** Artificial Intelligence and Machine Learning (AIML)
* **Domain:** AIML (Artificial Intelligence and Machine Learning)
* **Assignment:** Session 25 (AIML) - Optional Task (Mini Project)

---

## Project Overview

This project demonstrates an end-to-end Machine Learning pipeline for a **Diabetes Risk Prediction System**. It covers everything from dataset creation/loading, model implementation, hyperparameter tuning, model comparison, model persistence using `joblib`, and interactive deployment using Streamlit.

---

## Technical Stack & Libraries
* **Language:** Python 3.x
* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Machine Learning & Tuning:** `scikit-learn`, `xgboost`
* **Model Persistence:** `joblib`
* **Web Frontend:** `streamlit`

---

## Project Directory Structure

```text
task25/
│
├── diabetes.csv             # Clinical diabetes dataset
├── train_and_eval.py        # ML Pipeline (Preprocessing, Tuning, Evaluation, Saving)
├── scaler.pkl               # Saved StandardScaler object
├── best_diabetes_model.pkl  # Saved best-performing trained model
├── app.py                   # Streamlit web application frontend
└── README.md                # Project documentation