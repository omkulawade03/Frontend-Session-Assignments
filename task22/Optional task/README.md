# 🎗️ Breast Cancer Prediction System

# ASSIGNMENT 22 (Optional Task)

## Student Information

**Name:** Om Prashant Kulawade  
**College:** Zeal Polytechnic, Narhe, Pune  
**Branch:** Artificial Intelligence and Machine Learning (AIML)  
**Domain:** Artificial Intelligence and Machine Learning (AIML)

---

# Project Description

The Breast Cancer Prediction System is a Machine Learning project developed using the Logistic Regression algorithm. The application predicts whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** based on diagnostic features entered by the user. A Streamlit web application provides a simple and interactive interface for making predictions.

---

# Dataset Information

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset

**Target Variable:** `diagnosis`

Target Classes:

- **M** → Malignant (Cancer)
- **B** → Benign (No Cancer)

---

# Objectives

- Load and clean the dataset.
- Perform Exploratory Data Analysis (EDA).
- Select important features.
- Scale numerical features.
- Train a Logistic Regression model.
- Evaluate model performance.
- Save and reload the trained model.
- Develop an interactive Streamlit web application.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

# Project Structure

```
Breast_Cancer_Prediction/
│
├── data.csv                  # Dataset
├── eda.py                    # Data Cleaning & EDA
├── train_model.py            # Model Training
├── app.py                    # Streamlit Application
├── cancer_model.pkl          # Saved ML Model
├── scaler.pkl                # Standard Scaler
├── columns.pkl               # Feature Columns
├── requirements.txt          # Required Libraries
└── README.md                 # Project Documentation
```

---

# Installation

Install all required libraries:

```bash
python -m pip install -r requirements.txt
```

---

# Steps to Run the Project

## Step 1: Perform Data Cleaning & EDA

```bash
python eda.py
```

This will:

- Load dataset
- Handle missing values
- Remove unnecessary columns
- Encode target variable
- Display graphs
- Save cleaned dataset

---

## Step 2: Train the Machine Learning Model

```bash
python train_model.py
```

This will:

- Load cleaned dataset
- Scale numerical features
- Train Logistic Regression model
- Evaluate model
- Save:
  - cancer_model.pkl
  - scaler.pkl
  - columns.pkl

---

## Step 3: Run the Streamlit Application

```bash
python -m streamlit run app.py
```

Open your browser and visit:

```
http://localhost:8501
```

---

# Exploratory Data Analysis (EDA)

The following analyses are performed:

- Dataset Information
- Missing Value Analysis
- Statistical Summary
- Class Distribution
- Correlation Heatmap
- Feature Selection

---

# Machine Learning Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Data Scaling
6. Train-Test Split
7. Logistic Regression Model Training
8. Model Evaluation
9. Save Model
10. Load Model
11. Streamlit Deployment

---

# Model Evaluation

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# Features

- Breast Cancer Prediction
- Data Cleaning
- Exploratory Data Analysis
- Feature Selection
- Feature Scaling
- Logistic Regression
- Model Evaluation
- Save & Load Model
- Interactive Streamlit Interface

---

# Prediction Output

The application predicts one of the following:

🟢 **Benign (No Cancer)**

🔴 **Malignant (Cancer Detected)**

---

# Dataset Source

Breast Cancer Wisconsin (Diagnostic) Dataset

UCI Machine Learning Repository

https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

---

# Submitted By

**Name:** Om Prashant Kulawade

**College:** Zeal Polytechnic, Narhe, Pune

**Branch:** Artificial Intelligence and Machine Learning (AIML)

**Domain:** Artificial Intelligence and Machine Learning (AIML)

**Assignment:** Session 22 – Optional Task (Binary Classification Project)

**Academic Year:** 2026

---

# Thank You