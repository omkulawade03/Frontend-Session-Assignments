# 🎗️ Breast Cancer Prediction System

## ASSIGNMENT 22 (Optional Task)

### Student Information

**Name:** Om Prashant Kulawade  
**College:** Zeal Polytechnic, Narhe, Pune  
**Branch:** Artificial Intelligence and Machine Learning (AIML)  
**Domain:** Artificial Intelligence and Machine Learning (AIML)

---

# 📌 Project Description

The **Breast Cancer Prediction System** is a Machine Learning project developed using the **Logistic Regression** algorithm. The application predicts whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** based on diagnostic features entered by the user.

A **Streamlit Web Application** is developed to provide a simple and interactive interface for users to make predictions.

---

# 📊 Dataset Information

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset

**Target Variable:** `diagnosis`

### Target Classes

- **M** → Malignant (Cancer)
- **B** → Benign (No Cancer)

---

# 🎯 Objectives

- Load the dataset
- Perform Data Cleaning
- Perform Exploratory Data Analysis (EDA)
- Select Features
- Scale Numerical Features
- Train Logistic Regression Model
- Evaluate Model Performance
- Save and Load Trained Model
- Develop Streamlit Web Application

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

# 📂 Project Structure

```text
Breast_Cancer_Prediction/
│
├── data.csv
├── eda.ipynb
├── train_model.py
├── app.py
├── breast_cancer_model.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
└── README.md
```

---

# ⚙ Installation

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

---

# ▶️ Steps to Run the Project

## Step 1: Open EDA Notebook

```bash
jupyter notebook eda.ipynb
```

Perform:

- Dataset Loading
- Data Cleaning
- Missing Value Handling
- Exploratory Data Analysis
- Feature Selection
- Feature Scaling

---

## Step 2: Train the Machine Learning Model

```bash
python train_model.py
```

This will:

- Train Logistic Regression Model
- Evaluate Model
- Save the trained files:

```
breast_cancer_model.pkl
scaler.pkl
columns.pkl
```

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

# 📈 Exploratory Data Analysis (EDA)

The following analyses are performed:

- Dataset Information
- Missing Value Analysis
- Statistical Summary
- Class Distribution
- Correlation Heatmap
- Histogram Visualization
- Boxplot Visualization

---

# 🤖 Machine Learning Workflow

1. Dataset Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Feature Scaling
6. Train-Test Split
7. Logistic Regression Model Training
8. Prediction
9. Model Evaluation
10. Save Model
11. Load Saved Model
12. Streamlit Deployment

---

# 📊 Model Evaluation Metrics

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# ✨ Features

- Breast Cancer Prediction
- Binary Classification
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Selection
- Standard Scaling
- Logistic Regression
- Model Evaluation
- Save & Load Model
- Interactive Streamlit Interface

---

# 🎯 Prediction Output

The application predicts one of the following:

🟢 **Benign (No Cancer Detected)**

🔴 **Malignant (Cancer Detected)**

---

# 📸 Sample Output

### Streamlit Home Page

- User enters diagnostic feature values.
- Clicks the **Predict** button.

### Prediction Result

```
Prediction: Benign
```

or

```
Prediction: Malignant
```

---

# 📁 Generated Files

After running the project:

```
breast_cancer_model.pkl
scaler.pkl
columns.pkl
```

These files are used by the Streamlit application for prediction.

---

# 🌐 Dataset Source

**Breast Cancer Wisconsin (Diagnostic) Dataset**

UCI Machine Learning Repository

https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

---

# 👨‍🎓 Submitted By

**Name:** Om Prashant Kulawade

**College:** Zeal Polytechnic, Narhe, Pune

**Branch:** Artificial Intelligence and Machine Learning (AIML)

**Domain:** Artificial Intelligence and Machine Learning (AIML)

**Assignment:** Session 22 – Optional Task (Binary Classification Project)

**Academic Year:** 2026

---

# 🙏 Thank You