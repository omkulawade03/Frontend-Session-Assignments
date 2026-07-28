# Session 24: AIML Assignment — Multi-Model Training & Streamlit Web Application

## Student Details
* **Name:** Om Prashant Kulawade
* **College:** Zeal Polytechnic, Narhe, Pune
* **Branch:** Artificial Intelligence and Machine Learning (AIML)
* **Domain:** Artificial Intelligence and Machine Learning (AIML)
* **GitHub Repository:** [omkulawade03/Frontend-Session-Assignments](https://github.com/omkulawade03/Frontend-Session-Assignments.git)

---

## Project Overview
This project fulfills the requirements for **Session 24 (AIML)**. It covers data preprocessing, training and evaluating multiple Machine Learning classification and regression algorithms, selecting and serializing the top-performing models using `joblib`, and serving them through an interactive multi-model **Streamlit** web application.

---

## Assignment Structure & Implementation

### **Q1: Dataset Selection & Preprocessing**
* **Classification Dataset:** Iris Flower Dataset (Target: `species`)
* **Regression Dataset:** California Housing Dataset (Target: `MedHouseVal`)
* **Preprocessing Pipeline:**
  * Cleaned missing values and split features into independent ($X$) and dependent ($y$) variables.
  * Train-Test Split (80% Train, 20% Test).
  * Feature scaling applied via `StandardScaler` for distance- and gradient-based algorithms.

---

### **Q2: Classification Algorithms**
Evaluated five classification models using **Accuracy**, **Confusion Matrix**, and **Classification Report**:
1. Logistic Regression
2. Decision Tree Classifier
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN)
5. Naive Bayes

---

### **Q3: Regression Algorithms**
Evaluated four regression models using the **$R^2$ Score**:
1. Linear Regression
2. Decision Tree Regressor
3. Support Vector Regressor (SVR)
4. K-Nearest Neighbors Regressor

---

### **Q4: Best Model Selection & Serialization**
* Selected the best-performing models based on validation metrics.
* Saved models, feature schemas, scalers, and label maps into joblib `.pkl` artifacts (`best_classification_model.pkl`, `best_regression_model.pkl`, and `all_trained_models.pkl`).

---

### **Q5: Streamlit Multi-Model Web Application**
Built an interactive interface allowing users to:
* Select between **Classification** and **Regression** problem types.
* Choose any trained algorithm from a dropdown menu.
* Input feature values dynamically via numeric input controls.
* Obtain instant predictions and probability/value outputs.

---

## Repository Structure

```text
├── train_and_save.py         # Script for preprocessing, training, evaluation & saving models
├── app.py                    # Interactive Streamlit Web Application
├── best_classification_model.pkl  # Serialized best classifier
├── best_regression_model.pkl      # Serialized best regressor
├── all_trained_models.pkl         # Package of all trained algorithms & scalers
└── README.md                 # Project Documentation