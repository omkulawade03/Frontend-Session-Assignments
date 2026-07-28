import joblib
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Classification Imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Regression Imports
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# ==========================================
# Q1: DATASET SELECTION & PREPROCESSING
# ==========================================
print("--- Q1: DATASET SELECTION & PREPROCESSING ---")

# --- 1A. Classification Dataset (Iris) ---
iris = load_iris(as_frame=True)
clf_df = iris.frame

X_clf = clf_df.drop(columns=['target'])  # Independent Features
y_clf = clf_df['target']                 # Dependent Feature

# REPLACED: test_score=0.2 -> test_size=0.2
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Scaling features for distance/gradient based classifiers
clf_scaler = StandardScaler()
X_clf_train_scaled = clf_scaler.fit_transform(X_clf_train)
X_clf_test_scaled = clf_scaler.transform(X_clf_test)

# --- 1B. Regression Dataset (California Housing - Sampled for Speed) ---
california = fetch_california_housing(as_frame=True)
reg_df = california.frame.sample(n=3000, random_state=42) # Sampled for faster training

X_reg = reg_df.drop(columns=['MedHouseVal'])  # Independent Features
y_reg = reg_df['MedHouseVal']                 # Dependent Feature

# REPLACED: test_score=0.2 -> test_size=0.2
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scaling features for regression models
reg_scaler = StandardScaler()
X_reg_train_scaled = reg_scaler.fit_transform(X_reg_train)
X_reg_test_scaled = reg_scaler.transform(X_reg_test)


# ==========================================
# Q2: CLASSIFICATION ALGORITHMS EVALUATION
# ==========================================
print("\n--- Q2: CLASSIFICATION EVALUATION ---")

clf_models = {
    "Logistic Regression": (LogisticRegression(max_iter=200), True),
    "Decision Tree Classifier": (DecisionTreeClassifier(random_state=42), False),
    "Support Vector Machine (SVM)": (SVC(), True),
    "K-Nearest Neighbors (KNN)": (KNeighborsClassifier(), True),
    "Naive Bayes": (GaussianNB(), False)
}

clf_results = {}
trained_clf_models = {}

for name, (model, needs_scale) in clf_models.items():
    X_tr = X_clf_train_scaled if needs_scale else X_clf_train
    X_te = X_clf_test_scaled if needs_scale else X_clf_test
    
    # Train & Predict
    model.fit(X_tr, y_clf_train)
    y_pred = model.predict(X_te)
    
    # Evaluate
    acc = accuracy_score(y_clf_test, y_pred)
    cm = confusion_matrix(y_clf_test, y_pred)
    cr = classification_report(y_clf_test, y_pred, target_names=iris.target_names)
    
    clf_results[name] = acc
    trained_clf_models[name] = model
    
    print(f"\n>>> Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

# Comparison Table
clf_summary = pd.DataFrame(list(clf_results.items()), columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
print("\nClassification Models Comparison:")
print(clf_summary.to_string(index=False))


# ==========================================
# Q3: REGRESSION ALGORITHMS EVALUATION
# ==========================================
print("\n--- Q3: REGRESSION EVALUATION ---")

reg_models = {
    "Linear Regression": (LinearRegression(), True),
    "Decision Tree Regressor": (DecisionTreeRegressor(random_state=42), False),
    "Support Vector Regressor (SVR)": (SVR(), True),
    "K-Nearest Neighbors Regressor": (KNeighborsRegressor(), True)
}

reg_results = {}
trained_reg_models = {}

for name, (model, needs_scale) in reg_models.items():
    X_tr = X_reg_train_scaled if needs_scale else X_reg_train
    X_te = X_reg_test_scaled if needs_scale else X_reg_test
    
    # Train & Predict
    model.fit(X_tr, y_reg_train)
    y_pred = model.predict(X_te)
    
    # Evaluate
    r2 = r2_score(y_reg_test, y_pred)
    reg_results[name] = r2
    trained_reg_models[name] = model

# Comparison Table
reg_summary = pd.DataFrame(list(reg_results.items()), columns=['Model', 'R2 Score']).sort_values(by='R2 Score', ascending=False)
print("\nRegression Models Comparison:")
print(reg_summary.to_string(index=False))


'''# ==========================================
# Q4: BEST MODEL SELECTION & SAVING
# ==========================================
print("\n--- Q4: SAVING BEST MODELS ---")

best_clf_name = clf_summary.iloc[0]['Model']
best_reg_name = reg_summary.iloc[0]['Model']

print(f"Best Classification Model: {best_clf_name} (Accuracy: {clf_results[best_clf_name]:.4f})")
print(f"Best Regression Model: {best_reg_name} (R2 Score: {reg_results[best_reg_name]:.4f})")

# Packaging best models and preprocessing metadata
best_clf_payload = {
    "model_name": best_clf_name,
    "model": trained_clf_models[best_clf_name],
    "scaler": clf_scaler,
    "features": list(X_clf.columns),
    "target_names": list(iris.target_names),
    "needs_scaling": clf_models[best_clf_name][1]
}

best_reg_payload = {
    "model_name": best_reg_name,
    "model": trained_reg_models[best_reg_name],
    "scaler": reg_scaler,
    "features": list(X_reg.columns),
    "needs_scaling": reg_models[best_reg_name][1]
}

# Packaging all models so the Streamlit application can use any chosen algorithm
all_models_payload = {
    "classification": {
        "models": trained_clf_models,
        "scaler": clf_scaler,
        "features": list(X_clf.columns),
        "target_names": list(iris.target_names),
        "scaling_reqs": {name: req for name, (_, req) in clf_models.items()}
    },
    "regression": {
        "models": trained_reg_models,
        "scaler": reg_scaler,
        "features": list(X_reg.columns),
        "scaling_reqs": {name: req for name, (_, req) in reg_models.items()}
    }
}

joblib.dump(best_clf_payload, 'best_classification_model.pkl')
joblib.dump(best_reg_payload, 'best_regression_model.pkl')
joblib.dump(all_models_payload, 'all_trained_models.pkl')

print("Models, Scalers, and Feature maps successfully saved via joblib!")'''



