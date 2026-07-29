# train_model.py

import warnings
warnings.filterwarnings('ignore')  # Silence non-critical warnings

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

print("="*60)
print("TRAINING PIPELINE: BREAST CANCER DATASET")
print("="*60)

# 1. Load Dataset & Train-Test Split
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Dataset Loaded Successfully ({X.shape[0]} samples, {X.shape[1]} features)\n")

# 2. Hyperparameter Searches (Standard SVC without probability=True)
print("--> Running Hyperparameter Searches...")
knn_manual = KNeighborsClassifier(n_neighbors=5)

svm_grid = GridSearchCV(
    SVC(), 
    {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}, 
    cv=5
)
svm_grid.fit(X_train, y_train)

svm_random = RandomizedSearchCV(
    SVC(), 
    {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}, 
    n_iter=4, cv=5, random_state=42
)
svm_random.fit(X_train, y_train)

# 3. Model Definitions
models = {
    'KNN (Manual k=5)': knn_manual,
    'SVM (GridSearch Best)': svm_grid.best_estimator_,
    'SVM (RandomSearch Best)': svm_random.best_estimator_,
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

# 4. Train and Evaluate
results = []
best_acc = 0.0
best_model_name = ""
best_model_obj = None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append({'Model Architecture': name, 'Test Accuracy': f"{acc * 100:.2f}%"})
    
    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model_obj = model

# Print Summary Table
comparison_df = pd.DataFrame(results)
print("\n" + "="*60)
print("FINAL MODEL COMPARISON TABLE")
print("="*60)
print(comparison_df.to_string(index=False))

# 5. Save the Best Model
model_filename = 'best_model.pkl'
joblib.dump(best_model_obj, model_filename)

print("\n" + "="*60)
print(f"Best Performing Model : {best_model_name}")
print(f"Top Test Accuracy     : {best_acc * 100:.2f}%")
print(f"Saved Successfully To : {model_filename}")
print("="*60)