import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def main():
    print("=" * 60)
    print("STEP 1: Load and Prepare Dataset")
    print("=" * 60)
    
    # Generate synthetic health dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=8, 
        n_informative=6, 
        n_redundant=2, 
        random_state=42
    )
    
    feature_names = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Pregnancies']
    df = pd.DataFrame(X, columns=feature_names)
    df['Outcome'] = y
    
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'scaler.pkl')
    print("Dataset split: 800 train samples, 200 test samples.")
    print("Scaler saved as 'scaler.pkl'\n")

    results = {}

    print("=" * 60)
    print("STEP 2: Manual Tuning (KNN & SVM)")
    print("=" * 60)

    # Manual KNN
    knn_neighbors = [3, 5, 7, 11, 13, 15]
    best_knn_acc = 0
    best_k = None
    print("--- Manual KNN Search ---")
    for k in knn_neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test_scaled))
        print(f"k={k:2d} | Test Accuracy: {acc:.4f}")
        if acc > best_knn_acc:
            best_knn_acc = acc
            best_k = k
    print(f"Best KNN: k={best_k} with Accuracy: {best_knn_acc:.4f}\n")

    # Manual SVM
    svm_params = [(1, 'linear'), (1, 'rbf'), (10, 'linear'), (10, 'rbf'), (20, 'linear'), (20, 'rbf')]
    best_manual_svm_acc = 0
    best_manual_svm_param = None
    print("--- Manual SVM Search ---")
    for C, kernel in svm_params:
        svm = SVC(C=C, kernel=kernel, random_state=42)
        svm.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, svm.predict(X_test_scaled))
        print(f"C={C:2d}, Kernel={kernel:6s} | Test Accuracy: {acc:.4f}")
        if acc > best_manual_svm_acc:
            best_manual_svm_acc = acc
            best_manual_svm_param = (C, kernel)
    print(f"Best Manual SVM: C={best_manual_svm_param[0]}, Kernel='{best_manual_svm_param[1]}' | Accuracy: {best_manual_svm_acc:.4f}\n")

    print("=" * 60)
    print("STEP 3: Automated Search (GridSearchCV & RandomizedSearchCV)")
    print("=" * 60)

    # GridSearchCV on SVM
    param_grid_svm = {'C': [0.1, 1, 10, 20], 'kernel': ['linear', 'rbf']}
    grid_svm = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=5, scoring='accuracy')
    grid_svm.fit(X_train_scaled, y_train)
    grid_svm_acc = accuracy_score(y_test, grid_svm.best_estimator_.predict(X_test_scaled))
    results['SVM (GridSearch)'] = (grid_svm_acc, grid_svm.best_estimator_)
    print(f"GridSearchCV SVM Best Params : {grid_svm.best_params_}")
    print(f"GridSearchCV SVM Test Accuracy: {grid_svm_acc:.4f}\n")

    # RandomizedSearchCV on SVM
    rand_svm = RandomizedSearchCV(SVC(random_state=42), param_grid_svm, n_iter=5, cv=5, random_state=42, scoring='accuracy')
    rand_svm.fit(X_train_scaled, y_train)
    rand_svm_acc = accuracy_score(y_test, rand_svm.best_estimator_.predict(X_test_scaled))
    results['SVM (RandomSearch)'] = (rand_svm_acc, rand_svm.best_estimator_)
    print(f"RandomizedSearchCV SVM Best Params : {rand_svm.best_params_}")
    print(f"RandomizedSearchCV SVM Test Accuracy: {rand_svm_acc:.4f}\n")

    print("=" * 60)
    print("STEP 4: Ensemble Models (Random Forest & Boosting)")
    print("=" * 60)

    # Random Forest
    rf_grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]}
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, cv=5, scoring='accuracy')
    grid_rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, grid_rf.best_estimator_.predict(X_test))
    results['Random Forest (Tuned)'] = (rf_acc, grid_rf.best_estimator_)
    print(f"Random Forest Best Params : {grid_rf.best_params_}")
    print(f"Random Forest Test Accuracy: {rf_acc:.4f}\n")

    # AdaBoost
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    ada.fit(X_train, y_train)
    ada_acc = accuracy_score(y_test, ada.predict(X_test))
    results['AdaBoost'] = (ada_acc, ada)
    print(f"AdaBoost Test Accuracy         : {ada_acc:.4f}")

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    results['Gradient Boosting'] = (gb_acc, gb)
    print(f"Gradient Boosting Test Accuracy: {gb_acc:.4f}")

    # XGBoost
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    results['XGBoost'] = (xgb_acc, xgb)
    print(f"XGBoost Test Accuracy           : {xgb_acc:.4f}\n")

    print("=" * 60)
    print("STEP 5: Model Comparison & Final Model Selection")
    print("=" * 60)

    summary_df = pd.DataFrame([
        {'Model': name, 'Test Accuracy': acc} for name, (acc, _) in results.items()
    ]).sort_values(by='Test Accuracy', ascending=False).reset_index(drop=True)

    print(summary_df.to_string(index=False))

    best_model_name = summary_df.iloc[0]['Model']
    best_model_obj = results[best_model_name][1]
    
    print(f"\nTop Model Selected: {best_model_name}")
    joblib.dump(best_model_obj, 'best_diabetes_model.pkl')
    print("Saved best model artifact to 'best_diabetes_model.pkl'")

if __name__ == '__main__':
    main()