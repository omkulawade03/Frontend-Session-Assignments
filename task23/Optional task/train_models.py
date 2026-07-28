import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, r2_score, mean_squared_error
)

def prepare_breast_cancer_data():
    """Load and preprocess Breast Cancer Wisconsin dataset."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'df': df,
        'feature_names': list(X.columns),
        'target_names': list(data.target_names),
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }

def prepare_diabetes_data():
    """Load and preprocess Diabetes dataset."""
    data = load_diabetes(as_frame=True)
    df = data.frame
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'df': df,
        'feature_names': list(X.columns),
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }

def train_and_eval_classification(data_dict):
    """Train and evaluate all classification models."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(data_dict['X_train_scaled'], data_dict['y_train'])
        
        # Predictions
        y_pred = model.predict(data_dict['X_test_scaled'])
        y_proba = model.predict_proba(data_dict['X_test_scaled'])
        
        # Metrics
        acc = accuracy_score(data_dict['y_test'], y_pred)
        prec = precision_score(data_dict['y_test'], y_pred)
        rec = recall_score(data_dict['y_test'], y_pred)
        f1 = f1_score(data_dict['y_test'], y_pred)
        cm = confusion_matrix(data_dict['y_test'], y_pred)
        report = classification_report(data_dict['y_test'], y_pred, target_names=data_dict['target_names'], output_dict=True)
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm,
            'report': report,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
    return results

def train_and_eval_regression(data_dict):
    """Train and evaluate regression model."""
    model = LinearRegression()
    model.fit(data_dict['X_train_scaled'], data_dict['y_train'])
    
    y_pred = model.predict(data_dict['X_test_scaled'])
    
    r2 = r2_score(data_dict['y_test'], y_pred)
    mse = mean_squared_error(data_dict['y_test'], y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'Linear Regression': {
            'model': model,
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'y_pred': y_pred
        }
    }