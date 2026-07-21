# ==========================================================
# Breast Cancer Prediction
# train_model.py
# ==========================================================

# Import Required Libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ==========================================================
# Q1. Dataset Loading
# ==========================================================

print("========== Q1. Dataset Loading ==========")

df = pd.read_csv("data.csv")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

# ==========================================================
# Q2. Data Cleaning
# ==========================================================

print("\n========== Q2. Data Cleaning ==========")

print("\nMissing Values:")
print(df.isnull().sum())

# Remove unnecessary columns
df.drop(columns=["id", "Unnamed: 32"], errors="ignore", inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove missing values
df.dropna(inplace=True)

print("\nCleaned Dataset Shape:")
print(df.shape)

# ==========================================================
# Q3 & Q4. Feature Selection and Preprocessing
# ==========================================================

print("\n========== Q3 & Q4 ==========")

# Features and Target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Encode target
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Scale Features
scaler = StandardScaler()

X = scaler.fit_transform(X)

print("Preprocessing Completed Successfully.")

# ==========================================================
# Q5. Train-Test Split
# ==========================================================

print("\n========== Q5. Train Test Split ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# ==========================================================
# Q6. Model Building
# ==========================================================

print("\n========== Q6. Model Building ==========")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model Trained Successfully.")

# ==========================================================
# Q7. Prediction
# ==========================================================

print("\n========== Q7. Prediction ==========")

y_pred = model.predict(X_test)

print("\nFirst 10 Actual Values")
print(y_test[:10])

print("\nFirst 10 Predicted Values")
print(y_pred[:10])

# ==========================================================
# Q8. Model Evaluation
# ==========================================================

print("\n========== Q8. Model Evaluation ==========")

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================================
# Q9. Save Model
# ==========================================================

print("\n========== Q9. Save Model ==========")

joblib.dump(model, "breast_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(df.drop("diagnosis", axis=1).columns.tolist(), "columns.pkl")

print("Model Saved Successfully.")

# ==========================================================
# Q10. Load Saved Model
# ==========================================================

print("\n========== Q10. Load Saved Model ==========")

loaded_model = joblib.load("breast_cancer_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
loaded_columns = joblib.load("columns.pkl")

sample = {
    "radius_mean":17.99,
    "texture_mean":10.38,
    "perimeter_mean":122.8,
    "area_mean":1001.0,
    "smoothness_mean":0.1184,
    "compactness_mean":0.2776,
    "concavity_mean":0.3001,
    "concave points_mean":0.1471,
    "symmetry_mean":0.2419,
    "fractal_dimension_mean":0.07871,
    "radius_se":1.095,
    "texture_se":0.9053,
    "perimeter_se":8.589,
    "area_se":153.4,
    "smoothness_se":0.006399,
    "compactness_se":0.04904,
    "concavity_se":0.05373,
    "concave points_se":0.01587,
    "symmetry_se":0.03003,
    "fractal_dimension_se":0.006193,
    "radius_worst":25.38,
    "texture_worst":17.33,
    "perimeter_worst":184.6,
    "area_worst":2019.0,
    "smoothness_worst":0.1622,
    "compactness_worst":0.6656,
    "concavity_worst":0.7119,
    "concave points_worst":0.2654,
    "symmetry_worst":0.4601,
    "fractal_dimension_worst":0.1189
}

sample_df = pd.DataFrame([sample])

sample_df = sample_df[loaded_columns]

sample_scaled = loaded_scaler.transform(sample_df)

prediction = loaded_model.predict(sample_scaled)

print("\nPrediction Result")

if prediction[0] == 1:
    print("Tumor Prediction : Malignant")
else:
    print("Tumor Prediction : Benign")