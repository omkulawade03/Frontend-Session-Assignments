import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ==========================
# Q1. Load Dataset
# ==========================

df = pd.read_csv("heart.csv")

print("First 5 Records:")
print(df.head())

# Features and Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Save column names
columns = X.columns

print("\nDataset Shape:", df.shape)

# ==========================
# Q2. Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain-Test Split")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# ==========================
# Q3. Logistic Regression
# ==========================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# ==========================
# Q4. Prediction
# ==========================

y_pred = model.predict(X_test)

print("\nActual Values")
print(y_test.values[:10])

print("\nPredicted Values")
print(y_pred[:10])

# ==========================
# Q5. Confusion Matrix
# ==========================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("\nTrue Negative :", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive :", TP)

# ==========================
# Q6. Evaluation
# ==========================

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================
# Q7. Save Model
# ==========================

joblib.dump(model, "heart_model.pkl")
joblib.dump(columns, "columns.pkl")

# No scaler used
joblib.dump(None, "scaler.pkl")

print("\nModel Saved Successfully")

# ==========================
# Q8. Load Saved Model
# ==========================

loaded_model = joblib.load("heart_model.pkl")
loaded_columns = joblib.load("columns.pkl")

sample = {
    "Age": 45,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 130,
    "Cholesterol": 230,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 150,
    "ExerciseAngina": "N",
    "Oldpeak": 1.2,
    "ST_Slope": "Up"
}

sample_df = pd.DataFrame([sample])

sample_df = pd.get_dummies(sample_df)

sample_df = sample_df.reindex(columns=loaded_columns, fill_value=0)

prediction = loaded_model.predict(sample_df)

print("\nSample Prediction")

if prediction[0] == 1:
    print("Heart Disease: YES")
else:
    print("Heart Disease: NO")