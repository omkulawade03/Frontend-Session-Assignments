# ==========================================================
# Ford Car Price Predictor
# train_model.py
# ==========================================================

# ==========================================================
# Q1. Import Required Libraries
# ==========================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ==========================================================
# Q2. Load Dataset
# ==========================================================

df = pd.read_csv("ford_car_dataset.csv")

print(df.head())

print("\nDataset Information:\n")
print(df.info())

print("\nColumn Names:\n")
print(df.columns)

print("\nDataset Shape:")
print(df.shape)


# ==========================================================
# Q3. One-Hot Encoding
# ==========================================================

df = pd.get_dummies(
    df,
    columns=["model", "transmission", "fuelType"]
)

print("\nAfter One-Hot Encoding:")
print(df.head())

print("\nNew Shape:")
print(df.shape)


# ==========================================================
# Q4. Feature Selection
# ==========================================================

X = df.drop("price", axis=1)

y = df["price"]

print("\nFeatures Shape :", X.shape)
print("Target Shape :", y.shape)

print("\nFirst 5 Target Values:")
print(y.head())


# ==========================================================
# Q5. Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42

)

print("\nTraining Data Shape")
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print("\nTesting Data Shape")
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ==========================================================
# Q6. Feature Scaling
# ==========================================================

scaler = StandardScaler()

numerical_columns = [
    "year",
    "mileage",
    "tax",
    "mpg",
    "engineSize"
]

X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)

print("\nScaling Completed Successfully!")


# ==========================================================
# Q7. Train Linear Regression Model
# ==========================================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")


# ==========================================================
# Q8. Prediction
# ==========================================================

y_pred = model.predict(X_test)

print("\nFirst 5 Predictions:")

print(y_pred[:5])


# ==========================================================
# Q9. Model Evaluation (R2 Score)
# ==========================================================

r2 = r2_score(y_test, y_pred)

print("\nR2 Score :", round(r2, 4))


# ==========================================================
# Q10. Save Model Files
# ==========================================================

joblib.dump(model, "LR_model.pkl")

joblib.dump(scaler, "scaler.pkl")

joblib.dump(X.columns.tolist(), "columns.pkl")

print("\nFiles Saved Successfully!")

print("✔ LR_model.pkl")

print("✔ scaler.pkl")

print("✔ columns.pkl")