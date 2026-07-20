# =====================================
# Import Libraries
# =====================================

import pandas as pd

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("amazon.csv")

# =====================================
# Display First 5 Rows
# =====================================

print(df.head())

# =====================================
# Dataset Information
# =====================================

print("\nDataset Information:\n")
print(df.info())

# =====================================
# Column Names
# =====================================

print("\nColumn Names:\n")
print(df.columns)

# =====================================
# Dataset Shape
# =====================================

print("\nDataset Shape:")
print(df.shape)

# =====================================
# Data Cleaning
# =====================================

# Remove ₹ and comma from price columns
df["discounted_price"] = (
    df["discounted_price"]
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

df["actual_price"] = (
    df["actual_price"]
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Remove % symbol
df["discount_percentage"] = (
    df["discount_percentage"]
    .str.replace("%", "", regex=False)
    .astype(float)
)

# Convert rating to numeric
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Remove commas from rating_count
df["rating_count"] = (
    df["rating_count"]
    .str.replace(",", "", regex=False)
)

df["rating_count"] = pd.to_numeric(
    df["rating_count"],
    errors="coerce"
)

# Remove missing values
df.dropna(inplace=True)

print("\nAfter Cleaning:")
print(df.head())

print("\nData Types:")
print(df.dtypes)


# =====================================
# Feature Selection
# =====================================

# Select Input Features
X = df[[
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count"
]]

# Target Column
y = df["discounted_price"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFirst 5 Features:")
print(X.head())

print("\nFirst 5 Target Values:")
print(y.head())


# =====================================
# Train-Test Split
# =====================================

from sklearn.model_selection import train_test_split

# Split the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape")
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print("\nTesting Data Shape")
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# =====================================
# Feature Scaling
# =====================================

from sklearn.preprocessing import StandardScaler

# Create StandardScaler object
scaler = StandardScaler()

# Scale Training Data
X_train = scaler.fit_transform(X_train)

# Scale Testing Data
X_test = scaler.transform(X_test)

print("\nScaling Completed Successfully!")


# =====================================
# Model Training
# =====================================

from sklearn.linear_model import LinearRegression

# Create Linear Regression Model
model = LinearRegression()

# Train the Model
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# =====================================
# Prediction and Accuracy
# =====================================

from sklearn.metrics import r2_score

# Make Predictions
y_pred = model.predict(X_test)

# Calculate R2 Score
score = r2_score(y_test, y_pred)

print("\nFirst 5 Predictions:")
print(y_pred[:5])

print("\nR2 Score:", round(score, 4))

# =====================================
# Save Model and Objects
# =====================================

import joblib

# Save Trained Model
joblib.dump(model, "amazon_model.pkl")

# Save Scaler
joblib.dump(scaler, "amazon_scaler.pkl")

# Save Column Names
joblib.dump(X.columns.tolist(), "amazon_columns.pkl")

print("\nModel Saved Successfully!")

print("Files Created:")
print("✔ amazon_model.pkl")
print("✔ amazon_scaler.pkl")
print("✔ amazon_columns.pkl")
