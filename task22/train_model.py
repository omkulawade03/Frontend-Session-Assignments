# ==========================================
# Import Required Libraries
# ==========================================

import pandas as pd          # For loading and handling dataset
import joblib                # For saving and loading ML model

# Import Machine Learning modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Import evaluation metrics
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ==========================================
# Q1. Data Loading & Preprocessing
# ==========================================

# Load Heart Disease Dataset
df = pd.read_csv("heart.csv")

# Display first 5 rows of dataset
print("First 5 Records:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display dataset shape
print("\nDataset Shape:", df.shape)

# ------------------------------------------
# Separate Features (X) and Target (y)
# ------------------------------------------

# Independent variables (Input Features)
X = df.drop("HeartDisease", axis=1)

# Dependent variable (Target)
y = df["HeartDisease"]

# ------------------------------------------
# Encode Categorical Columns
# ------------------------------------------

# Convert categorical columns into numeric format
# drop_first=True avoids dummy variable trap
X = pd.get_dummies(X, drop_first=True)

# Save column names for future prediction
columns = X.columns

print("\nEncoded Feature Columns:")
print(columns)

# ==========================================
# Q2. Train-Test Split
# ==========================================
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("heart.csv")

# Separate Features and Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Now Split the Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# ==========================================
# Q3. Building Logistic Regression Model
# ==========================================

# Import Logistic Regression algorithm
from sklearn.linear_model import LogisticRegression

# ------------------------------------------
# Create the Logistic Regression model
# ------------------------------------------

# max_iter=1000 increases the maximum number
# of iterations to ensure the model converges.
model = LogisticRegression(max_iter=1000)

# ------------------------------------------
# Train (Fit) the Model
# ------------------------------------------

# Train the model using the training dataset
model.fit(X_train, y_train)

# Display success message
print("\nLogistic Regression Model Trained Successfully!")

# ==========================================
# Q4. Making Predictions
# ==========================================

# Predict values using testing dataset
y_pred = model.predict(X_test)

# Display first 10 actual values
print("\nFirst 10 Actual Values:")
print(y_test.values[:10])

# Display first 10 predicted values
print("\nFirst 10 Predicted Values:")
print(y_pred[:10])


# ==========================================
# Q5. Confusion Matrix
# ==========================================

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

# Extract values from Confusion Matrix
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

# Display Confusion Matrix labels
print("\nTrue Negative (TN):", TN)
print("False Positive (FP):", FP)
print("False Negative (FN):", FN)
print("True Positive (TP):", TP)

# ==========================================
# Q6. Model Evaluation
# ==========================================

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate Precision
precision = precision_score(y_test, y_pred)

# Calculate Recall
recall = recall_score(y_test, y_pred)

# Calculate F1 Score
f1 = f1_score(y_test, y_pred)

# Display Evaluation Metrics
print("\nModel Evaluation")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# Display Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================
# Q7. Save Model
# ==========================================

# Save trained Logistic Regression model
joblib.dump(model, "heart_model.pkl")

# Save feature column names
joblib.dump(columns, "columns.pkl")

# Save scaler
# (No scaler used in this project, so saving None)
joblib.dump(None, "scaler.pkl")

print("\nModel Saved Successfully")


# ==========================================
# Q8. Loading & Testing Saved Model
# ==========================================

# Import required libraries
import pandas as pd
import joblib

# ------------------------------------------
# Load Saved Model and Objects
# ------------------------------------------

# Load the trained model
loaded_model = joblib.load("heart_model.pkl")

# Load the feature column names
loaded_columns = joblib.load("columns.pkl")

# Load the scaler (None in this project)
loaded_scaler = joblib.load("scaler.pkl")

print("Model Loaded Successfully!")

# ------------------------------------------
# Create Sample Input Data
# ------------------------------------------

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

# Convert dictionary to DataFrame
sample_df = pd.DataFrame([sample])

# ------------------------------------------
# Preprocess the Input Data
# ------------------------------------------

# Apply One-Hot Encoding
sample_df = pd.get_dummies(sample_df)

# Match feature columns with training data
sample_df = sample_df.reindex(columns=loaded_columns, fill_value=0)

# Apply scaling if a scaler exists
if loaded_scaler is not None:
    sample_df = loaded_scaler.transform(sample_df)

# ------------------------------------------
# Make Prediction
# ------------------------------------------

prediction = loaded_model.predict(sample_df)

# Display Prediction Result
print("\nPrediction Result:")

if prediction[0] == 1:
    print("❤️ Heart Disease: YES")
else:
    print("💚 Heart Disease: NO")

    