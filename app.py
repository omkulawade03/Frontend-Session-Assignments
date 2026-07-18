# ==========================================================
# Q1. Import Required Libraries
# ==========================================================

# Streamlit is used to create the web application
import streamlit as st

# Pandas is used for data manipulation and DataFrame creation
import pandas as pd

# Joblib is used to load the saved machine learning model
import joblib


# ==========================================================
# Q2. Load Trained Model and Preprocessing Objects
# ==========================================================

# Load the trained Linear Regression model
model = joblib.load("LR_model.pkl")
print("Model Loaded Successfully!")

# Load the StandardScaler object
scaler = joblib.load("scaler.pkl")
print("Scaler Loaded Successfully!")

# Load the encoded column names
encoded_columns = joblib.load("columns.pkl")
print("Encoded Columns Loaded Successfully!")



# ==========================================================
# Q3. Configure Streamlit Page
# ==========================================================

# Set the page title and center the layout
st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)


# ==========================================================
# Q4. App Title and Description
# ==========================================================

st.title("Ford Car Price Predictor")

st.write("Enter the car details below to predict its selling price.")



# ==========================================================
# Q5. Numerical Input Fields
# ==========================================================

# Manufacturing Year
year = st.number_input(
    "Manufacturing Year",
    min_value=1996,
    max_value=2025,
    value=2018
)

# Mileage
mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=300000,
    value=30000
)

# Road Tax
tax = st.number_input(
    "Road Tax",
    min_value=0,
    max_value=600,
    value=150
)

# MPG
mpg = st.number_input(
    "MPG",
    min_value=10.0,
    max_value=100.0,
    value=50.0
)

# Engine Size
engine_size = st.number_input(
    "Engine Size",
    min_value=0.8,
    max_value=5.0,
    value=1.5
)




# ==========================================================
# Q6. Categorical Inputs using Dropdown
# ==========================================================

# Dropdown for Transmission
transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual", "Semi-Auto"]
)

# Dropdown for Fuel Type
fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
)



# ==========================================================
# Q7. Text Input and Predict Button
# Take car model name and create Predict Price button.
# ==========================================================

# Car Model Selection
car_model = st.selectbox(
    "Car Model",
    [
        "B-MAX",
        "C-MAX",
        "EcoSport",
        "Edge",
        "Escort",
        "Fiesta",
        "Focus",
        "Fusion",
        "Galaxy",
        "Grand C-MAX",
        "KA",
        "Ka+",
        "Kuga",
        "Mondeo",
        "Mustang",
        "Puma",
        "Ranger",
        "S-MAX",
        "Streetka",
        "Tourneo Connect",
        "Tourneo Custom",
        "Transit Tourneo"
    ]
)

# Predict Button
predict = st.button("Predict Price")



# ==========================================================
# Q8. Create DataFrame and Perform One-Hot Encoding
# ==========================================================

if predict:

    # Create dictionary from user input
    input_data = {
        "model": car_model,
        "year": year,
        "transmission": transmission,
        "mileage": mileage,
        "fuelType": fuel_type,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engine_size
    }

    # Convert dictionary into DataFrame
    input_df = pd.DataFrame([input_data])

    # Display Input Data
    st.subheader("Input Data")
    st.dataframe(input_df)

    # Perform One-Hot Encoding
    input_encoded = pd.get_dummies(input_df)

    # Match columns with training dataset
    input_encoded = input_encoded.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    # Display Encoded Data
    st.subheader("Encoded Input Data")
    st.dataframe(input_encoded)


# ==========================================================
# Q9. Feature Scaling and Prediction
# ==========================================================

    # Numerical columns used during training
    numerical_columns = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize"
    ]

    # Apply StandardScaler
    input_encoded[numerical_columns] = scaler.transform(
        input_encoded[numerical_columns]
    )

    # Predict Selling Price
    prediction = model.predict(input_encoded)

    # Display Prediction
    st.success(f"Predicted Price: £{prediction[0]:,.2f}")




# ==========================================================
# Q10. Mini Project - Complete Streamlit App
# ==========================================================

st.write("---")

st.subheader("Assignment 21")

st.write("Ford Car Price Predictor Web Application")

st.write("✔ Model Loaded")

st.write("✔ User Input Taken")

st.write("✔ Price Predicted Successfully")

st.success("Application Running Successfully!")

