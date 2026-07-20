# ==========================================================
# Q1. Setup and Libraries
# Import the required libraries.
# ==========================================================

import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# Q2. Loading Model and Preprocessing Objects
# Load the trained model, scaler and encoded columns.
# ==========================================================

# Load Trained Machine Learning Model
model = joblib.load("amazon_model.pkl")
print("Model Loaded Successfully!")

# Load Standard Scaler
scaler = joblib.load("amazon_scaler.pkl")
print("Scaler Loaded Successfully!")

# Load Encoded Column Names
columns = joblib.load("amazon_columns.pkl")
print("Encoded Columns Loaded Successfully!")



# ==========================================================
# Q3. Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Amazon Product Price Predictor",
    page_icon="🛒",
    layout="centered"
)


# ==========================================================
# Q4. Title and Description
# ==========================================================

st.title("Amazon Product Price Predictor")

st.write("Enter the product details below to predict the discounted price.")

st.write("---")



# ==========================================================
# Q5. Numerical Input Fields
# ==========================================================

actual_price = st.number_input(
    "Actual Price (₹)",
    min_value=1.0,
    value=1000.0
)

discount_percentage = st.number_input(
    "Discount Percentage (%)",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

rating = st.number_input(
    "Rating",
    min_value=0.0,
    max_value=5.0,
    value=4.0
)

rating_count = st.number_input(
    "Rating Count",
    min_value=0,
    value=100
)

# ==========================================================
# Q6. Categorical Input using Dropdowns
# ==========================================================

category = st.selectbox(
    "Product Category",
    [
        "Electronics",
        "Home & Kitchen",
        "Computers & Accessories",
        "Musical Instruments",
        "Office Products",
        "Health & Personal Care",
        "Home Improvement",
        "Car & Motorbike"
    ]
)

brand = st.selectbox(
    "Brand",
    [
        "Amazon Basics",
        "boAt",
        "Samsung",
        "Apple",
        "OnePlus",
        "HP",
        "Dell",
        "Sony",
        "JBL",
        "Logitech",
        "Realme",
        "Mi"
    ]
)

product_type = st.selectbox(
    "Product Type",
    [
        "Mobile Phone",
        "Laptop",
        "Headphones",
        "Smart Watch",
        "Keyboard",
        "Mouse",
        "Power Bank",
        "USB Cable",
        "Bluetooth Speaker",
        "Monitor",
        "Printer",
        "Charger"
    ]
)

# ==========================================================
# Q7. Predict Button
# ==========================================================

predict = st.button("Predict Price")


# ==========================================================
# Q8. Creating Input DataFrame
# ==========================================================

if predict:

    st.subheader("Product Details")

    display_df = pd.DataFrame({
        "Category": [category],
        "Brand": [brand],
        "Product Type": [product_type],
        "Actual Price": [actual_price],
        "Discount %": [discount_percentage],
        "Rating": [rating],
        "Rating Count": [rating_count]
    })

    st.dataframe(display_df)

    # DataFrame used for prediction
    input_df = pd.DataFrame({
        "actual_price": [actual_price],
        "discount_percentage": [discount_percentage],
        "rating": [rating],
        "rating_count": [rating_count]
    })

# ==========================================================
# Q9. Feature Scaling and Prediction
# ==========================================================

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted Discounted Price: ₹{prediction[0]:,.2f}"
    )
