import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained stacking regressor model from the pickle file
with open('stacking_model.pkl', 'rb') as f:
    stack_model = pickle.load(f)

with open('preprocessing_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)  # Load the saved preprocessing pipeline

# Define the feature names that the model expects (update this list to match your training features)
feature_names = ["OverallQual", "TotalArea", "TotalSF", "HouseAge"]

def predict_house_price(features):
    # Create a DataFrame from input features with correct column names
    data = pd.DataFrame([features], columns=feature_names)
    prediction = stack_model.predict(data)
    return prediction[0]

# Streamlit UI
st.title("Real Estate House Price Prediction")
st.write("Enter the features of the house:")

# Input fields for each feature (adjust based on your model's feature set)
OverallQual = st.slider("Overall Quality", min_value=1, max_value=10, value=5)
TotalArea = st.number_input("Total Area (sq ft)", min_value=500, max_value=10000, value=1500)
TotalSF = st.number_input("Total Square Feet", min_value=500, max_value=10000, value=1500)
HouseAge = st.number_input("House Age (years)", min_value=0, max_value=100, value=20)

# Create a dictionary with input features
input_features = {
    "OverallQual": OverallQual,
    "TotalArea": TotalArea,
    "TotalSF": TotalSF,
    "HouseAge": HouseAge
}

if st.button("Predict Price"):
    predicted_price = predict_house_price(input_features)
    st.write(f"The estimated house price is ${predicted_price:,.2f}")
