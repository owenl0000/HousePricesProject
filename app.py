import streamlit as st
import pickle
import numpy as np

# Load the trained XGBoost model
model = pickle.load(open('xgb_model.pkl', 'rb'))

# Define a function to make predictions using the model
def predict_house_price(features):
    prediction = model.predict([features])
    return prediction[0]

# Streamlit UI
st.title("Real Estate House Price Prediction")
st.write("Enter the features of the house:")

# Create input fields for house features (adjust based on your model's feature set)
OverallQual = st.slider("Overall Quality", min_value=1, max_value=10)
TotalArea = st.number_input("Total Area (sq ft)", min_value=500, max_value=10000)
TotalSF = st.number_input("Total Square Feet", min_value=500, max_value=10000)
HouseAge = st.number_input("House Age (years)", min_value=0, max_value=100)

# List of features based on input values (make sure this matches your trained model's features)
input_features = [OverallQual, TotalArea, TotalSF, HouseAge]  # Add other features as needed

# Make a prediction when the button is clicked
if st.button("Predict Price"):
    predicted_price = predict_house_price(input_features)
    st.write(f"The estimated house price is ${predicted_price:,.2f}")
