import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the models
model_path = 'best_model_decision_tree.joblib'
model = joblib.load(model_path)

# Load the scaler
scaler_path = 'scaler.joblib'
scaler = joblib.load(scaler_path)

# Streamlit app
st.title("Stock Price Prediction - Decision Trees")

open_price = st.number_input("Enter the opening price:")
ma50 = st.number_input("Enter the 50-day moving average:")
ma200 = st.number_input("Enter the 200-day moving average:")
rsi = st.number_input("Enter the Relative Strength Index (RSI):")
daily_return = st.number_input("Enter the Daily Return:")
volatility = st.number_input("Enter the Volatility:")

if st.button("Predict"):
    if open_price:
        # Prepare the input features
        sample_features = np.array([[open_price, ma50, ma200, rsi, daily_return, volatility]])

        # Scale the features
        scaled_features = scaler.transform(sample_features)

        # Predict using the model
        predictions = model.predict(scaled_features)

        # Display the predictions
        st.write(f"Predicted High Price: {predictions[0][0]:.2f}")
        st.write(f"Predicted Low Price: {predictions[0][1]:.2f}")
        st.write(f"Predicted Close Price: {predictions[0][2]:.2f}")
        st.write(f"Predicted Adjusted Close Price: {predictions[0][3]:.2f}")
        st.write(f"Predicted Volume: {predictions[0][4]:.2f}")
    else:
        st.write("Please enter a valid opening price.")
