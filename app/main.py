# app/main.py
import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("../model/pest_model.pkl")
crop_encoder = joblib.load("../model/crop_encoder.pkl")
risk_encoder = joblib.load("../model/risk_encoder.pkl")

st.title("🌾 Pest Infestation Risk Predictor")

# UI Inputs
crop = st.selectbox("Select Crop", crop_encoder.classes_)
temperature = st.slider("Temperature (°C)", 10, 50, 30)
humidity = st.slider("Humidity (%)", 10, 100, 60)
rainfall = st.slider("Rainfall (mm)", 0, 500, 100)
past = st.radio("Past Infestation?", ["No", "Yes"])

# Convert inputs
crop_encoded = crop_encoder.transform([crop])[0]
past_val = 1 if past == "Yes" else 0

# Predict
input_data = np.array([[crop_encoded, temperature, humidity, rainfall, past_val]])
pred = model.predict(input_data)[0]
pred_label = risk_encoder.inverse_transform([pred])[0]

st.subheader(f"🛡 Predicted Pest Risk: **{pred_label}**")
