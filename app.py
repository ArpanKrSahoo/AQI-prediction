import streamlit as st
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="AQI Predictor", 
    page_icon="🌍", 
    layout="centered"
)

# 2. Load the Model Safely
@st.cache_resource
def load_model():
    try:
        with open("top_8_aqi_features.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

# 3. App Header
st.title("🌍 AI-Powered AQI Predictor")
st.markdown("Enter the environmental parameters below to predict the current Air Quality Index (AQI).")
st.markdown("---")

# 4. Input Form using Columns
st.subheader("📊 Pollutant Levels & Weather")
col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, format="%.2f")
    no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, format="%.2f")
    co = st.number_input("CO (mg/m³)", min_value=0.0, format="%.2f")
    nh3 = st.number_input("NH3 (µg/m³)", min_value=0.0, format="%.2f")

with col2:
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, format="%.2f")
    so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, format="%.2f")
    o3 = st.number_input("O3 (µg/m³)", min_value=0.0, format="%.2f")
    temp = st.number_input("Temperature (°C)", format="%.2f")

st.markdown("---")

# 5. Prediction Logic
if st.button("Predict AQI 🚀", use_container_width=True):
    if model is None:
        st.error("⚠️ Model file 'top_8_aqi_features.pkl' not found. Please ensure it is in the same directory as this script.")
    else:
        # Construct the feature array
        features = np.array([[pm25, pm10, no2, so2, co, o3, nh3, temp]])

        if hasattr(model, "predict"):
            pred = model.predict(features)[0]
            
            # Determine AQI Category
            if pred <= 50:
                status = "🟢 Good"
            elif pred <= 100:
                status = "🟡 Satisfactory"
            elif pred <= 200:
                status = "🟠 Moderate"
            elif pred <= 300:
                status = "🔴 Poor"
            elif pred <= 400:
                status = "🟣 Very Poor"
            else:
                status = "🟤 Severe"

            # Display Results
            st.success("Prediction Complete!")
            
            # Using st.metric for a cleaner dashboard look
            st.metric(label="Predicted Air Quality Index", value=f"{pred:.1f}", delta=status, delta_color="off")
            
        else:
            st.error("Loaded file is not a trained scikit-learn model.")