import joblib
import streamlit as st
import pandas as pd
import numpy as np

# --- Load Model & Preprocessing ---
model = joblib.load('XGB_house_pred.pkl')
scaler = joblib.load('Scaler.pkl')
columns = joblib.load('columns.pkl')
input_columns = [col for col in columns if col.lower() != 'price']

# --- Page Config ---
st.set_page_config(
    page_title="🏡 House Price Predictor",
    layout="wide",
    page_icon="🏠"
)

# --- Custom CSS for colors & styling ---
st.markdown("""
<style>
body {
    background-color: #f5f5f5;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    height: 3em;
    width: 100%;
    border-radius: 10px;
    font-size: 18px;
}
.stSlider>div>div>div>div>div {
    color: #ff4b4b;
}
.stNumberInput>div>input {
    border-radius: 8px;
    border: 1px solid #ccc;
    padding: 5px;
}
h1, h2, h3, h4, h5, h6 {
    color: #333333;
}
.stExpanderHeader {
    background-color: #ffcccc !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# --- Title & Description ---
st.markdown("<h1 style='text-align:center; color:#ff4b4b;'>🏡 House Price Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Enter the house details below and get an instant prediction.</p>", unsafe_allow_html=True)
st.write("---")

# --- User Inputs in Colored Cards ---
user_data = {}

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🛏️ Bedrooms & Bathrooms")
        user_data['bedrooms'] = st.slider('Bedrooms', 0, 9, 3)
        user_data['bathrooms'] = st.slider('Bathrooms', 0.0, 8.0, 2.0)
        st.markdown("### 📐 Area Details")
        user_data['sqft_living'] = st.number_input('Living Area (sqft)', 370, 13540, 1000)
        user_data['sqft_lot'] = st.number_input('Lot Area (sqft)', 638, 1074218, 10000)
        user_data['floors'] = st.slider("Number of Floors", 1.0, 3.5, 2.0)

    with col2:
        st.markdown("### 🌊 View & Condition")
        user_data['waterfront'] = st.selectbox("Waterfront (0=No, 1=Yes)", [0, 1])
        user_data['view'] = st.selectbox("Has View (0=No, 1=Yes)", [0, 1])
        user_data['condition'] = st.slider('Condition (1-5)', 1, 5, 3)
        user_data['sqft_above'] = st.number_input('Sqft Above Ground', 370, 9410, 2100)
        user_data['sqft_basement'] = st.number_input('Sqft Basement', 0, 4820, 500)

    with col3:
        st.markdown("### 🏗️ Year & Location")
        user_data['yr_built'] = st.number_input('Year Built', 1900, 2023, 2000)
        user_data['yr_renovated'] = st.selectbox('Recently Renovated (0=No, 1=Yes)', [0, 1])
        user_data['city'] = st.selectbox('City Encoded Value', [-2, -1, 0, 1])
        user_data['statezip'] = st.selectbox('Statezip Encoded Value', [-2, -1, 0, 1])
        user_data['sale_months'] = st.slider('Sale Month', 1, 12, 6)

# Additional features in a colorful expander
with st.expander("⚙️ More Features", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        user_data['sqft_per_room'] = st.number_input('Sqft per Room', 100, 2000, 500)
        user_data['bath_per_bed'] = st.number_input('Bathrooms per Bedroom', 0.1, 3.0, 1.0)
    with col2:
        user_data['house_age'] = st.number_input('House Age', 0, 150, 20)
        user_data['renovated'] = st.selectbox('Renovated (0=No, 1=Yes)', [0, 1])
    with col3:
        user_data['living_to_lot_ratio'] = st.number_input('Living to Lot Ratio', 0.0, 2.0, 0.3)
        user_data['total_sqft'] = st.number_input('Total Sqft', 370, 13540, 2000)

# --- Prepare Data ---
try:
    feature_order = scaler.feature_names_in_
except AttributeError:
    feature_order = input_columns  

input_values = [user_data.get(col, 0) for col in feature_order]
input_data = pd.DataFrame([input_values], columns=feature_order)
scaled_input = scaler.transform(input_data)

# --- Predict Button with Gradient ---
if st.button("🔍 Predict Price"):
    prediction_log = model.predict(scaled_input)[0]
    prediction_original = np.expm1(prediction_log)
    usd_to_inr = 88.77
    prediction_inr = prediction_original * usd_to_inr

    st.markdown(
        f"""
        <div style='background: linear-gradient(90deg, #ff4b4b, #ff9999); padding: 20px; border-radius: 15px; text-align:center; color:white; font-size:24px;'>
        🏠 Estimated House Price: <b>${prediction_original:,.2f}</b> (≈ ₹{prediction_inr:,.0f} INR)
        </div>
        """,
        unsafe_allow_html=True
    )
    st.balloons()
