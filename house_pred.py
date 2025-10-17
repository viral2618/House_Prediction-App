import joblib
import streamlit as st
import pandas as pd
import numpy as np


model = joblib.load('XGB_house_pred.pkl')
scaler = joblib.load('Scaler.pkl')
columns = joblib.load('columns.pkl') 
input_columns = [col for col in columns if col.lower() != 'price']

st.title("🏡 House Price Prediction Application")
st.markdown("Provide the following details:")


user_data = {}
user_data['bedrooms'] = st.slider('Bedrooms', 0, 9, 3)
user_data['bathrooms'] = st.slider('Bathrooms', 0.0, 8.0, 2.0)
user_data['sqft_living'] = st.number_input('Living Area (sqft)', 370, 13540, 1000)
user_data['sqft_lot'] = st.number_input('Lot Area (sqft)', 638, 1074218, 10000)
user_data['floors'] = st.slider("Number of Floors", 1.0, 3.5, 2.0)
user_data['waterfront'] = st.selectbox("Waterfront (0=No, 1=Yes)", [0, 1])
user_data['view'] = st.selectbox("Has View (0=No, 1=Yes)", [0, 1])
user_data['condition'] = st.slider('Condition (1-5)', 1, 5, 3)
user_data['sqft_above'] = st.number_input('Sqft Above Ground', 370, 9410, 2100)
user_data['sqft_basement'] = st.number_input('Sqft Basement', 0, 4820, 500)
user_data['yr_built'] = st.number_input('Year Built', 1900, 2023, 2000)
user_data['yr_renovated'] = st.selectbox('Recently Renovated (0=No, 1=Yes)', [0, 1])
user_data['city'] = st.selectbox('City Encoded Value', [-2, -1, 0, 1])
user_data['statezip'] = st.selectbox('Statezip Encoded Value', [-2, -1, 0, 1])
user_data['sale_months'] = st.slider('Sale Month', 1, 12, 6)
user_data['sqft_per_room'] = st.number_input('Sqft per Room', 100, 2000, 500)
user_data['bath_per_bed'] = st.number_input('Bathrooms per Bedroom', 0.1, 3.0, 1.0)
user_data['house_age'] = st.number_input('House Age', 0, 150, 20)
user_data['renovated'] = st.selectbox('Renovated (0=No, 1=Yes)', [0, 1])
user_data['living_to_lot_ratio'] = st.number_input('Living to Lot Ratio', 0.0, 2.0, 0.3)
user_data['total_sqft'] = st.number_input('Total Sqft', 370, 13540, 2000)


try:
    feature_order = scaler.feature_names_in_
except AttributeError:
    feature_order = input_columns  
    
input_values = []
for col in feature_order:
    if col in user_data:
        input_values.append(user_data[col])
    else:
        input_values.append(0)  

input_data = pd.DataFrame([input_values], columns=feature_order)


scaled_input = scaler.transform(input_data)

if st.button("🔍 Predict Price"):
    prediction_log = model.predict(scaled_input)[0]   
    prediction_original = np.expm1(prediction_log)    
    
    usd_to_inr = 88.77
    prediction_inr = prediction_original * usd_to_inr
    
    st.success(f"🏠 Estimated House Price: ${prediction_original:,.2f} (≈ ₹{prediction_inr:,.0f} INR)")
