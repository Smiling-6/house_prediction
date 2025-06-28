import streamlit as st
import pandas as pd
import joblib

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load("trained_model.pkl")

model = load_model()

# Page title
st.title("🏠 House Price Predictor (Linear Regression)")

st.markdown("Enter the details of the house below to predict its price.")

# Input features — ensure these match the features used in training
bedrooms = st.number_input("Number of Bedrooms", min_value=0.0, value=3.0)
bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, value=2.0)
sqft_living = st.number_input("Living Area (sqft)", min_value=0, value=1500)
sqft_lot = st.number_input("Lot Area (sqft)", min_value=0, value=4000)
floors = st.number_input("Number of Floors", min_value=0.0, value=1.0)
waterfront = st.selectbox("Waterfront View", options=[0, 1])
view = st.slider("View Rating", 0, 4, 0)
condition = st.slider("Condition Rating", 1, 5, 3)
sqft_above = st.number_input("Above Ground Area (sqft)", min_value=0, value=1500)
sqft_basement = st.number_input("Basment Area (sqft)", min_value=0, value=4000)
yr_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2025, value=0)

# Create DataFrame for prediction
input_df = pd.DataFrame({
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'sqft_living': [sqft_living],
    'sqft_lot': [sqft_lot],
    'floors': [floors],
    'waterfront': [waterfront],
    'view': [view],
    'condition': [condition],
    'sqft_above': [sqft_above],
    'sqft_basement':[sqft_basement],
    'yr_built': [yr_built],
    'yr_renovated': [yr_renovated]
})

# Predict
if st.button("Predict Price"):
    predicted_price = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: ₹{predicted_price:,.0f}")
