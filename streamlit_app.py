import streamlit as st

# Title
st.title("🏠 Simple House Price Calculator")

# Input fields
area = st.number_input("Area (in sq.ft)", min_value=100, max_value=10000, value=1000)
bedrooms = st.slider("Number of Bedrooms", 1, 10, 2)
bathrooms = st.slider("Number of Bathrooms", 1, 10, 2)
parking = st.selectbox("Number of Parking Lots", [0, 1, 2, 3])

# Calculation
if st.button("Calculate Price"):
    price = area * 7000 + bedrooms * 1000 + bathrooms * 1000 + parking * 1500
    st.success(f"💰 Estimated Price: ₹ {price:,.2f}")

