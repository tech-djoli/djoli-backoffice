import streamlit as st
from delivery_routes import show_delivery_routes
from customer_map import show_customer_map

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Delivery Routes", "Customer Map"])

# Load the selected page
if page == "Delivery Routes":
    show_delivery_routes()
elif page == "Customer Map":
    show_customer_map()
