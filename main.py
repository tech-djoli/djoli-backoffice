import streamlit as st
import importlib

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Delivery Routes", "Customer Map"])

# Dynamically import the selected module
if page == "Delivery Routes":
    delivery_routes = importlib.import_module("delivery_routes")
elif page == "Customer Map":
    customer_map = importlib.import_module("customer_map")

