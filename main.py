import streamlit as st
import importlib

# Set page config must be the first Streamlit command in your script
st.set_page_config(page_title="My Multipage App", page_icon="🌍", layout="wide")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Delivery Routes", "Customer Map"])

# Dynamically import and execute the selected page module
if page == "Delivery Routes":
    delivery_routes = importlib.import_module("delivery_routes")
elif page == "Customer Map":
    customer_map = importlib.import_module("customer_map")
