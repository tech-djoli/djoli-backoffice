import streamlit as st
import importlib

# Set page config must be the first Streamlit command in your script
st.set_page_config(page_title="My Multipage App", page_icon="🌍", layout="wide")

# Create a sidebar for navigation with buttons
st.sidebar.title("Navigation")

# Create buttons for each page
if st.sidebar.button("Delivery Routes"):
    page = "delivery_routes"
elif st.sidebar.button("Customer Map"):
    page = "customer_map"
else:
    page = "delivery_routes"  # Default page if none is selected

# Dynamically import and execute the selected page module
if page == "delivery_routes":
    delivery_routes = importlib.import_module("delivery_routes")
elif page == "customer_map":
    customer_map = importlib.import_module("customer_map")
