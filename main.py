import streamlit as st

st.set_page_config(page_title="My Multipage App", page_icon="🌍", layout="wide")

# Set page config for the entire app
st.sidebar.header("Navigation.")
# Main Page
st.title("Main Page")
st.sidebar.success("Select a page above.")

st.title("Welcome to the Delivery and Customer Map App")
st.write("Use the sidebar to navigate between the Delivery Routes and Customer Map pages.")


