import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# Set page config for the entire app
st.sidebar.header("Navigation.")

st.title("Welcome to the Delivery and Customer Map App")
st.write("Use the sidebar to navigate between the Delivery Routes and Customer Map pages.")

switch_page("Delivery Routes")


