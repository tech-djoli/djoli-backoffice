import streamlit as st

# Set page config for the entire app
st.sidebar.header("Navigation.")

def admin_sidebar():
    with st.sidebar:
        st.page_link('./pages/delivery_routes.py', label='Delivery Route')
        st.page_link('./pages/customer_map.py', label='Customer Map')
        st.button('logout', key='logout')

admin_sidebar()


st.title("Welcome to the Delivery and Customer Map App")
st.write("Use the sidebar to navigate between the Delivery Routes and Customer Map pages.")
