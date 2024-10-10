import streamlit as st

st.set_page_config(page_title="My Multipage App", page_icon="🌍", layout="wide")

st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to", ["Main Page", "Delivery Routes", "Customer Map", "Producer Map"])

if page == "Main Page":
    st.title("Welcome to the Delivery and Customer Map App")
    st.write("Use the sidebar to navigate between the Delivery Routes and Customer Map pages.")
elif page == "Delivery Routes":
    import pages.deliveryroutes.py as page1
    page1.show_page()
elif page == "Customer Map":
    import pages.customermap.py as page2
    page2.show_page()
elif page == "Producer Map":
    import pages.producermap.py as page3
    page3.show_page()
