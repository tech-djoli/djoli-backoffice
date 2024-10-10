import pandas as pd
import numpy as np
import folium
import streamlit as st
from streamlit_folium import folium_static


producers = pd.read_csv('data/producers.csv')

def format_phone_number(number):
    if pd.isna(number):
        return np.nan
    number_str = str(int(number))
    if not number_str.startswith('0'):
        number_str = '0' + number_str
    return number_str

producers['contact1'] = producers['contact1'].apply(format_phone_number)
producers = producers.dropna(subset=['Latitude', 'Longitude'])


def create_map():
    m = folium.Map(zoom_start=6)

    coordinate_list = []

    for index, row in producers.iterrows():
        iframe_content = (
            f"<b>Name:</b> {row['last_name']} <br>"
            f"<b>Gender:</b> {row['gender (*)']} <br>"
            f"<b>Category:</b> {row['category']} <br>"
            f"<b>City:</b> {row['city']} <br>"
            f"<b>Location:</b> {row['location']} <br>"
            f"<b>Contact:</b> {row['contact1']} <br>"
        )

        popup = folium.Popup(iframe_content, min_width=150, max_width=300)

        if row['category'] == "cooperative":
            marker_color = 'orange'
        elif row['category'] == "producer":
            marker_color = 'green'
        else:
            marker_color = 'gray'

        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            icon=folium.Icon(color=marker_color, icon='map-marker', prefix='fa'),
            popup=popup
        ).add_to(m)

        coordinate_list.append([row['Latitude'], row['Longitude']])

        m.fit_bounds(coordinate_list)

    return m

map = create_map()
st.title('Djoli - Map of Producers')
st.markdown('')
st.markdown('**Click on a producer to find out more !**')

folium_static(map, width=750)
