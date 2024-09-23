import pandas as pd
import mysql.connector as connection
from operator import attrgetter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sshtunnel import SSHTunnelForwarder
import folium
import requests
import json
import streamlit as st
import math
from datetime import datetime, timedelta
import googlemaps

tunnel = SSHTunnelForwarder(
    ('31.207.38.195', 22),  
    ssh_username="root",
    ssh_password="Us9Gacz6rtG",
    remote_bind_address=('127.0.0.1', 3306),
    allow_agent=False,  
    host_pkey_directories=[]  
)

tunnel.start()

mydb = connection.connect(
    host='127.0.0.1',  
    database='backend',
    user='djoli',
    passwd='vmftza%WLfpc#cN@',
    port=tunnel.local_bind_port,  
    use_pure=True
)

zone = st.selectbox(
    "Selectionner une zone?",
    ("Zone 1", "Zone 2", "Zone 3", "Zone 4"),
    index=None,
    placeholder="Zone",
)

today = datetime.now()
date = st.date_input("Selectionner un jour de livraison", today)

query = """SELECT 
    s.id,
    st.title, 
    REPLACE(JSON_UNQUOTE(JSON_EXTRACT(s.location, '$.latitude')), '"', '') AS Latitude,
    REPLACE(JSON_UNQUOTE(JSON_EXTRACT(s.location, '$.longitude')), '"', '') AS Longitude,
    o.total_price,
    o.delivery_date,
    o.delivery_time 
FROM orders o
JOIN shops s ON s.id = o.shop_id
JOIN shop_translations st ON st.shop_id = s.id 
JOIN districts d ON s.district_id = d.id 
WHERE o.delivery_date = '{date}'
AND d.`zone` = '{zone}';
"""

df1 = pd.read_sql(query,mydb)




WAREHOUSE_LAT = 5.332637
WAREHOUSE_LNG = -4.010748
gmaps = googlemaps.Client(key='AIzaSyAtt8vThK0DTYrkYxdd88q8n4SUSwQlH2U')

def preprocess_df(df):
    # Check if the warehouse has already been added
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df = df[(df['Latitude'] != 'null') & (df['Longitude'] != 'null')]
  
    if not ((df['Latitude'] == WAREHOUSE_LAT) & (df['Longitude'] == WAREHOUSE_LNG)).any():
        # Create time windows for each delivery based on delivery_time
        df['delivery_time'] = pd.to_datetime(df['delivery_time'], format='%H:%M')
        df['delivery_end_time'] = df['delivery_time'] + timedelta(minutes=30)
        
        # Add warehouse as the first point (starting point)
        warehouse = {
            'id': 0,
            'title': 'Warehouse',
            'Latitude': WAREHOUSE_LAT,
            'Longitude': WAREHOUSE_LNG,
            'total_price': 0,
            'delivery_time': pd.to_datetime('07:30'),
            'delivery_end_time': pd.to_datetime('11:00')
        }
        
        # Insert warehouse as the first row in the dataframe
        df = pd.concat([pd.DataFrame([warehouse]), df], ignore_index=True)
    
    return df

def get_distance_matrix_in_batches(df, batch_size=10):
    # Create an empty distance and time matrix
    n = len(df)
    distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
    time_matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    origins = [(row['Latitude'], row['Longitude']) for index, row in df.iterrows()]
    
    # Make multiple smaller requests
    for i in range(0, n, batch_size):
        for j in range(0, n, batch_size):
            origin_batch = origins[i:i+batch_size]
            destination_batch = origins[j:j+batch_size]
            
            # Get distance matrix for this batch
            matrix = gmaps.distance_matrix(origin_batch, destination_batch, mode='driving')
            
            # Update distance_matrix and time_matrix
            for idx1, row in enumerate(matrix['rows']):
                for idx2, element in enumerate(row['elements']):
                    distance_matrix[i + idx1][j + idx2] = element['distance']['value']
                    time_matrix[i + idx1][j + idx2] = element['duration']['value']
    
    return distance_matrix, time_matrix

def nearest_neighbor_route(distance_matrix):
    n = len(distance_matrix)
    unvisited = set(range(1, n))  # Start from warehouse (index 0)
    route = [0]  # Start at warehouse
    
    while unvisited:
        last_node = route[-1]
        nearest = min(unvisited, key=lambda x: distance_matrix[last_node][x])
        route.append(nearest)
        unvisited.remove(nearest)
    
    # Return to warehouse
    route.append(0)
    return route

def get_directions(route, df):
    # Initialize an empty list to store all decoded points
    all_points = []

    # Split the route into chunks of 25 waypoints or less
    chunk_size = 23  # 23 waypoints, since origin and destination are included in the limit
    for i in range(0, len(route) - 1, chunk_size):
        # Define the chunk, ensuring the last stop is included
        chunk = route[i:i + chunk_size + 1]

        # Get the coordinates of the chunk in sequence
        waypoints = [(df.loc[j, 'Latitude'], df.loc[j, 'Longitude']) for j in chunk]

        # Request directions from Google Maps for this chunk
        directions = gmaps.directions(
            waypoints[0], waypoints[-1], waypoints=waypoints[1:-1], mode='driving'
        )

        # Check if directions were returned properly
        if directions and 'overview_polyline' in directions[0]:
            polyline = directions[0]['overview_polyline']['points']
            # Decode the polyline and extend the points list
            decoded_points = googlemaps.convert.decode_polyline(polyline)
            all_points.extend(decoded_points)

    # Return the full list of all points
    return all_points

def plot_route(df, points, route):
    # Create a map centered at the warehouse with cartodbpositron background
    m = folium.Map(location=[WAREHOUSE_LAT, WAREHOUSE_LNG], zoom_start=13, tiles="cartodbpositron")
    
    delivery_order = 1  # Counter for numbering deliveries

    # Add markers for each delivery point with popup information
    for stop in route:
        row = df.iloc[stop]
        
        # Popup content with custom CSS for increased width
        popup_content = f'''
            <div style="width: 200px;">  <!-- Set popup width to 200px -->
                <b>{row['title']}</b><br>
                <b>Heure de livraison:</b> {row['delivery_time'].strftime('%H:%M')}<br>
                <b>Total Commande:</b> {int(row['total_price'])} FCFA
            </div>
        '''
        
        # Warehouse (first point) is stop == 0, we don't number it.
        if stop == 0:
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_content, max_width=300),  # Adjust the max width as needed
                icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
            ).add_to(m)
        else:
            # For delivery points, we use the delivery_order counter for numbering with modern styling
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_content, max_width=300),  # Adjust the max width as needed
                icon=folium.DivIcon(html=f'''
                    <div style="
                        display: flex; 
                        justify-content: center; 
                        align-items: center; 
                        font-size: 14pt; 
                        font-weight: bold; 
                        color: white; 
                        background-color: #007bff; 
                        border: 2px solid black; 
                        width: 40px; 
                        height: 40px; 
                        border-radius: 50%; 
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
                        {delivery_order}
                    </div>
                ''')  # Centered bold marker with Flexbox
            ).add_to(m)
            delivery_order += 1  # Increment the delivery counter

    # Plot the route on the map using the decoded points
    if points:
        folium.PolyLine([(p['lat'], p['lng']) for p in points], color='blue', weight=5, opacity=0.8).add_to(m)
    
    return m

def generate_delivery_list(df, route):
    delivery_list = []
    for idx, stop in enumerate(route):
        if stop == 0:  # Skip warehouse
            continue
        
        row = df.iloc[stop]
        
        # Format each delivery's details
        delivery_info = f"""
        **Livraison {idx}:**
        - **Restaurant**: {row['title']}
        - **Heure de Livraison**: {row['delivery_time'].strftime('%H:%M')}
        - **Total Commande**: {int(row['total_price'])} FCFA
        """
        delivery_list.append(delivery_info)
    
    return delivery_list

# Preprocess the data
df1 = preprocess_df(df1)

# Get the distance and time matrix in batches
distance_matrix, time_matrix = get_distance_matrix_in_batches(df1, batch_size=10)

# Optimize the route
route = nearest_neighbor_route(distance_matrix)

# Get driving directions for the entire route (handling more than 25 waypoints)
points = get_directions(route, df1)

# Plot the route on a map with proper delivery numbering
if points:
    # Generate the Folium map
    map_ = plot_route(df1, points, route)
    
    # Display the Folium map in Streamlit using HTML
    st.components.v1.html(map_._repr_html_(), height=500)

    delivery_list = generate_delivery_list(df1, route)
    
    # Display the delivery list below the map
    st.markdown("### Instructions de Livraison")
    for delivery in delivery_list:
        st.markdown(delivery)

else:
    st.write("Could not generate a valid polyline for the route.")

