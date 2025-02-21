import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sshtunnel import SSHTunnelForwarder
import mysql.connector as connection
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.graph_objects as go

SSH_USERNAME = st.secrets["SSH_USERNAME"]
SSH_PASSWORD = st.secrets["SSH_PASSWORD"]
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]

tunnel = SSHTunnelForwarder(
    ('31.207.38.195', 22),
    ssh_username=SSH_USERNAME,
    ssh_password=SSH_PASSWORD,
    remote_bind_address=('127.0.0.1', 3306),
    allow_agent=False,
    host_pkey_directories=[]
)

tunnel.start()

mydb = connection.connect(
    host='127.0.0.1',
    database='backend',
    user=DB_USER,
    passwd=DB_PASSWORD,
    port=tunnel.local_bind_port,
    use_pure=True
)

query = """SELECT
    WEEK(o.delivery_date, 1) AS week,
    CEIL(DAYOFMONTH(o.delivery_date) / 7) AS week_of_month,
    ps.id AS standard_id,
    ps.name AS standard_name,
    ROUND(SUM(od.quantity * p.weight)) AS quantity_sold
FROM orders o
JOIN order_details od ON od.order_id = o.id
JOIN stocks s ON s.id = od.stock_id
JOIN products p ON p.id = s.countable_id
JOIN product_standards ps ON ps.id = p.product_standard_id
WHERE WEEK(o.delivery_date, 1) != WEEK(CURDATE(),1)
GROUP BY WEEK(o.delivery_date, 1), ps.id;
"""

df = pd.read_sql(query, mydb)
df = df.dropna()

# Create a dictionary to map standard_name to standard_id
standard_map = dict(zip(df['standard_name'], df['standard_id']))

default_index = list(df['standard_id']).index(1) if 1 in df['standard_id'].values else 0

# Streamlit selectbox to display unique standard names
selected_standard_name = st.selectbox(
    'Selectionner un Produit',
    df['standard_name'].unique(),
    index=default_index
)
selected_standard_id = standard_map[selected_standard_name]

# Streamlit slider for weeks to predict
weeks_prediction = st.slider("Nombre de Semaines à Prédire", 0, 16, 6)

# Train the model function
def train_model(df):
    X = df[['week', 'standard_id', 'week_of_month']]  # Add more features as needed
    y = df['quantity_sold']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

# Generate future predictions
def future_predictions(model, standard_id, weeks_prediction):
    df_filtered = df[df['standard_id'] == standard_id]

    last_week = df_filtered['week'].max()
    last_weekofmonth = df_filtered['week_of_month'].max()

    # Generate next weeks and week_of_month values
    next_weeks = np.arange(last_week + 1, last_week + (weeks_prediction + 1))
    next_weekofmonth = [(last_weekofmonth + i) % 4 + 1 for i in range(1, (weeks_prediction + 1))]

    # Prepare future weeks DataFrame
    future_weeks = pd.DataFrame({
        'week': next_weeks,
        'standard_id': [standard_id] * weeks_prediction,
        'week_of_month': next_weekofmonth
    })

    # Predict future quantities
    future_predictions = model.predict(future_weeks)

    # Prepare historical data
    historical_sales = df_filtered[['week', 'quantity_sold']]

    # Create DataFrame for future predictions
    future_sales = pd.DataFrame({
        'week': next_weeks,
        'quantity_sold': future_predictions
    })

    return historical_sales, future_sales

# Train the model
model = train_model(df)

# Get predictions
historical_sales, future_sales = future_predictions(model, selected_standard_id, weeks_prediction)

# Plot with Plotly
fig = go.Figure()

# Add historical data (solid line)
fig.add_trace(go.Scatter(
    x=historical_sales['week'],
    y=historical_sales['quantity_sold'],
    mode='lines',
    name='Historical Sales',
    line=dict(color='blue')
))

# Add future predictions (dashed line)
fig.add_trace(go.Scatter(
    x=future_sales['week'],
    y=future_sales['quantity_sold'],
    mode='lines',
    name='Predicted Sales',
    line=dict(color='orange', dash='dash')
))

# Customize layout
fig.update_layout(
    title=f'Historical vs Predicted Sales for {selected_standard_name}',
    xaxis_title='Weeks',
    yaxis_title='Quantity Sold',
    legend_title='Legend',
    template='plotly_white'
)

# Show the plot in Streamlit
st.plotly_chart(fig)
