import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import mysql.connector as connection
from sshtunnel import SSHTunnelForwarder

SSH_USERNAME = st.secrets["SSH_USERNAME"]
SSH_PASSWORD = st.secrets["SSH_PASSWORD"]
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]

  # Assuming you're using SQLite, change to your database connection method
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

sales = """
SELECT
    WEEK(o.delivery_date, 1) AS week,
    ps.id,
    ps.name AS product,
    ROUND(AVG(od.unit_price/p.weight)) AS unit_price
FROM orders o
JOIN order_details od ON od.order_id = o.id
JOIN stocks s ON s.id = od.stock_id
JOIN products p ON p.id = s.countable_id
JOIN product_standards ps ON ps.id = p.product_standard_id
WHERE o.delivery_date < '2024-11-01'
GROUP BY ps.id, WEEK(o.delivery_date, 1);
"""

purchases = """
SELECT
    WEEK(p.created_at, 1) AS week,
    ps.id,
    ps.name,
    ROUND(AVG(sm.price)) AS unit_price,
    p.origin
FROM purchases p
JOIN stock_movements sm ON sm.purchase_id = p.id
JOIN product_standards ps ON p.product_standard_id = ps.id
WHERE p.deleted_at IS NULL AND DATE_FORMAT(p.date, '%Y-%m-%d') < '2024-11-01'
GROUP BY ps.id, WEEK(p.created_at, 1), p.origin;
"""

# Fetch data from the database
sales_df = pd.read_sql(sales, mydb)
purchases_df = pd.read_sql(purchases, mydb)

# Streamlit UI - Selectbox for Product
product_options = sales_df['product'].unique()
selected_product = st.selectbox("Select a Product", product_options)

# Get the selected product's id
selected_product_id = sales_df[sales_df['product'] == selected_product]['id'].values[0]

# Filter sales and purchase data for the selected product
sales_df_filtered = sales_df[sales_df['id'] == selected_product_id]
purchases_df_filtered = purchases_df[purchases_df['id'] == selected_product_id]

# Merge sales and purchases data on 'week' and 'id'
merged_df = pd.merge(sales_df_filtered, purchases_df_filtered, on=['week', 'id'], how='left')

# Create the plot for Sales Price (green)
fig = px.line(merged_df,
              x='week',
              y='unit_price_x',  # Sales price
              title=f'Sales vs Purchase Prices for {selected_product} by Week',
              labels={'unit_price_x': 'Sales Price', 'week': 'Week'},
              line_shape='linear',
              color_discrete_sequence=['green'])  # Green line for Sales Price

# Adding the sales price line explicitly to the legend
fig.add_scatter(x=merged_df['week'],
                y=merged_df['unit_price_x'],  # Sales price
                mode='lines',
                name='Sales Price',  # Add to legend
                line=dict(color='green'))

# Plot purchase price curves by origin with custom colors
for origin, color in zip(['Adjamé', 'Bord champ', 'Supermarché'], ['orange', 'gold', 'orangered']):
    origin_data = merged_df[merged_df['origin'] == origin]
    fig.add_scatter(x=origin_data['week'],
                    y=origin_data['unit_price_y'],  # Purchase price
                    mode='lines',
                    name=f'Purchase Price - {origin}',
                    line=dict(dash='dot', color=color))  # Set custom color and dashed line

# Update layout for better visualization
fig.update_layout(title=f'Sales and Purchase Prices for {selected_product} by Week',
                  xaxis_title='Week',
                  yaxis_title='Price (FCFA)',
                  template='plotly_dark')

# Display the plot in Streamlit
st.plotly_chart(fig)
