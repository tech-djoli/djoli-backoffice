import pandas as pd
import mysql.connector as connection
from sshtunnel import SSHTunnelForwarder
from operator import attrgetter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import re
import pulp
import streamlit as st
from datetime import datetime, date, timedelta
from io import Bytes, StringIO

import base64
import math


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

def closest_past_sunday(selected_date):
    # Find the difference in days between the selected_date and the closest Sunday
    days_since_sunday = selected_date.weekday() + 1  # Monday is 0, Sunday is 6
    # Subtract the days to get the closest past Sunday
    past_sunday = selected_date - timedelta(days=days_since_sunday)
    return past_sunday


current_date = date.today()
selected_date = st.date_input("Date de Prédiction", value=current_date)
past_sunday = closest_past_sunday(selected_date)

query_top_sales = f"""WITH sales_data AS (
    SELECT
        ps.name AS standard_name,
        ps.id AS standard_id,
        ROUND(SUM(od.total_price)) AS product_total,
        ROUND((SUM(od.total_price) / total_sum) * 100, 2) AS product_percentage
    FROM backend.orders bo
    JOIN backend.order_details od ON od.order_id = bo.id
    JOIN backend.shop_translations st ON st.shop_id = bo.shop_id
    JOIN backend.stocks sk ON sk.id = od.stock_id
    JOIN backend.product_translations pt ON pt.product_id = sk.countable_id
    JOIN backend.products p ON p.id = sk.countable_id
    JOIN backend.product_standards ps ON ps.id = p.product_standard_id
    JOIN backend.category_translations ct ON ct.category_id = p.category_id
    CROSS JOIN (
        SELECT ROUND(SUM(od.total_price)) AS total_sum
        FROM backend.orders bo
        JOIN backend.order_details od ON od.order_id = bo.id
        WHERE bo.delivery_date BETWEEN DATE_SUB('{past_sunday}', INTERVAL 3 WEEK) AND '{past_sunday}'
    ) AS totals
    WHERE bo.delivery_date BETWEEN DATE_SUB('{past_sunday}', INTERVAL 3 WEEK) AND '{past_sunday}'
    GROUP BY ps.id, totals.total_sum
), ranked_data AS (
    SELECT
        sd.*,
        SUM(sd.product_total) OVER (ORDER BY sd.product_total DESC) AS running_total,
        SUM(sd.product_percentage) OVER (ORDER BY sd.product_total DESC) AS cumulative_percentage
    FROM sales_data sd
)
SELECT
    standard_name,
    standard_id,
    product_total,
    product_percentage
FROM ranked_data
WHERE cumulative_percentage <= 80
ORDER BY product_percentage DESC;
"""

top_sales = pd.read_sql(query_top_sales,mydb)
standard_ids = top_sales['standard_id'].tolist()
standard_ids_str = ','.join(map(str, standard_ids))

query_top_volumes = f"""
SELECT
    WEEKDAY(bo.delivery_date) AS week_day,
    CEIL(DATEDIFF(bo.delivery_date, '2023-01-01') / 7) AS week,
    ROUND(SUM(od.quantity * p.weight)) AS total_weight,
    ps.id AS standard_id,
    ps.name AS standard_name,
    WEEKDAY(bo.delivery_date) AS day_of_week,
    CONCAT('Semaine du ', DATE_FORMAT(DATE_SUB(bo.delivery_date, INTERVAL WEEKDAY(bo.delivery_date) DAY), '%d/%m/%Y')) AS semaine_du
FROM backend.orders bo
JOIN backend.order_details od ON od.order_id = bo.id
JOIN backend.shop_translations st ON st.shop_id = bo.shop_id
JOIN backend.stocks sk ON sk.id = od.stock_id
JOIN backend.product_translations pt ON pt.product_id = sk.countable_id
JOIN backend.products p ON p.id = sk.countable_id
JOIN backend.product_standards ps ON ps.id = p.product_standard_id
JOIN backend.category_translations ct ON ct.category_id = p.category_id
WHERE bo.delivery_date BETWEEN DATE_SUB('{past_sunday}', INTERVAL 3 WEEK) AND '{past_sunday}'
AND ps.id IN ({standard_ids_str})
GROUP BY WEEKDAY(bo.delivery_date), CEIL(DATEDIFF(bo.delivery_date, '2023-01-01') / 7), ps.id
ORDER BY week_day, week, total_weight DESC
"""

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def generate_top_sales():
    top_sales = pd.read_sql(query_top_sales,mydb)
    standard_ids = top_sales['standard_id'].tolist()
    standard_ids_str = ','.join(map(str, standard_ids))
    volumes_top_sales = pd.read_sql(query_top_volumes, mydb)
    cleaned_volumes = volumes_top_sales.groupby(['standard_id', 'day_of_week'], group_keys=False).apply(remove_outliers, 'total_weight')
    pivot_table = cleaned_volumes.pivot_table(
        index='standard_id',
        columns='day_of_week',
        values='total_weight',
        aggfunc='mean'
    )

    pivot_table = pivot_table.fillna(0)
    pivot_table = pivot_table.round(0).astype(int)
    pivot_table.reset_index(inplace=True)

    return pivot_table


def generate_predictions():

    pivot_table = generate_top_sales()
    # Reset the index without inserting it into the columns
    pivot_table = pivot_table.reset_index(drop=True)

    # Get the numeric representation of days of the week from the columns
    day_order = [col for col in pivot_table.columns if isinstance(col, int)]  # Numeric days (0 for Monday, ..., 6 for Sunday)

    # Create a new DataFrame to store the modified data with the 75% buffer rows
    new_rows = []

    # Iterate over each row of the original pivot_table
    for idx, row in pivot_table.iterrows():
        # Append the original row with 'Historique' type to the new_rows list
        original_row = row.copy()
        original_row['type'] = 'Historique'
        new_rows.append(original_row)

        # Create a new row with the same values but multiplied by 0.75
        new_row = row.copy()
        new_row['type'] = '75%'  # Set the 'type' column to '75%'

        # Multiply each value for the days of the week by 0.75
        for day in day_order:
            # Handle NaN values by replacing them with 0 before multiplication
            value = row[day] if pd.notna(row[day]) else 0
            # Use Python's built-in round() function instead of .round()
            new_row[day] = int(round(value * 0.75, 0))

        # Append the new 75% row to the new_rows list
        new_rows.append(new_row)

    # Create a new DataFrame from the new_rows list
    expanded_pivot_table = pd.DataFrame(new_rows)

    # Drop the unnecessary 'index' columns if they exist
    expanded_pivot_table.drop(columns=['index', 'level_0'], errors='ignore', inplace=True)

    # Reset index to ensure proper formatting
    expanded_pivot_table.reset_index(drop=True, inplace=True)

    # Display only the rows with type '75%'
    expanded_pivot_table = expanded_pivot_table[expanded_pivot_table['type'] == '75%']


    return expanded_pivot_table

def filter_expanded_pivot_table(expanded_pivot_table, selected_date):
    """
    Filter the expanded pivot table for the selected day of the week.

    Args:
        expanded_pivot_table (pd.DataFrame): The expanded pivot table with days as columns.
        selected_date (datetime.date): The selected date to determine the day of the week.

    Returns:
        pd.DataFrame: Filtered DataFrame with 'standard_id' and 'day_value'.
    """
    # Get the day of the week (0=Monday, ..., 6=Sunday)
    day_of_week = selected_date.weekday()

    # Select only the relevant columns: 'standard_id', the numeric day column, and 'type'
    filtered_df = expanded_pivot_table[['standard_id', day_of_week, 'type']]

    # Rename the numeric day column to 'day_value' for clarity
    filtered_df.rename(columns={day_of_week: 'day_value'}, inplace=True)

    return filtered_df

purchases = f"""
SELECT
    order_data.id AS product_id,
    order_data.name AS product_name,
    order_data.order_quantity,
    COALESCE(inventory_data.total_quantity, 0) AS inventory_quantity,
    GREATEST(order_data.order_quantity - COALESCE(inventory_data.total_quantity, 0), 0) AS complement
FROM
    (
        SELECT
            ps.id,
            ps.name,
            SUM(od.quantity * p.weight) AS order_quantity
        FROM orders bo
        JOIN order_details od ON od.order_id = bo.id
        JOIN stocks sk ON sk.id = od.stock_id
        JOIN products p ON p.id = sk.countable_id
        JOIN product_standards ps ON ps.id = p.product_standard_id
        WHERE bo.delivery_date = '{selected_date}' AND bo.status != 'canceled'
        GROUP BY ps.id, ps.name
    ) AS order_data
LEFT JOIN
    (
        SELECT
            ps.id,
            SUM(i.quantity * p.weight) AS total_quantity
        FROM inventories i
        JOIN products p ON i.product_id = p.id
        JOIN product_standards ps ON ps.id = p.product_standard_id
        WHERE DATE_FORMAT(i.`date`, '%Y-%m-%d') = DATE_SUB('{selected_date}', INTERVAL 1 DAY) AND i.deleted_at IS NULL
        GROUP BY ps.id
    ) AS inventory_data
ON order_data.id = inventory_data.id
ORDER BY complement DESC;
"""

# Execute the query and load the results into a DataFrame
purchases_df = pd.read_sql(purchases, mydb)


def process_purchases(purchases_df, filtered_df):
    """
    Process the purchases data by merging with predictions and computing
    complement_predictions and purchase_amount columns.

    Args:
        purchases_df (pd.DataFrame): The main DataFrame with purchase data.
        filtered_df (pd.DataFrame): The filtered DataFrame with predictions.

    Returns:
        pd.DataFrame: The merged and processed DataFrame.
    """
    # Ensure standard_id is properly aligned with product_id in purchases_df
    filtered_df = filtered_df.rename(columns={'standard_id': 'product_id', 'day_value': 'prediction'})

    # Perform a left join to keep all rows from purchases_df
    merged_df = purchases_df.merge(filtered_df[['product_id', 'prediction']], on='product_id', how='left')

    # Fill NaN values in the prediction column with 0
    merged_df['prediction'] = merged_df['prediction'].fillna(0)

    # Convert prediction column to integer for consistency
    merged_df['prediction'] = merged_df['prediction'].astype(int)

    # Step 1: Add the 'complement_predictions' column
    merged_df['complement_predictions'] = merged_df['prediction'] - merged_df['inventory_quantity']

    # Ensure no negative values in 'complement_predictions'
    merged_df['complement_predictions'] = merged_df['complement_predictions'].apply(lambda x: max(x, 0))

    # Step 2: Add the 'purchase_amount' column based on the condition
    merged_df['purchase_amount'] = merged_df.apply(
        lambda row: row['complement'] if row['complement'] >= row['complement_predictions'] else row['complement_predictions'],
        axis=1
    )

    return merged_df


stock_extras = f"""
SELECT
    sub.product_standard_id AS product_id,
    sub.product_standard_name AS product_standard,
    COALESCE(taille.extra_value, '') AS Taille,
    COALESCE(murissement.extra_value, '') AS Mûrissement,
    COALESCE(tiges.extra_value, '') AS Tiges,
    SUM(sub.order_quantity) AS total_order_quantity
FROM (
    SELECT
        od.stock_id,
        ps.id AS product_standard_id,
        ps.name AS product_standard_name,
        ut.title AS unit_title,
        SUM(od.quantity * p.weight) AS order_quantity
    FROM orders bo
    JOIN order_details od ON od.order_id = bo.id
    JOIN stocks sk ON sk.id = od.stock_id
    JOIN products p ON p.id = sk.countable_id
    JOIN product_standards ps ON ps.id = p.product_standard_id
    JOIN units u ON u.id = p.unit_id
    JOIN unit_translations ut ON u.id = ut.unit_id
    WHERE bo.delivery_date = '{selected_date}'
      AND bo.status != 'canceled'
    GROUP BY ps.id, ps.name, ut.title, od.stock_id
) AS sub
LEFT JOIN (
    SELECT
        se.stock_id,
        ev.value AS extra_value
    FROM stock_extras se
    JOIN extra_values ev ON ev.id = se.extra_value_id
    JOIN extra_group_translations egt ON egt.extra_group_id = ev.extra_group_id
    WHERE egt.title = 'Taille'
      AND ev.value IS NOT NULL
      AND ev.value != 'N/C'
      AND egt.deleted_at IS NULL
) AS taille ON taille.stock_id = sub.stock_id
LEFT JOIN (
    SELECT
        se.stock_id,
        ev.value AS extra_value
    FROM stock_extras se
    JOIN extra_values ev ON ev.id = se.extra_value_id
    JOIN extra_group_translations egt ON egt.extra_group_id = ev.extra_group_id
    WHERE egt.title = 'Mûrissement'
      AND ev.value IS NOT NULL
      AND ev.value != 'N/C'
      AND egt.deleted_at IS NULL
) AS murissement ON murissement.stock_id = sub.stock_id
LEFT JOIN (
    SELECT
        se.stock_id,
        ev.value AS extra_value
    FROM stock_extras se
    JOIN extra_values ev ON ev.id = se.extra_value_id
    JOIN extra_group_translations egt ON egt.extra_group_id = ev.extra_group_id
    WHERE egt.title = 'Tiges'
      AND ev.value IS NOT NULL
      AND ev.value != 'N/C'
      AND egt.deleted_at IS NULL
) AS tiges ON tiges.stock_id = sub.stock_id
GROUP BY
    sub.product_standard_id,
    sub.product_standard_name,
    Taille,
    Mûrissement,
    Tiges
HAVING
    Taille != '' OR Mûrissement != '' OR Tiges != ''
ORDER BY
    sub.product_standard_name, Taille, Mûrissement, Tiges;
"""

# Execute the query and load the results into a DataFrame
stock_extras_df = pd.read_sql(stock_extras, mydb)


def generate_purchase_text(merged_df, stock_extras_df, date_purchases):
    """
    Generate the purchase text for the given data.

    Args:
        merged_df (pd.DataFrame): DataFrame containing merged product data with purchase amounts.
        stock_extras_df (pd.DataFrame): DataFrame containing additional stock information.
        date_purchases (str): The purchase date as a string (YYYY-MM-DD).

    Returns:
        str: Formatted text with the purchase list.
    """
    # Initialize the purchase list text with the header
    purchase_text = f"Liste d'achats du {date_purchases}:\n\n"

    # Loop through each product in merged_df
    for _, row in merged_df.iterrows():
        # Add the main product information
        purchase_text += f"{row['product_name']}: {math.ceil(row['purchase_amount'])}\n"

        # Check if the product has extra information in stock_extras_df
        stock_info = stock_extras_df[stock_extras_df['product_id'] == row['product_id']]
        if not stock_info.empty:
            # Loop through each entry for this product in stock_extras_df
            for _, stock_row in stock_info.iterrows():
                purchase_text += (
                    f"  Dont {math.ceil(stock_row['total_order_quantity'])} "
                    f"{stock_row['Taille']} {stock_row['Mûrissement']} {stock_row['Tiges']}\n"
                )

    return purchase_text

expanded_pivot_table = generate_predictions()
filtered_df = filter_expanded_pivot_table(expanded_pivot_table, selected_date)
merged_df = process_purchases(purchases_df, filtered_df)
purchase_text = generate_purchase_text(merged_df, stock_extras_df, selected_date)

    # Display the text in Streamlit
st.text_area("Liste d'Achats", purchase_text, height=400)
text_file = StringIO(purchase_text)
st.download_button(
    label="Télécharger en format texte",
    data=text_file.getvalue(),
    file_name=f"Liste_d_Achats_{date_purchases}.txt",
    mime="text/plain"
    )

    # Add a download button for HTML file
html_content = f"<pre>{purchase_text}</pre>"
st.download_button(
    label="Télécharger en format HTML",
    data=html_content,
    file_name=f"Liste_d_Achats_{date_purchases}.html",
    mime="text/html"
    )
