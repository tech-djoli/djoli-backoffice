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
from datetime import date
from io import BytesIO
import base64
from dotenv import load_dotenv
import os

load_dotenv()

SSH_USERNAME = os.getenv("SSH_USERNAME")
SSH_PASSWORD = os.getenv("SSH_PASSWORD")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

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

current_date = date.today()
selected_date = st.date_input("Date de Prédiction", value=current_date)

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
        WHERE bo.delivery_date BETWEEN DATE_SUB('{selected_date}', INTERVAL 3 WEEK) AND '{selected_date}'
    ) AS totals
    WHERE bo.delivery_date BETWEEN DATE_SUB('{selected_date}', INTERVAL 3 WEEK) AND '{selected_date}'
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
    CASE
        WHEN WEEKDAY(bo.delivery_date) = 0 THEN 'Lundi'
        WHEN WEEKDAY(bo.delivery_date) = 1 THEN 'Mardi'
        WHEN WEEKDAY(bo.delivery_date) = 2 THEN 'Mercredi'
        WHEN WEEKDAY(bo.delivery_date) = 3 THEN 'Jeudi'
        WHEN WEEKDAY(bo.delivery_date) = 4 THEN 'Vendredi'
        WHEN WEEKDAY(bo.delivery_date) = 5 THEN 'Samedi'
        WHEN WEEKDAY(bo.delivery_date) = 6 THEN 'Dimanche'
        ELSE NULL
    END AS day_of_week,
    CONCAT('Semaine du ', DATE_FORMAT(DATE_SUB(bo.delivery_date, INTERVAL WEEKDAY(bo.delivery_date) DAY), '%d/%m/%Y')) AS semaine_du
FROM backend.orders bo
JOIN backend.order_details od ON od.order_id = bo.id
JOIN backend.shop_translations st ON st.shop_id = bo.shop_id
JOIN backend.stocks sk ON sk.id = od.stock_id
JOIN backend.product_translations pt ON pt.product_id = sk.countable_id
JOIN backend.products p ON p.id = sk.countable_id
JOIN backend.product_standards ps ON ps.id = p.product_standard_id
JOIN backend.category_translations ct ON ct.category_id = p.category_id
WHERE bo.delivery_date BETWEEN DATE_SUB('{selected_date}', INTERVAL 3 WEEK) AND '{selected_date}'
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
        index='standard_name',
        columns='day_of_week',
        values='total_weight',
        aggfunc='mean'
    )
    day_order = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    pivot_table = pivot_table[day_order]
    pivot_table = pivot_table.fillna(0)
    pivot_table = pivot_table.round(0).astype(int)
    pivot_table.reset_index(inplace=True)

    return pivot_table

def generate_predictions():
    pivot_table = generate_top_sales()

    new_rows = []
    day_order = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

    for idx, row in pivot_table.iterrows():
        original_row = row.copy()
        original_row['type'] = 'Historique'
        new_rows.append(original_row)
        new_row = row.copy()
        new_row['type'] = '75%'

        for day in day_order:
            value = row[day] if pd.notna(row[day]) else 0
            new_row[day] = int(round(value * 0.75, 0))
        new_rows.append(new_row)

    expanded_pivot_table = pd.DataFrame(new_rows)
    columns_order = ['standard_name', 'type'] + day_order
    expanded_pivot_table = expanded_pivot_table[columns_order]
    expanded_pivot_table.reset_index(drop=True, inplace=True)

    return expanded_pivot_table


expanded_pivot_table = generate_predictions()
with st.expander("Voir Tableau de Prévisions"):
    st.table(expanded_pivot_table)

st.download_button(
    label="Télécharger les prévisions",
    data=expanded_pivot_table.to_csv().encode('utf-8'),
    file_name='predictions.csv',
    mime='text/csv')

comparison_date = st.date_input("Date de Comparaison.", value=current_date)


query_historicals = f"""
SELECT
    WEEKDAY(bo.delivery_date) AS week_day,
    CEIL(DATEDIFF(bo.delivery_date, '2023-01-01') / 7) AS week,
    ROUND(SUM(od.quantity * p.weight)) AS total_weight,
    ps.id AS standard_id,
    ps.name AS standard_name,
    CASE
        WHEN WEEKDAY(bo.delivery_date) = 0 THEN 'Lundi'
        WHEN WEEKDAY(bo.delivery_date) = 1 THEN 'Mardi'
        WHEN WEEKDAY(bo.delivery_date) = 2 THEN 'Mercredi'
        WHEN WEEKDAY(bo.delivery_date) = 3 THEN 'Jeudi'
        WHEN WEEKDAY(bo.delivery_date) = 4 THEN 'Vendredi'
        WHEN WEEKDAY(bo.delivery_date) = 5 THEN 'Samedi'
        WHEN WEEKDAY(bo.delivery_date) = 6 THEN 'Dimanche'
        ELSE NULL
    END AS day_of_week,
    'Ventes' AS type
FROM backend.orders bo
JOIN backend.order_details od ON od.order_id = bo.id
JOIN backend.shop_translations st ON st.shop_id = bo.shop_id
JOIN backend.stocks sk ON sk.id = od.stock_id
JOIN backend.products p ON p.id = sk.countable_id
JOIN backend.product_standards ps ON ps.id = p.product_standard_id
JOIN backend.category_translations ct ON ct.category_id = p.category_id
WHERE WEEK(bo.delivery_date, 1) = WEEK('{comparison_date}',1)
AND ps.id IN ({standard_ids_str})
GROUP BY WEEKDAY(bo.delivery_date), CEIL(DATEDIFF(bo.delivery_date, '2023-01-01') / 7), ps.id

UNION ALL

SELECT
    WEEKDAY(sm.date) AS week_day,
    CEIL(DATEDIFF(sm.date, '2023-01-01') / 7) AS week,
    SUM(sm.quantity) AS total_weight,
    ps.id AS standard_id,
    ps.name AS standard_name,
    CASE
        WHEN WEEKDAY(sm.date) = 0 THEN 'Lundi'
        WHEN WEEKDAY(sm.date) = 1 THEN 'Mardi'
        WHEN WEEKDAY(sm.date) = 2 THEN 'Mercredi'
        WHEN WEEKDAY(sm.date) = 3 THEN 'Jeudi'
        WHEN WEEKDAY(sm.date) = 4 THEN 'Vendredi'
        WHEN WEEKDAY(sm.date) = 5 THEN 'Samedi'
        WHEN WEEKDAY(sm.date) = 6 THEN 'Dimanche'
        ELSE NULL
    END AS day_of_week,
    'Pertes' AS type
FROM stock_movements sm
JOIN products p ON sm.product_id = p.id
JOIN product_standards ps ON ps.id  = p.product_standard_id
WHERE WEEK(sm.date, 1) = WEEK('{comparison_date}',1)
AND sm.type = 'loss'
AND sm.deleted_at IS NULL
AND ps.id IN ({standard_ids_str})
GROUP BY WEEKDAY(sm.date), CEIL(DATEDIFF(sm.date, '2023-01-01') / 7), ps.id

UNION ALL

SELECT
    WEEKDAY(p.created_at) AS week_day,
    CEIL(DATEDIFF(p.created_at, '2023-01-01') / 7) AS week,
    ROUND(SUM(sm.quantity)) AS total_weight,
    ps.id AS standard_id,
    ps.name AS standard_name,
    CASE
        WHEN WEEKDAY(p.created_at) = 0 THEN 'Lundi'
        WHEN WEEKDAY(p.created_at) = 1 THEN 'Mardi'
        WHEN WEEKDAY(p.created_at) = 2 THEN 'Mercredi'
        WHEN WEEKDAY(p.created_at) = 3 THEN 'Jeudi'
        WHEN WEEKDAY(p.created_at) = 4 THEN 'Vendredi'
        WHEN WEEKDAY(p.created_at) = 5 THEN 'Samedi'
        WHEN WEEKDAY(p.created_at) = 6 THEN 'Dimanche'
        ELSE NULL
    END AS day_of_week,
    'Achats' AS type
FROM backend.stock_movements sm
JOIN purchases p ON p.id = sm.purchase_id
JOIN purchase_details pd ON pd.purchase_id = p.id
JOIN backend.product_standards ps ON ps.id = pd.purchaseable_id
WHERE WEEK(p.created_at, 1) = WEEK('{comparison_date}',1)
AND sm.type = 'purchase'
AND sm.deleted_at IS NULL
AND ps.id IN ({standard_ids_str})
GROUP BY WEEKDAY(p.created_at), CEIL(DATEDIFF(p.created_at, '2023-01-01') / 7), ps.id


UNION ALL

SELECT
    WEEKDAY(o.delivery_date) AS week_day,
    CEIL(DATEDIFF(o.delivery_date, '2023-01-01') / 7) AS week,
    ROUND(SUM(od.quantity * COALESCE(p.weight, 1))) AS total_weight,
    ps.id AS standard_id,
    ps.name AS standard_name,
    CASE
        WHEN WEEKDAY(o.delivery_date) = 0 THEN 'Lundi'
        WHEN WEEKDAY(o.delivery_date) = 1 THEN 'Mardi'
        WHEN WEEKDAY(o.delivery_date) = 2 THEN 'Mercredi'
        WHEN WEEKDAY(o.delivery_date) = 3 THEN 'Jeudi'
        WHEN WEEKDAY(o.delivery_date) = 4 THEN 'Vendredi'
        WHEN WEEKDAY(o.delivery_date) = 5 THEN 'Samedi'
        WHEN WEEKDAY(o.delivery_date) = 6 THEN 'Dimanche'
        ELSE NULL
    END AS day_of_week,
    'Rupture' AS type
FROM orders o
JOIN order_details od ON od.order_id = o.id
JOIN stocks s ON s.id = od.stock_id
JOIN products p ON p.id = s.countable_id
JOIN product_standards ps ON p.product_standard_id = ps.id
WHERE od.reason_delete = 'rupture'
AND WEEK(o.delivery_date, 1) = WEEK('{comparison_date}',1)
AND ps.id IN ({standard_ids_str})
GROUP BY WEEKDAY(o.delivery_date), CEIL(DATEDIFF(o.delivery_date, '2023-01-01') / 7), ps.id

UNION ALL

SELECT
    WEEKDAY(i.created_at) AS week_day,
    CEIL(DATEDIFF(i.created_at, '2023-01-01') / 7) AS week,
    ROUND(SUM(i.quantity)) AS total_weight,
    ps.id AS standard_id,
    ps.name AS standard_name,
    CASE
        WHEN WEEKDAY(i.created_at) = 0 THEN 'Lundi'
        WHEN WEEKDAY(i.created_at) = 1 THEN 'Mardi'
        WHEN WEEKDAY(i.created_at) = 2 THEN 'Mercredi'
        WHEN WEEKDAY(i.created_at) = 3 THEN 'Jeudi'
        WHEN WEEKDAY(i.created_at) = 4 THEN 'Vendredi'
        WHEN WEEKDAY(i.created_at) = 5 THEN 'Samedi'
        WHEN WEEKDAY(i.created_at) = 6 THEN 'Dimanche'
        ELSE NULL
    END AS day_of_week,
    'Stock' AS type
FROM inventories i
JOIN products p ON i.product_id = p.id
JOIN product_standards ps ON ps.id = p.product_standard_id
WHERE WEEK(DATE_FORMAT(i.created_at, '%Y-%m-%d'), 1) = WEEK('{comparison_date}',1)
AND ps.id IN ({standard_ids_str})
GROUP BY WEEKDAY(i.created_at), CEIL(DATEDIFF(i.created_at, '2023-01-01') / 7), ps.id;

"""


def generate_comparison():
    realised_volumes = pd.read_sql(query_historicals, mydb)
    pivot_table_realised = realised_volumes.pivot_table(
        index=['standard_name', 'type'],   # Keep both 'standard_name' and 'type' as index
        columns='day_of_week',             # Set 'day_of_week' as the columns (Lundi, Mardi, etc.)
        values='total_weight',             # Values are the 'total_weight'
        aggfunc='sum'                      # In case of duplicates, sum the values
    )

    day_order = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    pivot_table_realised = pivot_table_realised.reindex(columns=day_order, fill_value=0)
    pivot_table_realised = pivot_table_realised.fillna(0)
    pivot_table_realised = pivot_table_realised.round(0).astype(int)

    pivot_table_realised.reset_index(inplace=True)


    merged_table = pd.concat([expanded_pivot_table, pivot_table_realised])
    merged_table = merged_table.sort_values(by=['standard_name', 'type'])
    day_order = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    columns_order = ['standard_name', 'type'] + day_order
    merged_table = merged_table[columns_order]
    merged_table.reset_index(drop=True, inplace=True)


    return merged_table

merged_table = generate_comparison()

with st.expander("Voir Tableau de Comparaisons"):
    st.table(merged_table)

st.download_button(
    label="Télécharger la performance.",
    data=merged_table.to_csv().encode('utf-8'),
    file_name='comparaison.csv',
    mime='text/csv')


import matplotlib.pyplot as plt
import base64
from io import BytesIO

def plot_standard_name_report(df, standard_name):
    """
    Function to plot the values of different types for a given standard_name
    with fixed colors and days of the week on the x-axis.

    :param df: The DataFrame containing the data
    :param standard_name: The name of the standard to plot
    :return: Matplotlib figure object
    """
    # Filter the data for the given standard_name
    df_filtered = df[df['standard_name'] == standard_name]

    # Define color codes for each type
    color_mapping = {
        'Ventes': '#008000',      # Green
        '75%': '#0000FF',         # Blue
        'Achats': '#FFA500',      # Orange
        'Historique': '#8B4513',  # Brown
        'Stock': '#800080',       # Purple
        'Rupture': '#FF0000'      # Red
    }

    # Days of the week in the correct order
    days_of_week = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Iterate through each 'type' and plot a line with the specific color
    for t in df_filtered['type'].unique():
        df_type = df_filtered[df_filtered['type'] == t]
        plt.plot(
            days_of_week,
            df_type[days_of_week].values.flatten(),
            label=t,
            marker='o',
            color=color_mapping.get(t, '#000000')  # Default to black if type is not found
        )

    # Add title and labels
    plt.title(f'Performances {standard_name}')
    plt.xlabel('Jour Semaine')
    plt.ylabel('Poids Total')
    plt.legend(title='Type')
    plt.grid(True)

    return plt.gcf()


def save_plots_as_html(df):
    """
    Function to save plots as HTML with embedded images.

    :param df: The DataFrame containing the data
    :return: HTML content with embedded plots
    """
    html_content = "<html><body>\n"

    # Generate and save each plot for each standard_name in HTML format
    for name in df['standard_name'].unique():
        fig = plot_standard_name_report(df, name)

        # Save figure to buffer
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Embed plot in HTML
        html_content += f"<h2>{name}</h2>\n"
        html_content += f'<img src="data:image/png;base64,{img_str}"/>\n'

    html_content += "</body></html>"

    return html_content


def plot_standard_name_report_interactive(df, standard_name):
    """
    Function to create an interactive plot of values for different types for a given standard_name
    with fixed colors and days of the week on the x-axis using Plotly.

    :param df: The DataFrame containing the data
    :param standard_name: The name of the standard to plot
    :return: Plotly Figure object
    """
    # Filter the data for the given standard_name
    df_filtered = df[df['standard_name'] == standard_name]

    # Define color codes for each type
    color_mapping = {
        'Ventes': '#008000',      # Green
        '75%': '#0000FF',         # Blue
        'Achats': '#FFA500',      # Orange
        'Historique': '#8B4513',  # Brown
        'Stock': '#800080',       # Purple
        'Rupture': '#FF0000'      # Red
    }

    # Days of the week in the correct order
    days_of_week = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

    # Create an interactive figure
    fig = go.Figure()

    # Iterate through each 'type' and plot a line with the specific color
    for t in df_filtered['type'].unique():
        df_type = df_filtered[df_filtered['type'] == t]
        fig.add_trace(
            go.Scatter(
                x=days_of_week,
                y=df_type[days_of_week].values.flatten(),
                mode='lines+markers',
                name=t,
                line=dict(color=color_mapping.get(t, '#000000')),  # Default to black if type is not found
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        title=f'Performances {standard_name}',
        xaxis_title='Jour Semaine',
        yaxis_title='Poids Total',
        legend_title='Type',
        template="plotly_white"
    )

    return fig


st.divider()

st.caption("Tableau de Comparaison")
with st.expander("Voir Graphs de Comparaisons"):
    st.subheader("Comparaison par Produit")
    for name in merged_table['standard_name'].unique():
        st.subheader(f"{name}")
        fig = plot_standard_name_report_interactive(merged_table, name)
        st.plotly_chart(fig, use_container_width=True)


html_file_content = save_plots_as_html(merged_table)
st.download_button(
    label="Télécharger le rapport",
    data=html_file_content.encode("utf-8"),
    file_name=f"comparison_plots_{comparison_date}.html",
    mime="text/html"
)




