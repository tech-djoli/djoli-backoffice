import pandas as pd
import mysql.connector as connection
from sshtunnel import SSHTunnelForwarder
from operator import attrgetter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import re
import pulp
import streamlit as st

tunnel = SSHTunnelForwarder(
    ('31.207.38.195', 22),  # SSH server and port
    ssh_username="root",
    ssh_password="Us9Gacz6rtG",
    remote_bind_address=('127.0.0.1', 3306),
    allow_agent=False,  # Disable SSH key authentication
    host_pkey_directories=[]  # Disable default key file lookup
)

# Start the SSH tunnel
tunnel.start()

# MySQL connection through the SSH tunnel
mydb = connection.connect(
    host='127.0.0.1',  # Localhost since we are tunneling
    database='backend',
    user='djoli',
    passwd='vmftza%WLfpc#cN@',
    port=tunnel.local_bind_port,  # Use the forwarded local port
    use_pure=True
)

query = """SELECT
    o.id AS order_id,
    ps.name AS standard
FROM orders o
JOIN order_details od ON od.order_id = o.id
JOIN stocks s ON s.id = od.stock_id
JOIN products p ON p.id = s.countable_id
JOIN product_standards ps ON ps.id = p.product_standard_id;
"""

df = pd.read_sql(query,mydb)

basket = pd.get_dummies(df['standard']).groupby(df['order_id']).max()
basket.index.rename("orderID")

frequent_itemsets = apriori(basket, min_support=0.005, use_colnames=True)

st.title('Djoli - Frequent Item Set Associations')

metric = st.selectbox(
    'Select a metric.',
    ('support', 'confidence', 'lift', 'conviction', 'zhangs_metric'))

rules = association_rules(frequent_itemsets, metric=metric, min_threshold=0.05)

sku_num = st.slider("Number of SKUs by Item Set", 1, 5, 1)

rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))
rules['consequent_len'] = rules['consequents'].apply(lambda x: len(x))

# Filter the rules based on the length of antecedents and consequents
filtered_rules = rules[(rules['antecedent_len'] <= sku_num) & (rules['consequent_len'] <= sku_num)]
sorted_rules = filtered_rules.sort_values(by=["support", "confidence"], ascending=[False, False])

num_itemsets = st.slider("Number of Item Sets", 5, 100, 5)

top_rules = sorted_rules.head(num_itemsets)

for index, row in top_rules.iterrows():
    # Map SKU numbers to names for antecedents
    antecedents = " & ".join([f"{name}" for name in row['antecedents']])

    # Map SKU numbers to names for consequents
    consequents = " & ".join([f"{name}" for name in row['consequents']])

    confidence = row['confidence'] * 100  # Convert confidence to percentage
    support = row['support']

    st.subheader(f"{antecedents} & {consequents}\n")
    st.write(f"There is a {confidence:.2f}% probability of finding {consequents} in a transaction given that {antecedents} is present.\n")
    st.write(f"This item association has occurred in approximately {support:.2f}% of all transactions.\n")
    st.markdown(f"\n")
    st.divider()
