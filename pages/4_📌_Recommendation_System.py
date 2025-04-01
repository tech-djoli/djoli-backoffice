import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sshtunnel import SSHTunnelForwarder
import mysql.connector as connection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


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

query = """SELECT
    o.shop_id,
    st.title,
    d.zone,
    d.name,
    stt.title AS cuisine,
    s.type_establishment,
    o.id AS order_id,
    o.created_at,
    o.delivery_date,
    o.delivery_time,
    CASE
        WHEN o.created_at < CONCAT(o.delivery_date, ' 02:00:00') THEN 'J+1'
        ELSE 'J0'
    END AS order_type,
    DAYNAME(o.delivery_date) AS day_of_week,
    CEIL(DAYOFMONTH(o.delivery_date) / 7) AS week_of_month,
    p.id AS product_id,
    pt.title AS product_name,
    ut.title AS unit,
    ps.id AS standard_id,
    ps.name AS standard_name,
    od.quantity,
    od.quantity * p.weight AS total_weight,
    od.unit_price,
    od.total_price
FROM orders o
JOIN shops s ON o.shop_id = s.id
JOIN assign_shop_tags ast ON ast.shop_id = s.id
JOIN shop_tag_translations stt ON stt.shop_tag_id = ast.shop_tag_id
JOIN shop_translations st ON st.shop_id = s.id
JOIN order_details od ON od.order_id = o.id
JOIN stocks so ON so.id = od.stock_id
JOIN products p ON so.countable_id = p.id
JOIN unit_translations ut ON ut.unit_id = p.unit_id
JOIN product_translations pt ON pt.product_id = p.id
JOIN product_standards ps ON ps.id = p.product_standard_id
JOIN districts d ON d.id = s.district_id;

"""

df = pd.read_sql(query,mydb)
df['total_weight'].fillna(df['total_weight'].median(), inplace=True)

df['total_weight'].fillna(0, inplace=True)

unique_shops = df[['shop_id', 'title', 'type_establishment', 'cuisine']].drop_duplicates()
unique_product_standards = df[['standard_id', 'standard_name']].drop_duplicates()

features = ['shop_id','day_of_week', 'week_of_month', 'type_establishment', 'order_type', 'cuisine', 'standard_id']
target = 'total_weight'  # Assuming we are predicting total_weight for now

# Separate features and target
X = df[features]
y = df[target]

# One-hot encode categorical variables (day_of_week, type_establishment, order_type, cuisine, product_standard)
encoder = OneHotEncoder(sparse_output=False)

# Cache the model training process so it only runs once
@st.cache_resource
def train_model():

    # One-hot encode the training data
    customer_encoded = encoder.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(customer_encoded, y, test_size=0.2, random_state=42)

    # Train the Random Forest Regressor model
    model = RandomForestRegressor(
        n_estimators=500,       # Increase number of trees
        max_depth=10,           # Limit the maximum depth of trees
        min_samples_split=5,    # Require at least 5 samples to split a node
        min_samples_leaf=4,     # Minimum 4 samples at a leaf node
        random_state=42
    )

    cv_scores = cross_val_score(model, customer_encoded, y, cv=5, scoring='neg_mean_squared_error')

    model.fit(X_train, y_train)

    return model, encoder

model, encoder = train_model()


def predict_for_restaurant_with_filter(shop_id, model, encoder):
    # Extract details for the selected shop
    shop_details = unique_shops[unique_shops['shop_id'] == shop_id].iloc[0]

    # Days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    predictions = []

    # Define thresholds for filtering predictions
    min_predicted_weight = 1  # Only keep predictions with weight above 5 kg
    max_confidence_range = 3  # Stricter range for confidence intervals (narrower range)

    for day in days_of_week:
        # Generate input data for all product standards for this shop
        customer_data = pd.DataFrame({
            'shop_id': [shop_id] * len(unique_product_standards),
            'day_of_week': [day] * len(unique_product_standards),
            'week_of_month': [1] * len(unique_product_standards),  # Placeholder for week of the month
            'type_establishment': [shop_details['type_establishment']] * len(unique_product_standards),
            'order_type': ['J+1'] * len(unique_product_standards),  # Placeholder for order type
            'cuisine': [shop_details['cuisine']] * len(unique_product_standards),
            'standard_id': unique_product_standards['standard_id'].values,
            'standard_name': unique_product_standards['standard_name'].values
        })

        # One-hot encode the customer data
        customer_encoded = encoder.transform(customer_data.drop(columns=['standard_name']))

        # Predict total weight for each product standard
        predicted_weights = model.predict(customer_encoded)

        # Get predictions from individual trees for confidence intervals
        predictions_from_trees = [tree.predict(customer_encoded) for tree in model.estimators_]
        predictions_from_trees = np.array(predictions_from_trees)
        mean_prediction = np.mean(predictions_from_trees, axis=0)
        std_prediction = np.std(predictions_from_trees, axis=0)

        confidence_level = 1.96  # For 95% confidence interval
        lower_bound = mean_prediction - confidence_level * std_prediction
        upper_bound = mean_prediction + confidence_level * std_prediction

        # Filter predictions based on predicted weight and confidence interval
        for i in range(len(unique_product_standards)):
            confidence_range = upper_bound[i] - lower_bound[i]
            if predicted_weights[i] >= min_predicted_weight and confidence_range <= max_confidence_range:
                predictions.append({
                    'day_of_week': day,
                    'product_standard': unique_product_standards.iloc[i]['standard_name'],  # For display purposes
                    'predicted_weight': predicted_weights[i],
                    'confidence_interval': f"[{lower_bound[i]:.2f}, {upper_bound[i]:.2f}]",
                    'confidence_range': confidence_range
                })

    return predictions


# Streamlit app
st.title("Restaurant Product Demand Prediction")

# Select a restaurant from the DataFrame using st.selectbox
selected_restaurant_name = st.selectbox(
    "Select a restaurant",
    unique_shops['title'].unique()
)

# Get the selected restaurant's shop_id
selected_shop_id = unique_shops[unique_shops['title'] == selected_restaurant_name]['shop_id'].values[0]

# Button to trigger prediction
if st.button("Generate Predictions"):
    # Call the prediction function for the selected restaurant
    predictions = predict_for_restaurant_with_filter(selected_shop_id, model, encoder)

    # Display the predictions in a table
    if predictions:
        st.write(f"Predicted Products for {selected_restaurant_name}:")
        predictions_df = pd.DataFrame(predictions)
        st.dataframe(predictions_df)
    else:
        st.write("No predictions with low error.")

