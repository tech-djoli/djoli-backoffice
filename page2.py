import streamlit as st
st.sidebar.header("Attempt")
st.set_page_config(page_title="Attempt", page_icon="🌍")
st.markdown("# Attempt")

df_old = pd.read_csv('/Users/thomasgiannetti/Downloads/old_order_dates2.csv')
df_new = pd.read_csv('/Users/thomasgiannetti/Downloads/new_order_dates3.csv')

df_old['first_order_date'] = pd.to_datetime(df_old['first_order_date'], errors='coerce')
df_new['first_order_date'] = pd.to_datetime(df_new['first_order_date'], errors='coerce')

df_new.rename(columns={
    'Longitude': 'longitude',
    'Latitude': 'latitude',
    'shop_id': 'restaurantID',
    'title': 'name'
}, inplace=True)

# Merge the DataFrames on 'restaurantID' using an outer join
df_combined = pd.merge(
    df_old, df_new, 
    on='restaurantID', 
    how='outer',
    suffixes=('_old', '_new')
)

# Fill missing 'restaurantID' and 'name' with 'shop_id' and 'title' where appropriate
df_combined['restaurantID'] = df_combined['restaurantID'].combine_first(df_combined['restaurantID'])
df_combined['name'] = df_combined['name_old'].combine_first(df_combined['name_new'])

# Handle longitude and latitude: prioritize values from df_new, use df_old if df_new is missing
df_combined['longitude'] = df_combined['longitude_new'].combine_first(df_combined['longitude_old'])
df_combined['latitude'] = df_combined['latitude_new'].combine_first(df_combined['latitude_old'])

# Keep the oldest first_order_date between df_old and df_new
df_combined['first_order_date'] = df_combined[['first_order_date_old', 'first_order_date_new']].min(axis=1)

# Sum 'total_ordered' and 'num_orders' from both tables
df_combined['total_ordered'] = df_combined[['total_ordered_old', 'total_ordered_new']].sum(axis=1)
df_combined['num_orders'] = df_combined[['num_orders_old', 'num_orders_new']].sum(axis=1)

# Handle the 'type' column: use 'type_new' if available, otherwise use 'type_old'
df_combined['type'] = df_combined['type_new'].combine_first(df_combined['type_old'])

# Select only the desired columns
df_result = df_combined[['restaurantID', 'name', 'longitude', 'latitude', 'first_order_date', 'total_ordered', 'num_orders', 'type']]

# Drop any duplicate columns that were created during the merge
df_result = df_result.loc[:, ~df_result.columns.duplicated()]

df_result = df_result.dropna(subset=['latitude', 'longitude'])

