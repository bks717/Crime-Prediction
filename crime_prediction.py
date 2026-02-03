import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# 1. Load the CSV
file_path = "NCRB_Table_1A.1.csv"
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# 2. Clean the dataset
# Keep only State/UT names and total IPC crimes for each year (2020, 2021, 2022)
# The column names in the CSV are:
# Sl. No., State/UT, 2020, 2021, 2022, ...
print("Cleaning data...")
df_clean = df[['State/UT', '2020', '2021', '2022']].copy()

# Remove total rows. Based on inspection, they contain "Total" in "State/UT"
# "Total State (S)", "Total UT(S)", "Total All India"
df_clean = df_clean[~df_clean['State/UT'].str.contains('Total', case=False, na=False)]

# 3. Reshape the dataset
print("Reshaping data...")
# Melt the dataframe to: State | Year | Crime_Count
df_melted = df_clean.melt(id_vars=['State/UT'], 
                          var_name='Year', 
                          value_name='Crime_Count')
df_melted.rename(columns={'State/UT': 'State'}, inplace=True)

# Convert Year to integer
df_melted['Year'] = df_melted['Year'].astype(int)

# 4. Encode State names
print("Encoding state names...")
le = LabelEncoder()
df_melted['State_Encoded'] = le.fit_transform(df_melted['State'])

# 5. Split data
# Training: 2020, 2021
# Testing: 2022
print("Splitting data into train (2020-2021) and test (2022)...")
train_data = df_melted[df_melted['Year'].isin([2020, 2021])]
test_data = df_melted[df_melted['Year'] == 2022]

X_train = train_data[['State_Encoded', 'Year']]
y_train = train_data['Crime_Count']

X_test = test_data[['State_Encoded', 'Year']]
y_test = test_data['Crime_Count']

# 6. Train Random Forest Regressor
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 7. Evaluate model
print("Evaluating model...")
y_pred_2022 = rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred_2022)
mae = mean_absolute_error(y_test, y_pred_2022)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_2022))

print("-" * 30)
print(f"Model Evaluation (on 2022 data):")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print("-" * 30)

# 8. Predict for 2023
print("Predicting for 2023...")
states = df_melted['State'].unique()
states_encoded = le.transform(states)
year_2023 = [2023] * len(states)

X_2023 = pd.DataFrame({
    'State_Encoded': states_encoded,
    'Year': year_2023
})

predictions_2023 = rf_model.predict(X_2023)

# Create a clean results DataFrame
results_2023 = pd.DataFrame({
    'State': states,
    'Predicted_Crime_2023': predictions_2023.astype(int)
})

# Sort by predicted crime count for better visualization
results_2023 = results_2023.sort_values(by='Predicted_Crime_2023', ascending=False)

# 10. Print clean table
print("\nPredicted IPC Crime Counts for 2023:")
print(results_2023.to_string(index=False))

# 9. Plotting
print("Generating plots...")

# Plot 1: Map Visualization (PRIORITY)
print("Generating Map Visualization...")
try:
    import geopandas as gpd
    import matplotlib.colors as mcolors

    # Load India GeoJSON
    geojson_url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
    print(f"Downloading GeoJSON from {geojson_url}...")
    india_map = gpd.read_file(geojson_url)
    
    # Mapping dictionary for known mismatches
    name_mapping = {
        "Andaman & Nicobar Island": "Andaman and Nicobar Islands",
        "Arunanchal Pradesh": "Arunachal Pradesh",
        "Dadra and Nagar Haveli and Daman and Diu": "Dadra and Nagar Haveli and Daman and Diu",
        "NCT of Delhi": "Delhi",
        "Jammu & Kashmir": "Jammu and Kashmir",
        "Odisha": "Odisha",
        "Telangana": "Telangana"
    }
    
    india_map['ST_NM'] = india_map['ST_NM'].replace(name_mapping)
    
    # Merge
    map_data = india_map.merge(results_2023, left_on='ST_NM', right_on='State', how='left')
    
    # Plot - Create a figure with 2 subplots (Map on left, Text list on right)
    fig, (ax_map, ax_text) = plt.subplots(1, 2, figsize=(24, 12), gridspec_kw={'width_ratios': [3, 1]})
    
    # 1. Map Plot
    map_data.plot(column='Predicted_Crime_2023', 
                  ax=ax_map, 
                  legend=True, 
                  legend_kwds={'label': "Predicted IPC Crimes 2023", 'shrink': 0.6},
                  cmap='YlOrRd',
                  edgecolor='black',
                  linewidth=0.8,
                  missing_kwds={'color': 'lightgrey'})
    
    ax_map.set_title('Predicted IPC Crime Counts for 2023', fontsize=16, fontweight='bold')
    ax_map.set_axis_off()

    # 2. Side Panel List
    ax_text.axis('off')
    ax_text.set_title('State-wise Predictions', fontsize=14, fontweight='bold', pad=20)
    
    # Prepare text content
    # Sort by prediction count descending
    sorted_data = results_2023.sort_values(by='Predicted_Crime_2023', ascending=False)
    
    # Create a table or list
    # We will simply place text to make it look like a clean list
    # Header
    header_text = f"{'State':<25} | {'Count':>10}"
    ax_text.text(0, 1.0, header_text, fontsize=12, fontweight='bold', family='monospace', transform=ax_text.transAxes)
    ax_text.text(0, 0.98, "-"*40, fontsize=12, family='monospace', transform=ax_text.transAxes)
    
    # Rows
    y_pos = 0.95
    # If there are too many, we might need to split columns or reduce font. 
    # With ~36 states, it fits vertically if tight, or we split.
    # Let's try fitting straight down first.
    
    for idx, row in sorted_data.iterrows():
        state_name = row['State']
        # Truncate long names slightly if needed for better fit
        if len(state_name) > 23:
             state_name = state_name[:20] + "..."
        
        count_str = f"{int(row['Predicted_Crime_2023']):,}"
        line_str = f"{state_name:<25} | {count_str:>10}"
        
        ax_text.text(0, y_pos, line_str, fontsize=10, family='monospace', transform=ax_text.transAxes)
        y_pos -= 0.025 # Decrement position
        
        if y_pos < 0: # Stop if we run out of space (unlikely with this spacing)
            break

    plt.tight_layout()
    print("Displaying Map with Side Panel... (Please close the window to exit)")
    plt.show()
    
except Exception as e:
    print(f"Could not generate map: {e}")
    print("Please ensure internet connection is available to download GeoJSON.")

# Other plots are commented out or secondary to ensure Map is seen first
# ... (Bar and Comparison plots removed/commented for specific focus as per user reaction)



# Plot 2: Actual vs Predicted for 2022
