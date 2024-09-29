import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib

# Step 1: Load the training data
training_data = pd.read_csv("training.csv")

# Step 2: Handle missing values
# Impute missing values for 'cummulative_continous_volume' with the mean
imputer = SimpleImputer(strategy='mean')
training_data['cummulative_continous_volume'] = imputer.fit_transform(training_data[['cummulative_continous_volume']])

# Drop rows where 'close_volume' is missing
training_data_clean = training_data.dropna(subset=['close_volume'])

# Step 3: Define features and labels
features = training_data_clean[['cummulative_continous_volume', 'seconds_bucket']]
labels = training_data_clean['close_volume']

# Step 4: Train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(features, labels)

# Step 5: Save the trained model using joblib
joblib.dump(model, 'close_volume_model.joblib')

print("Model has been trained and saved as 'close_volume_model.joblib'.")
