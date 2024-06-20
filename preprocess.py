import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load the dataset
data_path = 'Dataset.xlsx'
data = pd.read_excel(data_path)

# Drop rows with missing values in features or target
data = data.dropna(subset=['Temperature', 'Salinity', 'UVB', 'ChlorophyllaFlor'])

for i in data['Temperature']:
    try: 
        row = data.iloc[[i]].values.tolist()[0]
    except TypeError:
        data = data.drop(labels=i, axis=0)

# Select features and target
X = data[['Temperature', 'Salinity', 'UVB']]
y = data['ChlorophyllaFlor']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the processed data and the scaler
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
joblib.dump((X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test), os.path.join(output_dir, 'processed_data.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

print("Preprocessing complete. Data saved to the output folder.")
