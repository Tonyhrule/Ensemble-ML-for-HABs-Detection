import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import PolynomialFeatures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
output_dir = 'output'
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(os.path.join(output_dir, 'processed_data.pkl'))

# Load the scaler and models
models_dir = 'models'
scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
rf_best = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
gb_best = joblib.load(os.path.join(models_dir, 'gb_model.pkl'))
nn_best = joblib.load(os.path.join(models_dir, 'nn_model.pkl'))

# Recreate the polynomial features transformer with the same degree as used in training
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

def predict_chlorophyll(temperature, salinity, uvb):
    # Create a DataFrame for the input features to include feature names
    features = pd.DataFrame([[temperature, salinity, uvb]], columns=['Temperature', 'Salinity', 'UVB'])
    
    # Scale the input features
    features_scaled = scaler.transform(features)
    
    # Apply polynomial transformation
    features_poly = poly.fit_transform(features_scaled)
    
    # Make predictions with each model
    rf_pred = rf_best.predict(features_poly)
    gb_pred = gb_best.predict(features_poly)
    nn_pred = nn_best.predict(features_poly)
    
    # Ensemble prediction (simple averaging)
    ensemble_pred = (rf_pred + gb_pred + nn_pred) / 3
    
    return ensemble_pred[0]

def categorize_chlorophyll_a(chlorophyll_a_value):
    if chlorophyll_a_value < 2:
        return "Low Risk (0-5% Probability)"
    elif 2 <= chlorophyll_a_value < 7:
        return "Moderate Risk (5-25% Probability)"
    elif 7 <= chlorophyll_a_value < 12:
        return "High Risk (25-50% Probability)"
    elif chlorophyll_a_value >= 12:
        return "Very High Risk (50-100% Probability)"
    else:
        return "Invalid Value"

# Example usage
error_list = []
for i in range(len(X_test)):
    row = X_test.iloc[i]
    test_row = y_test.iloc[i]
    try:
        chlorophyll_a_corrected = round(predict_chlorophyll(row['Temperature'], row['Salinity'], row['UVB']), 4)
        percent_error = ((chlorophyll_a_corrected - test_row) / test_row) * 100
        error_list.append(percent_error)

        # Get risk category
        risk_category = categorize_chlorophyll_a(chlorophyll_a_corrected)
        print(f"Predicted Chlorophyll a Corrected: {chlorophyll_a_corrected} µg/L - {risk_category}")
    except Exception as e:
        logging.error(f"Error predicting for row {i}: {e}")

# Plotting percent error
plt.figure(figsize=(10, 6))
plt.plot(range(len(error_list)), error_list, marker='o', markersize=3, alpha=0.7)
plt.xlabel('Trial')
plt.ylabel('Percent Error')
plt.title('Model Percent Error over each Trial')
plt.ylim(-50, 100)  # Set y-axis limits to focus on main range
plt.grid(True)
plt.show()
