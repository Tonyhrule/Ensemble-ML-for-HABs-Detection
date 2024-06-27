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
poly.fit(X_train_scaled)  # Fit the transformer to the training data

def predict_chlorophyll(temperature, salinity, uvb):
    # Create a DataFrame for the input features to include feature names
    features = pd.DataFrame([[temperature, salinity, uvb]], columns=['Temperature', 'Salinity', 'UVB'])
    
    # Scale the input features
    features_scaled = scaler.transform(features)
    
    # Apply polynomial transformation
    features_poly = poly.transform(features_scaled)
    
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
    actual_value = y_test.iloc[i]
    try:
        chlorophyll_a_corrected = round(predict_chlorophyll(row['Temperature'], row['Salinity'], row['UVB']), 4)
        percent_error = abs((chlorophyll_a_corrected - actual_value) / actual_value) * 100
        error_list.append(percent_error)

        # Print percent error
        print(f"Percent Error for row {i}: {percent_error:.2f}%")
        
    except Exception as e:
        logging.error(f"Error predicting for row {i}: {e}")

# Calculate average percent error
average_error = np.mean(error_list)

# Sort the error list in ascending order
error_list_sorted = sorted(error_list)

# Plotting percent error
plt.figure(figsize=(10, 6))
plt.scatter(range(len(error_list_sorted)), error_list_sorted, marker='o', s=10, alpha=0.7)  # Smaller dots
plt.axhline(y=average_error, color='r', linestyle='--', label=f'Average Error: {average_error:.2f}%')
plt.xlabel('Sorted Trial Index', fontsize=14)  # Updated x-axis label
plt.ylabel('Percent Error', fontsize=14)
plt.title('Model Percent Error in Increasing Order', fontsize=16)
plt.ylim(0, 100)  # Set y-axis limits from 0 to 100%
plt.xlim(0, len(error_list_sorted) - 1)  # Set x-axis limits to fit the data tightly
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)  # Adjust x-axis tick labels
plt.yticks(fontsize=12)  # Adjust y-axis tick labels
plt.show()
