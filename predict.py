import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

output_dir = 'output'
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(os.path.join(output_dir, 'processed_data.pkl'))

# Load the scaler and models
models_dir = 'models'
output_dir = 'output'
scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
rf = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
gb = joblib.load(os.path.join(models_dir, 'gb_model.pkl'))
nn = joblib.load(os.path.join(models_dir, 'nn_model.pkl'))

def predict_chlorophyll(temperature, salinity, uvb):
    # Create a DataFrame for the input features to include feature names
    features = pd.DataFrame([[temperature, salinity, uvb]], columns=['Temperature', 'Salinity', 'UVB'])
    
    # Scale the input features
    features_scaled = scaler.transform(features)
    
    # Make predictions with each model
    rf_pred = rf.predict(features)
    gb_pred = gb.predict(features_scaled)
    nn_pred = nn.predict(features_scaled)
    
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

# Example
error_list = []
for i in range(len(X_test['Temperature'])):
    row = X_test.iloc[[i]].values.tolist()[0]
    test_row = y_test.iloc[[i]].values.tolist()[0]
    chlorophyll_a_corrected = round(predict_chlorophyll(row[0], row[1], row[2]), 4)
    percent_error = ((chlorophyll_a_corrected - test_row)/test_row) * 100
    error_list.append(percent_error)
    
    # Get risk category
    risk_category = categorize_chlorophyll_a(chlorophyll_a_corrected)
    print(f"Predicted Chlorophyll a Corrected: {chlorophyll_a_corrected} µg/L - {risk_category}")

# Plotting percent error
plt.figure(figsize=(10, 6))        # Set the figure size
plt.plot(range(len(error_list)), error_list, marker=None)  # Plot the error_list values
plt.xlabel('Trial')                # Label for the x-axis
plt.ylabel('Percent Error')        # Label for the y-axis
plt.title('Model Percent Error over each Trial')  # Title of the plot
plt.grid(True)                     # Add grid for better readability
plt.show()
