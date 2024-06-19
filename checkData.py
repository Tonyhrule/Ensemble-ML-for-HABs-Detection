import joblib
import os
import numpy as np

# Load the processed data and the scaler
output_dir = 'output'
processed_data_path = os.path.join(output_dir, 'processed_data.pkl')
scaler_path = os.path.join(output_dir, 'scaler.pkl')

# Load the processed data
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(processed_data_path)

# Load the scaler
scaler = joblib.load(scaler_path)

# Print summaries
print("Processed Data Summary:")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Print a few examples
print("\nExample features (first 5 rows of X_train):")
print(X_train[:5])
print("\nExample of scaled features (first 5 rows of X_train_scaled):")
print(X_train_scaled[:5])

print("\nExample of target values (first 5 rows of y_train):")
print(y_train[:5].to_string(index=False))

# Check scaler functionality
example_data = np.array([[20.0, 30.0, 0.1]])
scaled_example = scaler.transform(example_data)
print("\nExample of scaler transformation (for data [20.0, 30.0, 0.1]):")
print(scaled_example)