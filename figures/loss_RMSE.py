import joblib
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Load the processed data
output_dir = 'output'
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(os.path.join(output_dir, 'processed_data.pkl'))

# Load the trained models
models_dir = 'models'
rf = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
gb = joblib.load(os.path.join(models_dir, 'gb_model.pkl'))
nn = joblib.load(os.path.join(models_dir, 'nn_model.pkl'))

# Recreate the polynomial features transformer and transform the data
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Make predictions with each model
rf_pred = rf.predict(X_test_poly)
gb_pred = gb.predict(X_test_poly)
nn_pred = nn.predict(X_test_poly)

# Ensemble predictions (simple averaging)
ensemble_pred = (rf_pred + gb_pred + nn_pred) / 3

# Calculate Root Mean Squared Error for each model
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

# Print the RMSE for each model
print(f"Random Forest RMSE: {rf_rmse}")
print(f"Gradient Boosting RMSE: {gb_rmse}")
print(f"Neural Network RMSE: {nn_rmse}")
print(f"Ensemble RMSE: {ensemble_rmse}")

# Plotting the results
models = ['Random Forest', 'Gradient Boosting', 'Neural Network', 'Ensemble']
rmse_values = [rf_rmse, gb_rmse, nn_rmse, ensemble_rmse]

plt.figure(figsize=(10, 6))
plt.bar(models, rmse_values, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error')
plt.title('Model Performance Comparison')
plt.show()