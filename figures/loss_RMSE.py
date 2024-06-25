import joblib
from sklearn.metrics import root_mean_squared_error
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load the processed data
output_dir = 'output'
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(os.path.join(output_dir, 'processed_data.pkl'))

# Load the trained models
models_dir = 'models'
rf = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
gb = joblib.load(os.path.join(models_dir, 'gb_model.pkl'))
nn = joblib.load(os.path.join(models_dir, 'nn_model.pkl'))

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
# Make predictions with each model
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
X_test_unscaled = poly.transform(X_test)

rf_pred = rf.predict(X_test_unscaled)
gb_pred = gb.predict(X_test_poly)
nn_pred = nn.predict(X_test_poly)


# Ensemble predictions (simple averaging)
ensemble_pred = (rf_pred + gb_pred + nn_pred) / 3

# Calculate Mean Squared Error for each model
rf_mse = root_mean_squared_error(y_test, rf_pred)
gb_mse = root_mean_squared_error(y_test, gb_pred)
nn_mse = root_mean_squared_error(y_test, nn_pred)
ensemble_mse = root_mean_squared_error(y_test, ensemble_pred)

# Print the MSE for each model
print(f"Random Forest MSE: {rf_mse}")
print(f"Gradient Boosting MSE: {gb_mse}")
print(f"Neural Network MSE: {nn_mse}")
print(f"Ensemble MSE: {ensemble_mse}")

# Plotting the results
models = ['Random Forest', 'Gradient Boosting', 'Neural Network', 'Ensemble']
mse_values = [rf_mse, gb_mse, nn_mse, ensemble_mse]

print(ensemble_mse)

plt.figure(figsize=(10, 6))
plt.bar(models, mse_values, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error Loss')
plt.title('Model Performance Comparison')
plt.show()