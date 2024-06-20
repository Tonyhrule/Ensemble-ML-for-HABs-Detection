import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import os

# Load the processed data
output_dir = 'output'
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(os.path.join(output_dir, 'processed_data.pkl'))

# Train base models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
nn = MLPRegressor(hidden_layer_sizes=(70, 70), max_iter=500, random_state=42)

# Fit the models
rf.fit(X_train, y_train)
gb.fit(X_train_scaled, y_train)
nn.fit(X_train_scaled, y_train)

# Make predictions
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test_scaled)
nn_pred = nn.predict(X_test_scaled)

# Ensemble predictions (simple averaging)
ensemble_pred = (rf_pred + gb_pred + nn_pred) / 3

# Evaluate the models
rf_mse = mean_squared_error(y_test, rf_pred)
gb_mse = mean_squared_error(y_test, gb_pred)
nn_mse = mean_squared_error(y_test, nn_pred)
ensemble_mse = mean_squared_error(y_test, ensemble_pred)

print(f"Random Forest MSE: {rf_mse}")
print(f"Gradient Boosting MSE: {gb_mse}")
print(f"Neural Network MSE: {nn_mse}")
print(f"Ensemble MSE: {ensemble_mse}")

# Save the models to a new folder
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
joblib.dump(rf, os.path.join(models_dir, 'rf_model.pkl'))
joblib.dump(gb, os.path.join(models_dir, 'gb_model.pkl'))
joblib.dump(nn, os.path.join(models_dir, 'nn_model.pkl'))
joblib.dump(ensemble_pred, os.path.join(models_dir, 'ensemble_pred.pkl'))

print("Training complete. Models saved to the models folder.")

