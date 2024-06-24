import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the processed data
output_dir = 'output'
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(os.path.join(output_dir, 'processed_data.pkl'))

max_epochs = 50

# Initialize models
rf = RandomForestRegressor(n_estimators=10, warm_start=True, random_state=42)
gb = GradientBoostingRegressor(n_estimators=10, warm_start=True, random_state=42)
nn = MLPRegressor(hidden_layer_sizes=(200, 200), max_iter=1, warm_start=True, random_state=42)

rf_train_losses = []
gb_train_losses = []
nn_train_losses = []

# Train Random Forest with incremental updates
for epoch in range(max_epochs):
    rf.n_estimators += 10  # Increase the number of trees
    rf.fit(X_train, y_train)
    rf_train_predictions = rf.predict(X_train)
    rf_loss = mean_squared_error(y_train, rf_train_predictions)
    rf_train_losses.append(rf_loss)
    print(f'Epoch {epoch+1}/{max_epochs}, Random Forest Loss: {rf_loss}')

# Save the trained model
joblib.dump(rf, os.path.join(models_dir, 'rf_model.pkl'))

# Train Gradient Boosting with incremental updates
for epoch in range(max_epochs):
    gb.n_estimators += 10  # Increase the number of trees
    gb.fit(X_train_scaled, y_train)
    gb_train_predictions = gb.predict(X_train_scaled)
    gb_loss = mean_squared_error(y_train, gb_train_predictions)
    gb_train_losses.append(gb_loss)
    print(f'Epoch {epoch+1}/{max_epochs}, Gradient Boosting Loss: {gb_loss}')

# Save the trained model
joblib.dump(gb, os.path.join(models_dir, 'gb_model.pkl'))

# Train Neural Network with incremental updates
for epoch in range(max_epochs):
    nn.partial_fit(X_train_scaled, y_train)
    nn_train_predictions = nn.predict(X_train_scaled)
    nn_loss = mean_squared_error(y_train, nn_train_predictions)
    nn_train_losses.append(nn_loss)
    print(f'Epoch {epoch+1}/{max_epochs}, Neural Network Loss: {nn_loss}')

# Save the trained model
joblib.dump(nn, os.path.join(models_dir, 'nn_model.pkl'))

# Plotting the training losses
plt.figure(figsize=(10, 6))
plt.plot(range(len(rf_train_losses)), rf_train_losses, label='Random Forest', marker='o')
plt.plot(range(len(gb_train_losses)), gb_train_losses, label='Gradient Boosting', marker='o')
plt.plot(range(len(nn_train_losses)), nn_train_losses, label='Neural Network', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Model Training Losses')
plt.legend()
plt.grid(True)
plt.show()