import joblib
import os
import matplotlib.pyplot as plt

# Load the processed data
output_dir = 'output'
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(os.path.join(output_dir, 'processed_data.pkl'))

# Load the trained models
models_dir = 'models'
rf = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
gb = joblib.load(os.path.join(models_dir, 'gb_model.pkl'))
nn = joblib.load(os.path.join(models_dir, 'nn_model.pkl'))

# Make predictions with each model
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test_scaled)
nn_pred = nn.predict(X_test_scaled)

# Ensemble predictions (simple averaging)
ensemble_pred = (rf_pred + gb_pred + nn_pred) / 3

# Plotting the residuals
plt.figure(figsize=(10, 6))

# Random Forest residuals
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_test - rf_pred, color='blue')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Residuals')
plt.title('Random Forest Residuals')

# Gradient Boosting residuals
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_test - gb_pred, color='green')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Residuals')
plt.title('Gradient Boosting Residuals')

# Neural Network residuals
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_test - nn_pred, color='red')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Residuals')
plt.title('Neural Network Residuals')

# Ensemble residuals
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_test - ensemble_pred, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Residuals')
plt.title('Ensemble Residuals')

plt.tight_layout()
plt.show()
