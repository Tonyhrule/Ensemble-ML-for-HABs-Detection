import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
import numpy as np

# Load the processed data
output_dir = 'output'
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(os.path.join(output_dir, 'processed_data.pkl'))

# Load the trained models
models_dir = 'models'
rf = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
gb = joblib.load(os.path.join(models_dir, 'gb_model.pkl'))
nn = joblib.load(os.path.join(models_dir, 'nn_model.pkl'))

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_scaled = imputer.fit_transform(X_train_scaled)
X_test_scaled = imputer.transform(X_test_scaled)
y_train = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test = imputer.transform(y_test.values.reshape(-1, 1)).ravel()

# Recreate the polynomial features transformer and transform the test data
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Make predictions with each model
rf_pred = rf.predict(X_test_poly)
gb_pred = gb.predict(X_test_poly)
nn_pred = nn.predict(X_test_poly)

# Ensemble predictions (simple averaging)
ensemble_pred = (rf_pred + gb_pred + nn_pred) / 3

# Function to calculate and format RMSE for legends
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Plotting the residuals
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

residuals = [(rf_pred, 'Random Forest', 'blue'),
             (gb_pred, 'Gradient Boosting', 'green'),
             (nn_pred, 'Neural Network', 'red'),
             (ensemble_pred, 'Ensemble', 'purple')]

for ax, (pred, title, color) in zip(axes.flatten(), residuals):
    residuals = y_test - pred
    rmse = calculate_rmse(y_test, pred)
    ax.scatter(y_test, residuals, color=color, s=10, alpha=0.7, label=f'RMSE: {rmse:.2f}')
    ax.axhline(y=0, color='black', linestyle='--')
    ax.set_xlabel('Actual', fontsize=14)
    ax.set_ylabel('Residuals', fontsize=14)
    ax.set_title(f'{title} Residuals', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    # Set y-limits manually as in the picture
    ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()
