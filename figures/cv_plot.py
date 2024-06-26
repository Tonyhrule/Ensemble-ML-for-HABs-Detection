import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Debug: Start of the script
print("Starting the script...")

# Load the data
output_dir = 'output'
print("Loading processed data...")
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(os.path.join(output_dir, 'processed_data.pkl'))
print("Data loaded successfully.")

# Load the scaler and models
models_dir = 'models'
print("Loading scaler and models...")
scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
rf_best = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
gb_best = joblib.load(os.path.join(models_dir, 'gb_model.pkl'))
nn_best = joblib.load(os.path.join(models_dir, 'nn_model.pkl'))
stacker_best = joblib.load(os.path.join(models_dir, 'stacker_model.pkl'))
print("Scaler and models loaded successfully.")

# Recreate the polynomial features transformer with the same degree as used in training
print("Creating polynomial features...")
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print("Polynomial features created successfully.")

def plot_cv_results(cv_results, model_names):
    print("Plotting cross-validation results...")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, model_name, results in zip(axes, model_names, cv_results):
        mean_score = np.mean(-results)
        ax.plot(range(1, len(results) + 1), -results, 'o-', label=f'{model_name}')
        ax.fill_between(range(1, len(results) + 1), -results - np.std(-results), -results + np.std(-results), alpha=0.2)
        ax.set_title(f'{model_name} (Mean MSE: {mean_score:.2f})')
        ax.set_xlabel('Fold')
        ax.set_ylabel('MSE')
        ax.legend()
        ax.grid(True)
    
    fig.suptitle('Cross-Validation Results for Different Models', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("Plotting done.")

# Cross-validation setup
print("Setting up cross-validation...")
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
print("Cross-validation setup done.")

# Perform cross-validation and collect results
print("Performing cross-validation for Random Forest...")
cv_results_rf = cross_val_score(rf_best, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
print("Random Forest CV done.")

print("Performing cross-validation for Gradient Boosting...")
cv_results_gb = cross_val_score(gb_best, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
print("Gradient Boosting CV done.")

print("Performing cross-validation for Neural Network...")
cv_results_nn = cross_val_score(nn_best, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
print("Neural Network CV done.")

print("Performing cross-validation for Stacked Model...")
cv_results_stacker = cross_val_score(stacker_best, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
print("Stacked Model CV done.")

# Plot the cross-validation results
print("Plotting the results...")
plot_cv_results([cv_results_rf, cv_results_gb, cv_results_nn, cv_results_stacker], 
                ['Random Forest', 'Gradient Boosting', 'Neural Network', 'Stacked Model'])
print("Script completed.")
