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
    print("Plotting cross-validation results (MSE)...")
    X = range(1, len(cv_results[0]) + 1)
    rf_y = [i * -1 for i in cv_results[0]]
    gb_y = [i * -1 for i in cv_results[1]]
    nn_y = [i * -1 for i in cv_results[2]]
    stacker_y = [i * -1 for i in cv_results[3]]

    rf_mean = np.mean(-cv_results[0])
    gb_mean = np.mean(-cv_results[1])
    nn_mean = np.mean(-cv_results[2])
    stack_mean = np.mean(-cv_results[3])

    plt.plot(X, rf_y, color='blue', linewidth=2, alpha=0.3)
    plt.axhline(y=rf_mean, color='blue', linestyle='dashed', label=f'Mean Random Forest MSE: {rf_mean:.3f}')

    plt.plot(X, gb_y, color='green', linewidth=2, alpha=0.3)
    plt.axhline(y=gb_mean, color='green', linestyle='dashed', label=f'Mean Gradient Boosting MSE: {gb_mean:.3f}')

    plt.plot(X, nn_y, color='red', linewidth=2, alpha=0.3)
    plt.axhline(y=nn_mean, color='red', linestyle='dashed', label=f'Mean Neural Network MSE: {nn_mean:.3f}')

    plt.plot(X, stacker_y, color='purple', linewidth=2, alpha=0.2)
    plt.axhline(y=stack_mean, color='purple', linestyle='dashed', label=f'Mean Ensemble MSE: {stack_mean:.3f}')

    plt.xlabel('Fold', fontsize=10)
    plt.ylabel('MSE', fontsize=10)
    plt.legend(fontsize=10)

    plt.grid(True)
    
    plt.title('Cross-Validation Results for Different Models (MSE)', fontsize=13)
    plt.show()
    print("Plotting done.")

def plot_cv_results_rmse(cv_results, model_names):
    print("Plotting cross-validation results (RMSE)...")
    X = range(1, len(cv_results[0]) + 1)
    rf_y = [np.sqrt(i * -1) for i in cv_results[0]]
    gb_y = [np.sqrt(i * -1) for i in cv_results[1]]
    nn_y = [np.sqrt(i * -1) for i in cv_results[2]]
    stacker_y = [np.sqrt(i * -1) for i in cv_results[3]]

    rf_mean = np.mean([np.sqrt(i * -1) for i in cv_results[0]])
    gb_mean = np.mean([np.sqrt(i * -1) for i in cv_results[1]])
    nn_mean = np.mean([np.sqrt(i * -1) for i in cv_results[2]])
    stack_mean = np.mean([np.sqrt(i * -1) for i in cv_results[3]])

    plt.plot(X, rf_y, color='blue', linewidth=2, alpha=0.3)
    plt.axhline(y=rf_mean, color='blue', linestyle='dashed', label=f'Mean Random Forest RMSE: {rf_mean:.3f}')

    plt.plot(X, gb_y, color='green', linewidth=2, alpha=0.3)
    plt.axhline(y=gb_mean, color='green', linestyle='dashed', label=f'Mean Gradient Boosting RMSE: {gb_mean:.3f}')

    plt.plot(X, nn_y, color='red', linewidth=2, alpha=0.3)
    plt.axhline(y=nn_mean, color='red', linestyle='dashed', label=f'Mean Neural Network RMSE: {nn_mean:.3f}')

    plt.plot(X, stacker_y, color='purple', linewidth=2, alpha=0.2)
    plt.axhline(y=stack_mean, color='purple', linestyle='dashed', label=f'Mean Ensemble RMSE: {stack_mean:.3f}')

    plt.xlabel('Fold', fontsize=10)
    plt.ylabel('RMSE', fontsize=10)
    plt.legend(fontsize=10)

    plt.grid(True)
    
    plt.title('Cross-Validation Results for Different Models (RMSE)', fontsize=13)
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
plot_cv_results_rmse([cv_results_rf, cv_results_gb, cv_results_nn, cv_results_stacker], 
                     ['Random Forest', 'Gradient Boosting', 'Neural Network', 'Stacked Model'])
print("Script completed.")
