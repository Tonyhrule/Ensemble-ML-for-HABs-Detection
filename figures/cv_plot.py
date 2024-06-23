import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures

# Load the data
output_dir = 'output'
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(os.path.join(output_dir, 'processed_data.pkl'))

# Load the scaler and models
models_dir = 'models'
scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
rf_best = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
gb_best = joblib.load(os.path.join(models_dir, 'gb_model.pkl'))
nn_best = joblib.load(os.path.join(models_dir, 'nn_model.pkl'))
stacker_best = joblib.load(os.path.join(models_dir, 'stacker_model.pkl'))

# Recreate the polynomial features transformer with the same degree as used in training
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)

def plot_cv_results(cv_results, model_names):
    plt.figure(figsize=(10, 6))
    for model_name, results in zip(model_names, cv_results):
        plt.plot(range(1, len(results) + 1), -results, label=model_name, marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Mean Squared Error')
    plt.title('Cross-Validation Results')
    plt.legend()
    plt.grid(True)
    plt.show()

# Cross-validation setup
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

# Perform cross-validation and collect results
cv_results_rf = cross_val_score(rf_best, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
cv_results_gb = cross_val_score(gb_best, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
cv_results_nn = cross_val_score(nn_best, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
cv_results_stacker = cross_val_score(stacker_best, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')

# Plot the cross-validation results
plot_cv_results([cv_results_rf, cv_results_gb, cv_results_nn, cv_results_stacker], 
                ['Random Forest', 'Gradient Boosting', 'Neural Network', 'Stacked Model'])
