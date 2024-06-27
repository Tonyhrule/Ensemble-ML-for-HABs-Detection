import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(output_dir):
    logging.info(f"Loading data from {output_dir}")
    return joblib.load(os.path.join(output_dir, 'processed_data.pkl'))

def save_model(model, model_name, models_dir):
    joblib.dump(model, os.path.join(models_dir, f'{model_name}_model.pkl'))
    logging.info(f'{model_name} model saved.')

def hyperparameter_tuning(model, param_grid, X_train, y_train, random_search=False):
    logging.info(f"Starting hyperparameter tuning for {model.__class__.__name__}")
    if random_search:
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5, n_jobs=-1, verbose=2, random_state=42)
    else:
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    search.fit(X_train, y_train)
    logging.info(f"Best parameters for {model.__class__.__name__}: {search.best_params_}")
    return search.best_estimator_

def plot_cv_results(cv_results, model_names):
    plt.figure(figsize=(10, 6))
    for i, (model_name, results) in enumerate(zip(model_names, cv_results)):
        plt.plot(range(1, len(results) + 1), -results, label=model_name, marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Mean Squared Error')
    plt.title('Cross-Validation Results')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_residuals(models, X_test, y_test):
    plt.figure(figsize=(14, 8))
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        plt.subplot(2, 2, i + 1)
        plt.scatter(y_test, residuals, s=10, alpha=0.7)
        plt.hlines(0, min(y_test), max(y_test), colors='r', linestyles='dashed')
        plt.xlabel('Actual')
        plt.ylabel('Residuals')
        plt.title(f'{name} Residuals')
    plt.tight_layout()
    plt.show()

def main():
    output_dir = 'output'
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_data(output_dir)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_scaled = imputer.fit_transform(X_train_scaled)
    X_test_scaled = imputer.transform(X_test_scaled)
    y_train = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test = imputer.transform(y_test.values.reshape(-1, 1)).ravel()

    # Polynomial Features for better feature engineering
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    # Hyperparameter Tuning for Random Forest
    rf = RandomForestRegressor(random_state=42)
    param_grid_rf = {
        'n_estimators': [50, 100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_best = hyperparameter_tuning(rf, param_grid_rf, X_train_poly, y_train, random_search=True)
    save_model(rf_best, 'rf', models_dir)

    # Hyperparameter Tuning for Gradient Boosting
    gb = GradientBoostingRegressor(random_state=42)
    param_grid_gb = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6]
    }
    gb_best = hyperparameter_tuning(gb, param_grid_gb, X_train_poly, y_train, random_search=True)
    save_model(gb_best, 'gb', models_dir)

    # Hyperparameter Tuning for Neural Network
    nn = MLPRegressor(random_state=42, early_stopping=True)
    param_grid_nn = {
        'hidden_layer_sizes': [(50,), (100,), (200,), (100, 100), (200, 200)],
        'learning_rate_init': [0.001, 0.01, 0.05, 0.1],
        'max_iter': [500, 1000]
    }
    nn_best = hyperparameter_tuning(nn, param_grid_nn, X_train_poly, y_train, random_search=True)
    save_model(nn_best, 'nn', models_dir)

    # Create Stacking Regressor
    estimators = [
        ('rf', rf_best),
        ('gb', gb_best),
        ('nn', nn_best)
    ]
    stacker = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
    stacker.fit(X_train_poly, y_train)
    save_model(stacker, 'stacker', models_dir)

    # Plotting the training losses (cross-validation results)
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
    cv_results_rf = cross_val_score(rf_best, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
    cv_results_gb = cross_val_score(gb_best, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
    cv_results_nn = cross_val_score(nn_best, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
    cv_results_stacker = cross_val_score(stacker, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')

    plot_cv_results([cv_results_rf, cv_results_gb, cv_results_nn, cv_results_stacker], 
                    ['Random Forest', 'Gradient Boosting', 'Neural Network', 'Stacked Regressor'])

    # Plotting residuals
    models = {
        'Random Forest': rf_best,
        'Gradient Boosting': gb_best,
        'Neural Network': nn_best,
        'Ensemble': stacker
    }
    plot_residuals(models, X_test_poly, y_test)

if __name__ == "__main__":
    main()
