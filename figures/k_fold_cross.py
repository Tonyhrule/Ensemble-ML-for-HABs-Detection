import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.model_selection import KFold
from matplotlib import pyplot
from sklearn.model_selection import LeaveOneOut

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

# Create polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Define cross-validation strategy

cv = LeaveOneOut()


rf_means = []
gb_means = []
nn_means = []
rf_mins = []
gb_mins = []
nn_mins = []
rf_maxes = []
gb_maxes = []
nn_maxes = []
 # define the test condition
cv = KFold(n_splits=10, shuffle=True, random_state=1)
rf_ideal = cross_val_score(rf_best, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
gb_ideal = cross_val_score(gb_best, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
nn_ideal = cross_val_score(nn_best, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)


rf_idea = -rf_ideal
gb_ideal = -gb_ideal
nn_ideal = -nn_ideal

rf_mean = np.mean(rf_ideal)
gb_mean = np.mean(gb_ideal)
nn_mean = np.mean(nn_ideal)

# evaluate k value
folds = range(2,31)
for k in folds:
    cv = KFold(n_splits=k, shuffle=True, random_state=1)
    # Perform cross-validation with regression scoring metric
    rf_scores = cross_val_score(rf_best, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    gb_scores = cross_val_score(gb_best, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    nn_scores = cross_val_score(nn_best, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

    # Convert negative MSE to positive for interpretability
    rf_scores = -rf_scores
    gb_scores = -gb_scores
    nn_scores = -nn_scores

    rf_mean = np.mean(rf_scores)
    gb_mean = np.mean(gb_scores)
    nn_mean = np.mean(nn_scores)

    rf_min = min(rf_scores)
    gb_min = min(gb_scores)
    nn_min = min(nn_scores)

    rf_max = max(rf_scores)
    gb_max = max(gb_scores)
    nn_max = max(nn_scores)

    # store mean accuracy
    rf_means.append(rf_mean)
    gb_means.append(gb_mean)
    nn_means.append(nn_mean)
    # store min and max relative to the mean
    rf_mins.append(rf_mean - rf_min)
    gb_mins.append(gb_mean - gb_min)
    nn_mins.append(nn_mean - nn_min)

    rf_maxes.append(rf_max - rf_mean)
    gb_maxes.append(gb_max - gb_mean)
    nn_maxes.append(nn_max - nn_mean)

    print(f"{k}/31")
    # line plot of k mean values with min/max error bars
pyplot.errorbar(folds, rf_means, yerr=[rf_mins, rf_maxes], fmt='o')
pyplot.plot(folds, [rf_ideal for _ in range(len(folds))], color='r')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.title('K Fold Cross-Validation Results Random Forest')
pyplot.show()
pyplot.errorbar(folds, gb_means, yerr=[gb_mins, gb_maxes], fmt='o')
pyplot.plot(folds, [gb_ideal for _ in range(len(folds))], color='r')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.title('K Fold Cross-Validation Results Gradient Boosting Model')
pyplot.show()
pyplot.errorbar(folds, rf_means, yerr=[nn_mins, nn_maxes], fmt='o')
pyplot.plot(folds, [nn_ideal for _ in range(len(folds))], color='r')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.title('K Fold Cross-Validation Results Neural Network')
pyplot.show()
# plot the ideal case in a separate color