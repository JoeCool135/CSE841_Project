from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from data.dataset import prepare_stroke_dataset
from plot_results import plot_results

# Load the dataset with label encoding
X_train, X_test, y_train, y_test = prepare_stroke_dataset(binary="label", categorical="label")

# Create the parameter grid with hyperparameters most
# relevant to logistic regression
param_grid = {
  'penalty': ['l1', 'l2', 'elasticnet'],
  'C': [0.1, 0.5, 1, 5, 10],
  'class_weight': [None, 'balanced'],
  'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
  'max_iter': [100, 500, 1000]
}

# Instantiate the grid search model with 5-fold cross-validation
grid_search = GridSearchCV(
  estimator=LogisticRegression(random_state=42),
  param_grid=param_grid,
  cv=5,
  n_jobs=-1,
  scoring=['accuracy', 'f1_macro'],
  refit='f1_macro'
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Retrieve the optimal hyperparameter combination
best_params = grid_search.best_params_

print(best_params)

# Instantiate the model with the best hyperparameters
model = LogisticRegression(
  random_state=42,
  penalty=best_params['penalty'],
  C=best_params['C'],
  class_weight=best_params['class_weight'],
  solver=best_params['solver'],
  max_iter=best_params['max_iter']
)

# Fit the model to the training set
model.fit(X_train, y_train)

# Make predictions on the training and testing sets
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Plot the final performance
plot_results(y_pred_train, y_train, y_pred_test, y_test)
