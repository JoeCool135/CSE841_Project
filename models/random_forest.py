from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from data.dataset import prepare_stroke_dataset
from plot_results import plot_results

# Load the dataset with label encoding
X_train, X_test, y_train, y_test = prepare_stroke_dataset(binary="label", categorical="label")

# Create the parameter grid with hyperparameters most
# relevant to a random forest
param_grid = {
  'n_estimators': [100, 200, 500],
  "criterion": ['gini', 'entropy', 'log_loss'],
  'max_depth': [1, 2, 3],
  'min_samples_split': [2, 4, 6],
  'min_samples_leaf': [1, 3, 4],
  'max_features': ['auto', 'sqrt', 'log2'],
  "bootstrap": [True, False]
}

# Instantiate the grid search model with 5-fold cross-validation
grid_search = GridSearchCV(
  estimator=RandomForestClassifier(random_state=42),
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
model = RandomForestClassifier(
  random_state=42,
  n_estimators=best_params['n_estimators'],
  criterion=best_params['criterion'],
  max_depth=best_params['max_depth'],
  min_samples_split=best_params['min_samples_split'],
  min_samples_leaf=best_params['min_samples_leaf'],
  max_features=best_params['max_features'],
  bootstrap=best_params['bootstrap']
)

# Fit the model to the training set
model.fit(X_train, y_train)

# Make predictions on the training and testing sets
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Plot the final performance
plot_results(y_pred_train, y_train, y_pred_test, y_test)

