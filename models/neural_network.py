from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from data.dataset import prepare_stroke_dataset
from plot_results import plot_results

# Load the dataset with label encoding
X_train, X_test, y_train, y_test = prepare_stroke_dataset(binary="label", categorical="label")

# Create the parameter grid with hyperparameters most
# relevant to a neural network
param_grid = {
  'hidden_layer_sizes': [(4, 2)],
  'activation': ['tanh', 'relu'],
  'alpha': [0.0001, 0.05],
  'batch_size': [16, 32, 64, 'auto'],
  'learning_rate': ['constant', 'adaptive']
}

# Instantiate the grid search model with 5-fold cross-validation
grid_search = GridSearchCV(
  estimator=MLPClassifier(random_state=42),
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
model = MLPClassifier(
  random_state=42,
  hidden_layer_sizes=best_params['hidden_layer_sizes'],
  batch_size=best_params['batch_size'],
  learning_rate=best_params['learning_rate'],
  alpha=best_params['alpha']
)

# Fit the model to the training set
model.fit(X_train, y_train)

# Make predictions on the training and testing sets
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Plot the final performance
plot_results(y_pred_train, y_train, y_pred_test, y_test)