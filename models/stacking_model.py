from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

from data.dataset import prepare_stroke_dataset
from plot_results import plot_results

# Load the dataset with label encoding
X_train, X_test, y_train, y_test = prepare_stroke_dataset(binary="label", categorical="label")

# Define the base models
base_models = [
  ('rf', RandomForestClassifier(
    random_state=42,
    n_estimators=100,
    criterion='gini',
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='auto',
    bootstrap=True
  )),
  ('mlp', MLPClassifier(
    random_state=42,
    hidden_layer_sizes=(4, 2),
    learning_rate='constant',
    batch_size=16,
    alpha=0.0001
  ))
]

# Initialize Stacking Classifier with the Meta Learner (Logistic Regression)
model = StackingClassifier(
  estimators=base_models,
  final_estimator=LogisticRegression()
)

# Create the parameter grid with hyperparameters most
# relevant to logistic regression
param_dist = {
  'final_estimator__C': [0.1, 0.5, 1, 5, 10],
  'final_estimator__penalty': ['l1', 'l2', 'elasticnet'],
  'final_estimator__class_weight': [None, 'balanced'],
  'final_estimator__solver': ['liblinear'],
  'final_estimator__max_iter': [100, 500]
}

# Instantiate the random search model with 5-fold cross-validation.
# This model will try 50 random combinations of hyperparameters
random_search = RandomizedSearchCV(
  estimator=model,
  n_iter=50,
  param_distributions=param_dist,
  cv=5,
  n_jobs=-1,
  scoring='f1_macro'
)

# Fit the grid search to the data
random_search.fit(X_train, y_train)

print(random_search.best_params_)

# Fit the model to the training set
model.fit(X_train, y_train)

# Make predictions on the training and testing sets
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Plot the final performance
plot_results(y_pred_train, y_train, y_pred_test, y_test)
