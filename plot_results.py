import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, matthews_corrcoef


def plot_results(y_pred_train, y_train, y_pred_test, y_test):
  fig = metrics.RocCurveDisplay.from_predictions(y_true=y_train, y_pred=y_pred_train, name="Train")
  metrics.RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred_test, name="Test", ax=fig.ax_)
  plt.show()

  print(classification_report(y_test, y_pred_test))

  print(f"Matthews Correlation Coefficient: {matthews_corrcoef(y_test, y_pred_test)}")


def plot_final_results():

  # Set the width of the bars
  barWidth = 0.7

  df = pd.DataFrame({
    'Name': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Logistic Regression': [0.76, 0.57, 0.75, 0.57],
    'Random Forest': [0.70, 0.57, 0.78, 0.54],
    'Neural Network': [0.72, 0.57, 0.77, 0.55],
    'Stacking Model': [0.75, 0.58, 0.78, 0.56],
  })
  colors = ['red', 'green', 'blue', 'yellow']

  df.plot(x="Name", y=['Logistic Regression', 'Random Forest', 'Neural Network', 'Stacking Model'],
          width=barWidth,
          color=colors,
          edgecolor='black',
          kind="bar")

  plt.ylim(0, 1)
  plt.xticks(rotation='horizontal')
  plt.xlabel('Metrics', fontweight='bold')
  plt.ylabel('Score', fontweight='bold')

  plt.legend(loc='lower right')
  plt.show()


def plot_final_mcc():

  # Matthews correlation coefficient values for the 4 models
  scores = [0.273, 0.282, 0.276, 0.290]
  subjects = ['Logistic Regression', 'Random Forest', 'Neural Network', 'Stacking Model']
  colors = ['red', 'green', 'blue', 'yellow']

  # Define the figure with size and y-limit values
  plt.figure(figsize=[10, 8])
  plt.ylim(0.25, 0.3)

  # Plot the scores
  plt.bar(subjects, scores, color=colors, edgecolor='black')

  # Add labels
  plt.title('Matthews Correlation Coefficient', fontweight='bold')
  plt.xlabel('Models', fontweight='bold')
  plt.ylabel('Scores', fontweight='bold')

  plt.show()
