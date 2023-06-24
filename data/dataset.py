import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

encoder = LabelEncoder()
scaler = StandardScaler()


# Mean impute a column  in a dataframe
def mean_impute(df, col_name):
  mean = df[col_name].mean()
  df[col_name] = df[col_name].fillna(mean)
  return df


# Label encode a column in a dataframe
def label_encode(df, col_name):
  df[col_name] = encoder.fit_transform(df[col_name])
  return df


# One-hot encode a column in a dataframe
def one_hot_encode(df, col_name):
  one_hot = pd.get_dummies(df[col_name])
  df = df.drop(columns=col_name)
  df = df.join(one_hot)
  return df


# Categorize a column, then label encode
def categorize_ordinal_encode(df, col_name, bins, labels):
  mapping = {label: num for num, label in enumerate(labels)}
  df[col_name] = pd.cut(df[col_name], bins=bins, labels=labels)
  df[col_name] = df[col_name].replace(mapping)
  return df


# Prepare the stroke detection dataset
# binary: Type of encoding to perform on binary attributes
# categorical: Type of encoding to perform on categorical attributes
# numerical: Type of encoding to perform on numerical attributes
def prepare_stroke_dataset(binary=None, categorical=None, numerical=None):

  # Load the data
  data = pd.read_csv('../data/healthcare-dataset-stroke-data.csv')

  # Remove the id attribute, as it does not provide valuable information
  data = data.drop(["id"], axis=1)

  # Prepare the gender column:
  # 1. Remove the row with "Other", as it only occurs once
  # 2. Label Encode, or one-hot encode
  data = data[data.gender != 'Other']

  if binary == "label":
    data = label_encode(data, 'gender')
  elif binary == "onehot":
    data = one_hot_encode(data, 'gender')

  # Prepare the age column:
  # 1. Categorize into bins and ordinal encode
  if numerical == "ordinal":
    age_bins = [-np.inf, 13, 20, 36, 60, np.inf]
    age_labels = ['Child', 'Teen', 'Young Adult', 'Middle-Aged', 'Senior']
    data = categorize_ordinal_encode(data, 'age', age_bins, age_labels)

  # Prepare the ever married column:
  # 1. Label Encode, or one-hot encode
  if binary == "label":
    data = label_encode(data, 'ever_married')
  elif binary == "onehot":
    data = one_hot_encode(data, 'ever_married')

  # Prepare the work type column:
  # 1. One-Hot Encode, or label encode
  if categorical == "onehot":
    data = one_hot_encode(data, 'work_type')
  elif categorical == "label":
    data = label_encode(data, 'work_type')

  # Prepare the residence type column:
  # 1. Label Encode, or one-hot encode
  if binary == "label":
    data = label_encode(data, 'Residence_type')
  elif binary == "onehot":
    data = one_hot_encode(data, 'Residence_type')

  # Prepare the avg glucose level column:
  # 1. Categorize into bins and ordinal encode
  if numerical == "ordinal":
    avg_glucose_bins = [-np.inf, 100, 125, np.inf]
    avg_glucose_labels = ['Normal', 'Prediabetes', 'Diabetes']
    data = categorize_ordinal_encode(data, 'avg_glucose_level', avg_glucose_bins, avg_glucose_labels)

  # Prepare the bmi column:
  # 1. Mean impute to fill in rows with missing values
  # 2. Categorize into bins and ordinal encode
  data = mean_impute(data, 'bmi')

  if numerical == "ordinal":
    bmi_bins = [-np.inf, 18.5, 24.9, 29.9, np.inf]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    data = categorize_ordinal_encode(data, 'bmi', bmi_bins, bmi_labels)

  # Prepare the smoking status column:
  # 1. One-Hot Encode, or label encode
  if categorical == "onehot":
    data = one_hot_encode(data, 'smoking_status')
  elif categorical == "label":
    data = label_encode(data, 'smoking_status')

  # Split the data into features and targets
  X = data.drop('stroke', axis=1)
  y = data['stroke']

  # Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Standardize the features
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  # Use SMOTE to balance the training set
  sm = SMOTE(random_state=42)
  X_train, y_train = sm.fit_resample(X_train, y_train)

  return X_train, X_test, y_train, y_test

