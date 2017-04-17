import pandas as pd
import tensorflow as tf


# Create references to the training and testing files
training_file = 'rft-teaching-file/training-file.csv'
testing_file = 'rft-teaching-file/testing-file.csv'


COLUMNS = ["person_id", "region", "residence_type", "family_composition", "population_base",
           "sex", "age", "marital_status", "student",
           "country_of_birth", "health", "ethnic_group", "religion",
           "economic_activity", "occupation", "industry", "hours_worked","social_grade"]

# All the data is categorical 
CATEGORICAL_COLUMNS = ["region", "residence_type", "family_composition", "population_base",
           "sex", "age", "marital_status", "student",
           "country_of_birth", "health", "ethnic_group", "religion",
           "economic_activity", "occupation", "industry", "hours_worked","social_grade"]


LABEL_COLUMN = "label"

tf_train = pd.read_csv(training_file, names=COLUMNS, skipinitialspace=True)
tf_test = pd.read_csv(testing_file, names=COLUMNS, skipinitialspace=True, skiprows=1)



def input_fn(df):

  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(tf_train)

def eval_input_fn():
  return input_fn(tf_test)
