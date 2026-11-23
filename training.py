"""
Model training for the Dynamic Risk Assessment System.

Loads the ingested dataset, fits the LogisticRegression specified by the
project template, and pickles the fitted model to ``trainedmodel.pkl``.
"""

import json
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression


################### Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])

FEATURES = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
TARGET = 'exited'


################# Function for training the model
def train_model():
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    X = df[FEATURES]
    y = df[TARGET]

    # use this logistic regression for training
    # NOTE: ``multi_class`` was removed in scikit-learn 1.8 (it was already
    # the default behaviour), so it is intentionally omitted here.
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               n_jobs=None, penalty='l2', random_state=0,
                               solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    # Fit the logistic regression to your data
    model.fit(X, y)

    # Write the trained model to your workspace in a file called trainedmodel.pkl
    os.makedirs(model_path, exist_ok=True)
    trained_model_path = os.path.join(model_path, 'trainedmodel.pkl')
    with open(trained_model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model trained on {len(df)} rows and saved to {trained_model_path}")


if __name__ == '__main__':
    train_model()
