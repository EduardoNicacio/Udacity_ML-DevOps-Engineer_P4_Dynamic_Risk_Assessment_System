"""
Model scoring for the Dynamic Risk Assessment System.

``score_model`` always takes the model and the DataFrame explicitly and only
writes a score file when ``score_path`` is supplied. That keeps the API, the
drift check and the training/validation steps independent of each other.
"""

import json
import os
import pickle
from datetime import datetime

import pandas as pd
from sklearn.metrics import f1_score

import dbsetup


################# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])

FEATURES = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
TARGET = 'exited'


################# Function for model scoring
def score_model(model, dataframe, score_path=None):
    """Calculate the supplied model's F1 score on the supplied DataFrame.

    Parameters
    ----------
    model : fitted sklearn estimator
    dataframe : pandas.DataFrame with FEATURES and TARGET columns
    score_path : str or None
        When provided, the F1 score is written to this path. API requests and
        drift checks pass ``None`` so the deployed baseline is never touched.
    """
    X = dataframe[FEATURES]
    y = dataframe[TARGET]

    predictions = model.predict(X)
    f1 = float(f1_score(y, predictions, zero_division=0))

    if score_path is not None:
        directory = os.path.dirname(score_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(score_path, 'w') as f:
            f.write(str(f1))

        _record_score(f1, score_path)

    return f1


def _record_score(f1, score_path):
    """Log a persisted score in SQLite (standout feature, fail-safe)."""
    try:
        dbsetup.init_db()
        model_file = os.path.join(os.path.dirname(score_path) or '.', 'trainedmodel.pkl')
        with dbsetup.get_connection() as connection:
            connection.execute(
                'INSERT INTO model_scores (f1_score, scored_at, model_path) VALUES (?, ?, ?)',
                (f1, datetime.now().isoformat(timespec='seconds'), model_file),
            )
    except Exception as exc:  # never let bookkeeping break scoring
        print(f"Warning: could not log score to database: {exc}")


if __name__ == '__main__':
    trained_model_path = os.path.join(model_path, 'trainedmodel.pkl')
    with open(trained_model_path, 'rb') as f:
        model = pickle.load(f)

    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    score_path = os.path.join(model_path, 'latestscore.txt')
    result = score_model(model, test_df, score_path=score_path)
    print(f"F1 score: {result}")
