"""
Diagnostics for the Dynamic Risk Assessment System.

Provides model predictions, summary statistics, missing-value percentages,
ingestion/training timings and a dependency version table.
"""

import json
import os
import pickle
import subprocess
import sys
import timeit
from datetime import datetime

import pandas as pd

import dbsetup


################## Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

FEATURES = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
TARGET = 'exited'


################## Function to get model predictions
def model_predictions(dataframe):
    """Predict with the model currently deployed in production."""
    model_file = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Select the same predictor columns used for training; never the target.
    predictions = model.predict(dataframe[FEATURES])
    return [int(value) for value in predictions]


################## Function to get summary statistics
def dataframe_summary():
    """Mean, median and standard deviation for every numeric column."""
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    numeric_columns = df.select_dtypes(include='number').columns.tolist()

    statistics = []
    for column in numeric_columns:
        statistics.append({'column': column, 'stat': 'mean', 'value': float(df[column].mean())})
        statistics.append({'column': column, 'stat': 'median', 'value': float(df[column].median())})
        statistics.append({'column': column, 'stat': 'std', 'value': float(df[column].std())})

    return statistics


################## Function to check missing data
def missing_data():
    """Percentage of missing values for every column of the ingested dataset."""
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    rows = len(df)

    missing = []
    for column in df.columns:
        percentage = 0.0 if rows == 0 else float(df[column].isna().sum() / rows * 100)
        missing.append({'column': column, 'missing_pct': percentage})

    return missing


################## Function to get timings
def execution_time():
    """Time ingestion.py followed by training.py.

    Each script runs in its own interpreter (``sys.executable``) so timing is
    measured in isolation. The subprocesses write to ``output_folder_path`` /
    ``output_model_path`` only; the deployed artefacts are never touched.
    """
    timings = []

    for script in ('ingestion.py', 'training.py'):
        start = timeit.default_timer()
        subprocess.run([sys.executable, script], check=True, capture_output=True)
        elapsed = timeit.default_timer() - start
        timings.append(float(elapsed))
        print(f"  {script}: {elapsed:.4f}s")

    _record_diagnostics(timings)

    return timings


def _record_diagnostics(timings):
    """Log timings and missing-data snapshot in SQLite (fail-safe standout)."""
    try:
        dbsetup.init_db()
        with dbsetup.get_connection() as connection:
            connection.execute(
                'INSERT INTO diagnostics_log '
                '(run_at, ingestion_time, training_time, missing_data_json) '
                'VALUES (?, ?, ?, ?)',
                (
                    datetime.now().isoformat(timespec='seconds'),
                    timings[0],
                    timings[1],
                    json.dumps(missing_data()),
                ),
            )
    except Exception as exc:  # never let bookkeeping break diagnostics
        print(f"Warning: could not log diagnostics to database: {exc}")


################## Function to check dependencies
def outdated_packages_list():
    """Report installed vs. latest version for every requirement.

    Uses ``pip list --outdated`` / ``pip list`` so the check works on Windows
    and inside WSL without importing packaging internals. Nothing is upgraded.
    """
    def _pip_json(*args):
        result = subprocess.run([sys.executable, '-m', 'pip', *args],
                                capture_output=True, text=True)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

    outdated = {entry['name'].lower(): entry for entry in _pip_json('list', '--outdated', '--format=json')}
    installed = {entry['name'].lower(): entry['version'] for entry in _pip_json('list', '--format=json')}

    requirements_file = 'requirements.txt'
    if not os.path.exists(requirements_file):
        return []

    packages = []
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name = line
            for separator in ('=', '>', '<', '~', '!'):
                name = name.split(separator)[0]
            packages.append(name.strip())

    report = []
    for package in packages:
        key = package.lower()
        current = installed.get(key, 'N/A')
        latest = outdated[key]['latest_version'] if key in outdated else current
        report.append({
            'package': package,
            'current_version': current,
            'latest_version': latest,
        })

    return report


if __name__ == '__main__':
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    print("=== Predictions (deployed model, testdata) ===")
    print(model_predictions(test_df))

    print("\n=== Summary Statistics ===")
    for row in dataframe_summary():
        print(f"  {row['column']:22s} {row['stat']:7s} {row['value']:.4f}")

    print("\n=== Missing Data ===")
    for row in missing_data():
        print(f"  {row['column']:22s} {row['missing_pct']:.2f}%")

    print("\n=== Execution Times ===")
    timings = execution_time()
    print(f"  ingestion.py : {timings[0]:.4f}s")
    print(f"  training.py  : {timings[1]:.4f}s")

    print("\n=== Dependency Check ===")
    for row in outdated_packages_list():
        print(f"  {row['package']:20s} {row['current_version']:12s} {row['latest_version']}")
