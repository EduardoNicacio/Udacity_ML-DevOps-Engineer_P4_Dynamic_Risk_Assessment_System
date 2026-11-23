"""
Flask API for the Dynamic Risk Assessment System.

Endpoints
---------
POST/GET /prediction     -> deployed model predictions for a supplied CSV
GET      /scoring        -> deployed model F1 score on the configured test data
GET      /summarystats   -> mean / median / std for numeric ingested columns
GET      /diagnostics    -> timings, missing data and dependency versions
"""

import json
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, request

import diagnostics
from scoring import score_model


###################### Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

prediction_model = None


####################### Prediction Endpoint
@app.route('/prediction', methods=['GET', 'POST', 'OPTIONS'])
def predict():
    """Load the CSV named in the request and predict with the deployed model."""
    filepath = ''

    if request.method == 'POST':
        body = request.get_json(silent=True) or {}
        filepath = body.get('filepath') or request.args.get('filepath', '')
    else:
        filepath = request.args.get('filepath', '')

    if not filepath:
        return jsonify({'error': 'filepath is required'}), 400

    if not os.path.exists(filepath):
        return jsonify({'error': f'file not found: {filepath}'}), 404

    dataframe = pd.read_csv(filepath)
    predictions = diagnostics.model_predictions(dataframe)
    return jsonify({'predictions': predictions}), 200


####################### Scoring Endpoint
@app.route('/scoring', methods=['GET', 'OPTIONS'])
def scoring_endpoint():
    """F1 score of the deployed model. Never rewrites the baseline score."""
    model_file = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    f1 = score_model(model, test_df)  # score_path=None on purpose
    return jsonify({'f1_score': f1}), 200


####################### Summary Statistics Endpoint
@app.route('/summarystats', methods=['GET', 'OPTIONS'])
def summarystats_endpoint():
    """Means, medians and standard deviations for numeric columns."""
    return jsonify({'summary_statistics': diagnostics.dataframe_summary()}), 200


####################### Diagnostics Endpoint
@app.route('/diagnostics', methods=['GET', 'OPTIONS'])
def diagnostics_endpoint():
    """Ingestion/training timings, missing values and dependency versions."""
    return jsonify({
        'execution_time': diagnostics.execution_time(),
        'missing_data': diagnostics.missing_data(),
        'dependency_check': diagnostics.outdated_packages_list(),
    }), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
