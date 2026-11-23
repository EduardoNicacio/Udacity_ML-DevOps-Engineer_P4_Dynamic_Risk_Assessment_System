"""
Deployment for the Dynamic Risk Assessment System.

Copies the trained model, its score and the ingestion record into
``prod_deployment_path`` so serving and drift detection both use the exact
same artefacts.
"""

import json
import os
import shutil


################## Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


#################### Function for deployment
def store_model_into_pickle():
    """Copy trainedmodel.pkl, latestscore.txt and ingestedfiles.txt to prod."""
    os.makedirs(prod_deployment_path, exist_ok=True)

    artefacts = [
        (os.path.join(model_path, 'trainedmodel.pkl'), 'trainedmodel.pkl'),
        (os.path.join(model_path, 'latestscore.txt'), 'latestscore.txt'),
        (os.path.join(dataset_csv_path, 'ingestedfiles.txt'), 'ingestedfiles.txt'),
    ]

    for source, name in artefacts:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Missing deployment artefact: {source}")
        shutil.copy2(source, os.path.join(prod_deployment_path, name))

    print(f"Deployed model, score and ingestion record to {prod_deployment_path}")


if __name__ == '__main__':
    store_model_into_pickle()
