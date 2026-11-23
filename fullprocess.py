"""
End-to-end automation for the Dynamic Risk Assessment System.

Flow
----
1. Compare the input folder with the deployed ingestion record.
   No new files -> stop immediately (no ingestion, no reporting).
2. Ingest all data in the input folder.
3. Score the *deployed* model on the newly ingested data (no file written).
4. Retrain and redeploy only when the new F1 is lower than the baseline.
5. Always run reporting (confusionmatrix2.png) and the API calls
   (apireturns2.txt) against the model that is deployed after step 4.
"""

import glob
import json
import os
import pickle
import subprocess
import sys

import pandas as pd

# Import modules from the project
import deployment
import diagnostics
import ingestion
import reporting
import scoring
import training

with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']
prod_deployment_path = config['prod_deployment_path']
test_data_path = config['test_data_path']


################## Check and read new data
deployed_ingestion_record = os.path.join(prod_deployment_path, 'ingestedfiles.txt')

if not os.path.exists(deployed_ingestion_record):
    raise SystemExit(
        f"Missing {deployed_ingestion_record}. Run the initial ingestion, training, "
        "scoring and deployment steps before fullprocess.py."
    )

with open(deployed_ingestion_record, 'r') as f:
    previously_ingested = [line.strip() for line in f if line.strip()]

# Compare by basename so the check is independent of slash direction and of
# whether the record holds Windows or POSIX paths.
previously_ingested_names = {os.path.basename(path) for path in previously_ingested}

available_files = sorted(glob.glob(os.path.join(input_folder_path, '*.csv')))
new_files = [path for path in available_files
             if os.path.basename(path) not in previously_ingested_names]


################## Deciding whether to proceed, part 1
if not new_files:
    print('No new data found. Pipeline stopping: no ingestion, retraining or reporting.')
    raise SystemExit(0)

print(f'New files detected in {input_folder_path}: {new_files}')

# Ingest to update finaldata.csv and the local ingestedfiles.txt
ingestion.merge_multiple_dataframe()


################## Checking for model drift
baseline_score_path = os.path.join(prod_deployment_path, 'latestscore.txt')
with open(baseline_score_path, 'r') as f:
    baseline_score = float(f.read().strip())

deployed_model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
with open(deployed_model_path, 'rb') as f:
    deployed_model = pickle.load(f)

# Drift is measured on the newly ingested data, never on testdata.csv.
new_data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
new_score = scoring.score_model(deployed_model, new_data)  # score_path=None

print(f'Baseline F1 (deployed model): {baseline_score:.6f}')
print(f'New F1 (deployed model on ingested data): {new_score:.6f}')


################## Deciding whether to proceed, part 2 / Re-deployment
if new_score < baseline_score:
    print('Model drift detected: retraining and redeploying.')

    training.train_model()

    # The replacement is scored on the configured test data, not on the new
    # ingested data, and that score becomes the new baseline.
    replacement_model_path = os.path.join(output_model_path, 'trainedmodel.pkl')
    with open(replacement_model_path, 'rb') as f:
        replacement_model = pickle.load(f)

    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    replacement_score_path = os.path.join(output_model_path, 'latestscore.txt')
    replacement_score = scoring.score_model(replacement_model, test_df,
                                            score_path=replacement_score_path)

    deployment.store_model_into_pickle()
    print(f'Replacement model deployed (test-data F1 = {replacement_score:.6f}).')
else:
    print('No drift: keeping the deployed model and baseline score unchanged.')


################## Diagnostics and reporting
# Runs in both drift and no-drift cases, after any redeployment, using the
# model that is deployed at this point. The "2" suffix preserves the first-run
# confusionmatrix.png and apireturns.txt artefacts.
print('Generating second-run reports against the currently deployed model...')

reporting.score_model_reporting('confusionmatrix2.png')

try:
    subprocess.run([sys.executable, 'apicalls.py', '--second-run'], check=True)
except subprocess.CalledProcessError:
    # Most likely cause: app.py is not listening on port 8000 (e.g. a cron
    # run). Fail with an actionable message instead of a raw traceback; the
    # confusion matrix above has already been written.
    raise SystemExit(
        'ERROR: apicalls.py failed, so apireturns2.txt was NOT generated. '
        'Start the API first (python app.py) so it listens on port 8000, '
        'then re-run fullprocess.py. confusionmatrix2.png is already saved.'
    )

print('Full process complete.')
