"""
Call every API endpoint, combine the responses and save them to the workspace.

``python apicalls.py``               -> apireturns.txt
``python apicalls.py --second-run``  -> apireturns2.txt

The Flask app (app.py) must be listening on port 8000.
"""

import json
import os
import sys

import requests

# Specify a URL that resolves to your workspace
URL = 'http://127.0.0.1:8000'

with open('config.json', 'r') as f:
    config = json.load(f)

output_model_path = os.path.join(config['output_model_path'])


def call_all_endpoints(output_suffix='', timeout=300):
    """Call all four endpoints and write the combined payload."""
    scoring_response = requests.get(f'{URL}/scoring', timeout=timeout)
    scoring_response.raise_for_status()

    prediction_response = requests.post(
        f'{URL}/prediction',
        json={'filepath': 'testdata/testdata.csv'},
        timeout=timeout,
    )
    prediction_response.raise_for_status()

    summary_response = requests.get(f'{URL}/summarystats', timeout=timeout)
    summary_response.raise_for_status()

    diagnostics_response = requests.get(f'{URL}/diagnostics', timeout=timeout)
    diagnostics_response.raise_for_status()

    responses = {
        'scoring': scoring_response.json(),
        'prediction': prediction_response.json(),
        'summarystats': summary_response.json(),
        'diagnostics': diagnostics_response.json(),
    }

    os.makedirs(output_model_path, exist_ok=True)
    output_path = os.path.join(output_model_path, f'apireturns{output_suffix}.txt')
    with open(output_path, 'w') as f:
        json.dump(responses, f, indent=2)

    print(f"API responses written to {output_path}")
    return responses


if __name__ == '__main__':
    suffix = ''
    if len(sys.argv) > 1 and sys.argv[1] == '--second-run':
        suffix = '2'

    call_all_endpoints(output_suffix=suffix)
