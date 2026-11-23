"""
Data ingestion for the Dynamic Risk Assessment System.

Reads every CSV file found in ``input_folder_path``, concatenates them, removes
duplicate rows, and writes ``finaldata.csv`` plus ``ingestedfiles.txt`` to
``output_folder_path``.

File names are auto-discovered with ``glob`` so that adding new data to the
input folder (``practicedata``, ``sourcedata``, the validation folders, ...)
requires no code changes.
"""

import glob
import json
import os
from datetime import datetime

import pandas as pd

import dbsetup


############# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


############# Function for data ingestion
def merge_multiple_dataframe():
    """Combine every CSV in the input folder, drop duplicates and save."""
    csv_files = sorted(glob.glob(os.path.join(input_folder_path, '*.csv')))

    if not csv_files:
        # Fail loudly: silently writing an empty ingestedfiles.txt would leave
        # the ingestion record inconsistent with the (stale) finaldata.csv.
        raise FileNotFoundError(
            f"No CSV files found in input folder: {input_folder_path}"
        )

    frames = [pd.read_csv(path) for path in csv_files]
    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(inplace=True)

    os.makedirs(output_folder_path, exist_ok=True)
    combined.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)

    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        for path in csv_files:
            f.write(path + '\n')

    _record_ingestion(csv_files)

    print(f"Ingested {len(csv_files)} file(s), {len(combined)} unique row(s) "
          f"-> {os.path.join(output_folder_path, 'finaldata.csv')}")


def _record_ingestion(csv_files):
    """Log the ingested files in SQLite (standout feature, fail-safe)."""
    try:
        dbsetup.init_db()
        timestamp = datetime.now().isoformat(timespec='seconds')
        with dbsetup.get_connection() as connection:
            connection.executemany(
                'INSERT INTO ingested_files (filename, ingested_at) VALUES (?, ?)',
                [(path, timestamp) for path in csv_files],
            )
    except Exception as exc:  # never let bookkeeping break ingestion
        print(f"Warning: could not log ingestion to database: {exc}")


if __name__ == '__main__':
    merge_multiple_dataframe()
