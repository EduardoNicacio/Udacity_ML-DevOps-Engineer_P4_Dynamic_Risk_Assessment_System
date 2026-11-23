"""
Reporting for the Dynamic Risk Assessment System.

Produces a confusion matrix for the deployed model on the configured test data
and, as a standout feature, a PDF diagnostic report bundling the matrix, the
deployed F1 score, summary statistics, missing data and the ingestion record.
"""

import json
import os

import matplotlib

matplotlib.use('Agg')  # non-interactive backend (Windows / headless friendly)

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402

import diagnostics  # noqa: E402


############### Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

TARGET = 'exited'


############## Function for reporting
def score_model_reporting(output_name='confusionmatrix.png'):
    """Build a confusion matrix for the deployed model and save it as a PNG.

    ``output_name`` lets the second (fullprocess) run write
    ``confusionmatrix2.png`` without overwriting the first-run artefact.
    """
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    predictions = diagnostics.model_predictions(test_df)
    actuals = test_df[TARGET].astype(int).tolist()

    matrix = confusion_matrix(actuals, predictions)

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                     display_labels=['Not Exited', 'Exited'])
    display.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Confusion Matrix - Attrition Risk Model')
    fig.tight_layout()

    os.makedirs(model_path, exist_ok=True)
    output_path = os.path.join(model_path, output_name)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Confusion matrix saved to {output_path}")
    return output_path


############## Standout: PDF diagnostic report
def generate_pdf_report(cm_image_path, output_name='model_report.pdf'):
    """Bundle the confusion matrix and diagnostics into a PDF report."""
    try:
        from fpdf import FPDF
    except ImportError:
        print("fpdf2 is not installed. Skipping PDF report (pip install fpdf2).")
        return None

    summary = diagnostics.dataframe_summary()
    missing = diagnostics.missing_data()

    with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
        f1_score = f.read().strip()

    with open(os.path.join(dataset_csv_path, 'ingestedfiles.txt'), 'r') as f:
        ingested = f.read().strip()

    pdf = FPDF()
    pdf.add_page()

    def heading(text):
        pdf.set_font('Helvetica', 'B', 12)
        pdf.multi_cell(0, 8, text, new_x='LMARGIN', new_y='NEXT')

    def line(text):
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 6, text, new_x='LMARGIN', new_y='NEXT')

    pdf.set_font('Helvetica', 'B', 16)
    pdf.multi_cell(0, 10, 'Attrition Risk Model - Diagnostic Report',
                   new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(4)

    heading(f'F1 score (deployed model): {f1_score}')
    pdf.ln(2)

    if cm_image_path and os.path.exists(cm_image_path):
        pdf.image(cm_image_path, x=30, w=150)
        pdf.ln(2)

    heading('Summary Statistics')
    for row in summary:
        line(f"  {row['column']} | {row['stat']}: {row['value']:.4f}")
    pdf.ln(2)

    heading('Missing Data (%)')
    for row in missing:
        line(f"  {row['column']}: {row['missing_pct']:.2f}%")
    pdf.ln(2)

    heading('Ingested Files')
    for ingested_file in ingested.splitlines():
        line(f'  {ingested_file}')

    os.makedirs(model_path, exist_ok=True)
    pdf_path = os.path.join(model_path, output_name)
    pdf.output(pdf_path)

    print(f"PDF report saved to {pdf_path}")
    return pdf_path


if __name__ == '__main__':
    matrix_path = score_model_reporting('confusionmatrix.png')
    generate_pdf_report(matrix_path)
