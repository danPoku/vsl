# Insurance Brokerage Analytics Pipeline

This repository contains a dataset (`visal_data_cleaned.csv`) with insurance policy records. The `pipeline.py` script provides a simple machine learning pipeline for analyzing these records and can be re-used with similar datasets in the future.

## Features

- **Data loading** – reads a CSV file and parses dates.
- **Metric calculation** – reports totals, averages, payment status counts and the top businesses with outstanding balances.
- **Payment default prediction** – logistic regression model forecasts payment status and default risk.
- **Client segmentation** – clusters clients based on premium and balance behaviour to estimate lifetime commission value.
- **Premium regression** – linear model suggesting premium levels given exposures and brokerage factors.
- **Risk exposure summary** – aggregates exposures by business line and currency.
- **Anomaly detection** – flags suspicious premium or commission entries using z-scores.

## Requirements

- Python 3.12+
- `pandas` and `scikit-learn`

Install the requirements with:

```bash
pip install pandas scikit-learn
```

## Usage

Run the analytics pipeline using:

```bash
python3 pipeline.py --data visal_data_cleaned.csv
```

This prints computed metrics followed by model accuracy and a classification report.

Additional analyses such as client segmentation, premium regression, risk exposure summaries and anomaly detection are also printed. Future datasets with the same column structure can be provided via the `--data` argument for analysis.

