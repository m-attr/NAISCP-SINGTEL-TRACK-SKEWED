# NAISC-SingtelTrack-DataDrift

Simple launch guide for the NAISC Singtel 2026 pipeline and dashboard.

## 1) Prerequisites

- Python 3.12
- CPU runtime 

## 2) Install Dependencies

From repo root:

```bash
pip install -r requirements.txt
```

## 3) Run The Pipeline

Black-box command format:

```bash
python ./src/main.py --train_data_filepath <train_data_filepath> --test_data_filepath <test_data_filepath>
```

Local example:

```bash
python ./src/main.py --train_data_filepath ./train.csv --test_data_filepath ./test.csv
```

- Runs drift detection and mitigation.
- Trains/predicts with the fixed LightGBM setup.
- Writes output artifacts in repo root.

## 4) Output Files

After a successful pipeline run, expect at least:

- prediction.csv
- latest_metrics.json

prediction.csv contains exactly 2 columns:

- CustomerID
- probability_score

## 5) Launch The Dashboard

Run from repo root:

```bash
streamlit run ./src/app.py
```

Then open the local URL shown by Streamlit (typically http://localhost:8501).

The dashboard reads pipeline artifacts (such as latest_metrics.json and drift_report.json) from the workspace.

## 6) Quick Troubleshooting

- If command not found for streamlit:

```bash
python -m streamlit run ./src/app.py
```

- If files are missing in dashboard, run the pipeline once first so artifacts are generated.