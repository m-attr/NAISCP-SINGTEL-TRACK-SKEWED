# DataDrift V2

DataDrift is a one-model, bounded-memory competition pipeline that detects unusual feature movement, chooses a conservative Keep / Repair / Drop decision, freezes one reusable preparation plan, and then fits the required LightGBM exactly once.

The internal decision system distinguishes ordinary keeps, same-semantic population drift, and dataset-wide systemic drift. Repair is allowed only for strongly evidenced percentile reversal, pure scale change, pure offset change, or unambiguous category relabeling. Large movement by itself is never a reason to drop a feature; dropping is a last resort for severe novel missingness or unusable categorical support. Every repair parameter and category mapping is learned once before fitting and reused unchanged for every prediction chunk.

## Production pipeline

Install with Python 3.12:

```bash
pip install -r requirements.txt
```

Run the competition entry point:

```bash
python src/main.py \
  --train_data_filepath train.csv \
  --test_data_filepath test.csv
```

A successful run writes:

- `prediction.csv` with exactly `CustomerID` and `probability_score`;
- `latest_metrics.json` and `latest_metrics_summary.json`;
- `drift_report.json` with feature evidence and final actions;
- `preparation_plan.json` with the frozen run-plan summary.

Set `PREDICTION_OUTPUT_PATH` to place `prediction.csv` elsewhere. The remaining small run artifacts are written to the current working directory.

The production model is `lightgbm==4.6.0` with the fixed competition constructor. A successful run performs one—and only one—final fit. Test `ChurnStatus` is excluded by column selection before test values are read.

## External public evaluation

Development evaluation is separate and runs only after predictions exist:

```bash
python tools/evaluate_public_predictions.py \
  --prediction_filepath prediction.csv \
  --labelled_test_filepath test.csv \
  --output_filepath external_evaluation.json
```

The production pipeline neither imports this evaluator nor receives its result.

## Real dashboard artifacts

Build the compact versioned dashboard contract from a completed run:

```bash
python tools/build_dashboard_artifacts.py \
  --train_data_filepath train.csv \
  --test_data_filepath test.csv \
  --run_artifact_dir . \
  --external_evaluation_filepath external_evaluation.json \
  --output_filepath dashboard_artifacts/dashboard_run.json
```

This utility performs bounded scans, fits no model, never reads test `ChurnStatus`, stores no raw rows, and feeds nothing back into production. Omit `--external_evaluation_filepath` when no legitimate external score exists; the dashboard will display “Not available.”

## Dashboard

Launch the artifact-driven multipage Streamlit application:

```bash
streamlit run dashboard/app.py
```

Or use the root compatibility entry point:

```bash
streamlit run app.py
```

By default the dashboard reads `dashboard_artifacts/dashboard_run.json`. Set `DATADRIFT_DASHBOARD_ARTIFACT` to select another run. Missing artifacts produce an explicit unavailable state—never generated fallback data.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite covers the exact model constructor and fit boundary, test-target isolation, anonymous feature names, `YY-MMM` chronology, reusable transformations, prediction chunk invariance, multi-seed positive and negative drift scenarios, systemic-drift safeguards, missingness relative to history, action policy, CLI/output compliance, and dashboard trust/schema validation.

## Source layout

```text
src/
  artifacts/       production JSON writing
  common/          contract columns, month parsing, shared types
  detection/       drift and relationship evidence only
  mitigation/      conservative keep / repair / last-resort drop policy
  model/           one fixed LightGBM boundary
  pipeline/        production orchestration
  preprocessing/   frozen plan construction and reuse
  runtime/         bounded scans, streaming, telemetry
  main.py          competition CLI
dashboard/         read-only multipage Streamlit application
tools/             external evaluator, dashboard artifact builder, and smoke-data generator
tests/             compliance and regression suite
```
