# ml-ops-CICD-class

## Environment

Use Python 3.10 for local development and automated tests. `mlflow==2.7.1` is significantly safer on Python 3.10 than on Python 3.14, where one of its transitive dependencies (`pyarrow`) may fall back to an unsupported source build. The AWS Lambda deploy script still creates the target function with runtime `python3.12`.

## Local setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pytest
python src/train.py
```

## MLflow

Running `python src/train.py` trains the churn model, stores a confusion matrix in `artifacts/`, and logs the run to MLflow.

```bash
mlflow ui
```

By default, runs are stored locally in `mlruns/`. If you want a remote server, set `MLFLOW_TRACKING_URI` before running the training script.

## AWS deploy

The workflow deploy step is manual and expects these repository secrets:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_LAMBDA_ROLE_ARN`
- `AWS_LAMBDA_FUNCTION_NAME`
