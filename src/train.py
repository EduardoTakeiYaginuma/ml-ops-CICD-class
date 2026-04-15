"""
Train a churn model and log the run with MLflow.
"""

from argparse import ArgumentParser
import math
from pathlib import Path
import os

import mlflow

import matplotlib
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path("data/Churn_Modelling.csv")
ARTIFACTS_DIR = Path("artifacts")
TARGET_COL = "Exited"
FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
CATEGORICAL_COLUMNS = ["Geography", "Gender"]
NUMERIC_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
TEST_SIZE = 0.3
RANDOM_STATE = 1912
REBALANCE_RANDOM_STATE = 1234
MAX_NEIGHBORS = 21


def rebalance(data):
    """
    Downsample the majority class to match the minority class size.
    """
    churn_0 = data[data[TARGET_COL] == 0]
    churn_1 = data[data[TARGET_COL] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0

    churn_maj_downsample = resample(
        churn_maj,
        n_samples=len(churn_min),
        replace=False,
        random_state=REBALANCE_RANDOM_STATE,
    )

    return (
        pd.concat([churn_maj_downsample, churn_min])
        .sample(frac=1, random_state=REBALANCE_RANDOM_STATE)
        .reset_index(drop=True)
    )


def split_data(df):
    """
    Filter the dataset, rebalance the target, and create train/test splits.
    """
    data = df.loc[:, FEATURE_COLUMNS + [TARGET_COL]].copy()
    data_balanced = rebalance(data)

    X = data_balanced[FEATURE_COLUMNS]
    y = data_balanced[TARGET_COL]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_model():
    """
    Create a preprocessing and classification pipeline.
    """
    col_transf = make_column_transformer(
        (StandardScaler(), NUMERIC_COLUMNS),
        (
            OneHotEncoder(handle_unknown="ignore", drop="first"),
            CATEGORICAL_COLUMNS,
        ),
        remainder="passthrough",
    )

    return Pipeline(
        steps=[
            ("preprocessor", col_transf),
            ("classifier", KNeighborsClassifier()),
        ]
    )


def get_neighbor_grid(train_size, cv_folds=2):
    """
    Build a safe odd-valued search grid for n_neighbors.
    """
    min_fold_train_size = train_size - math.ceil(train_size / cv_folds)
    max_neighbors = min(MAX_NEIGHBORS, min_fold_train_size)
    grid = [neighbor for neighbor in range(1, max_neighbors + 1, 2)]
    return grid or [1]


def get_cv_folds(y_train):
    """
    Choose a valid number of CV folds based on the smallest class count.
    """
    min_class_size = int(y_train.value_counts().min())
    return max(2, min(5, min_class_size))


def train(X_train, y_train):
    """
    Train the churn classification pipeline with a grid search over n_neighbors.
    """
    model = build_model()
    cv_folds = get_cv_folds(y_train)
    neighbor_grid = get_neighbor_grid(len(X_train), cv_folds)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid={"classifier__n_neighbors": neighbor_grid},
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=1,
    )
    grid_search.fit(X_train, y_train)

    if mlflow.active_run():
        best_model = grid_search.best_estimator_
        signature = infer_signature(X_train, best_model.predict(X_train))
        input_example = X_train.iloc[:3]
        try:
            mlflow.sklearn.log_model(
                best_model,
                "model",
                signature=signature,
                registered_model_name="churn-model",
                input_example=input_example,
            )
        except MlflowException:
            mlflow.sklearn.log_model(
                best_model,
                "model",
                signature=signature,
                input_example=input_example,
            )

    return grid_search


def evaluate(model, X_test, y_test):
    """
    Generate predictions and common binary classification metrics.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    return metrics, y_pred


def save_confusion_matrix(y_true, y_pred, output_path):
    """
    Persist a confusion matrix figure for later inspection and logging.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conf_mat = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    fig, ax = plt.subplots(figsize=(6, 4))
    display.plot(ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def log_to_mlflow(model, metrics, artifact_paths):
    """
    Log parameters, metrics, and artifacts to MLflow.
    """
    import mlflow
    import mlflow.sklearn

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "bank-churn"))

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": "knn_grid_search",
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "rebalance_random_state": REBALANCE_RANDOM_STATE,
                "best_n_neighbors": model.best_params_["classifier__n_neighbors"],
                "cv_folds": model.cv,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_metric("best_cv_accuracy", model.best_score_)
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))
        mlflow.sklearn.log_model(model.best_estimator_, artifact_path="model")


def run_training(data_path=DATA_PATH, artifacts_dir=ARTIFACTS_DIR, log_mlflow=True):
    """
    Execute the end-to-end training flow.
    """
    data_path = Path(data_path)
    artifacts_dir = Path(artifacts_dir)

    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train(X_train, y_train)
    metrics, y_pred = evaluate(model, X_test, y_test)

    confusion_matrix_path = save_confusion_matrix(
        y_test,
        y_pred,
        artifacts_dir / "confusion_matrix.png",
    )

    if log_mlflow:
        log_to_mlflow(model, metrics, [confusion_matrix_path])

    return model, metrics, confusion_matrix_path


def parse_args():
    """
    Parse command-line arguments for training.
    """
    parser = ArgumentParser()
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser.parse_args()


def main():
    mlflow.set_experiment("churn-exp")
    args = parse_args()
    with mlflow.start_run():
        run_name = "churn-knn-grid-search-model"
        mlflow.set_tag("mlflow.runName", run_name)
        df = pd.read_csv(args.data_path)
        X_train, X_test, y_train, y_test = split_data(df)
        cv_folds = get_cv_folds(y_train)
        neighbor_grid = get_neighbor_grid(len(X_train), cv_folds)
        model = train(X_train, y_train)
        metrics, y_pred = evaluate(model, X_test, y_test)
        confusion_matrix_path = save_confusion_matrix(
            y_test,
            y_pred,
            Path(args.artifacts_dir) / "confusion_matrix.png",
        )

        mlflow.log_param("model", "KNeighborsClassifier")
        mlflow.log_param(
            "neighbor_grid",
            ",".join(str(value) for value in neighbor_grid),
        )
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param(
            "best_n_neighbors",
            model.best_params_["classifier__n_neighbors"],
        )
        mlflow.log_metric("best_cv_accuracy", model.best_score_)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_image(plt.imread(confusion_matrix_path), "confusion_matrix.png")

        print(
            "Best n_neighbors: "
            f"{model.best_params_['classifier__n_neighbors']}"
        )
        print(f"Best CV accuracy: {model.best_score_:.2f}")
        print(f"Accuracy score: {metrics['accuracy']:.2f}")
        print(f"Precision score: {metrics['precision']:.2f}")
        print(f"Recall score: {metrics['recall']:.2f}")
        print(f"F1 score: {metrics['f1']:.2f}")
        print(f"Confusion matrix saved to: {confusion_matrix_path}")


if __name__ == "__main__":
    main()
