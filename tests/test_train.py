import pandas as pd

import train


def make_sample_df():
    rows = []
    for idx in range(16):
        rows.append(
            {
                "CreditScore": 600 + idx,
                "Geography": "France" if idx % 2 == 0 else "Spain",
                "Gender": "Female" if idx % 3 == 0 else "Male",
                "Age": 30 + idx,
                "Tenure": idx % 10,
                "Balance": 1000.0 * (idx + 1),
                "NumOfProducts": 1 if idx % 2 == 0 else 2,
                "HasCrCard": idx % 2,
                "IsActiveMember": (idx + 1) % 2,
                "EstimatedSalary": 50000.0 + (idx * 1000),
                "Exited": 1 if idx in {1, 3, 5, 7, 9, 11} else 0,
            }
        )
    return pd.DataFrame(rows)


def test_rebalance_matches_class_counts():
    df = make_sample_df()
    balanced = train.rebalance(df)
    counts = balanced["Exited"].value_counts()

    assert counts[0] == counts[1]


def test_run_training_flow_without_mlflow(tmp_path):
    df = make_sample_df()
    data_path = tmp_path / "sample.csv"
    artifacts_dir = tmp_path / "artifacts"
    df.to_csv(data_path, index=False)

    model, metrics, confusion_matrix_path = train.run_training(
        data_path=data_path,
        artifacts_dir=artifacts_dir,
        log_mlflow=False,
    )

    assert model is not None
    assert hasattr(model, "best_params_")
    assert "classifier__n_neighbors" in model.best_params_
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert confusion_matrix_path.exists()
