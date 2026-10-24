
"""
Tourism MLOps - Model Training and Evaluation

Steps:
1. Load train.csv and test.csv.
2. Build a preprocessing + Random Forest pipeline.
3. Tune hyperparameters with GridSearchCV.
4. Log parameters, metrics and model to MLflow.
5. Evaluate the best model on the held-out test set.
6. Save the model to deployment/ and reports to artifacts/reports/.

Expected structure:
tourism_project/
    artifacts/
        train.csv
        test.csv
    deployment/
    scripts/
        prep.py
        train.py
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "ProdTaken"
RANDOM_STATE = 42


def get_project_dir():
    """Find the tourism_project root from this script's location."""
    script_path = Path(__file__).resolve()

    if script_path.parent.name.lower() in {
        "scripts", "model_building"
    }:
        return script_path.parent.parent

    raise RuntimeError(
        "Could not determine project root. Save this file under "
        "tourism_project/scripts/train.py."
    )


def load_dataset(path, name):
    """Load a prepared dataset and separate predictors from target."""
    if not path.exists():
        raise FileNotFoundError(
            f"{name} dataset not found: {path}. "
            "Run prep.py first."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    if df.empty:
        raise ValueError(f"{name} dataset is empty.")

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' is missing from {path}."
        )

    if df[TARGET].isna().any():
        raise ValueError(f"{name} target contains missing values.")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError(
            f"{name} target must contain both 0 and 1."
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    return X, y


def build_pipeline(X):
    """Create preprocessing steps and the Random Forest model."""
    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        col for col in X.columns
        if col not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("No predictor columns were found.")

    preprocessing = ColumnTransformer(transformers)

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        (
            "model",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=1,
            ),
        ),
    ])

    # 12 combinations; GridSearchCV selects the best by cross-validation F1.
    parameter_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 15],
        "model__min_samples_leaf": [1, 2],
        "model__min_samples_split": [2, 5],
    }

    return pipeline, parameter_grid


def main():
    project_dir = get_project_dir()

    artifacts_dir = project_dir / "artifacts"
    reports_dir = artifacts_dir / "reports"
    deployment_dir = project_dir / "deployment"

    reports_dir.mkdir(parents=True, exist_ok=True)
    deployment_dir.mkdir(parents=True, exist_ok=True)

    # Load the outputs produced by prep.py.
    X_train, y_train = load_dataset(
        artifacts_dir / "train.csv", "Training"
    )
    X_test, y_test = load_dataset(
        artifacts_dir / "test.csv", "Testing"
    )

    if set(X_train.columns) != set(X_test.columns):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    X_test = X_test[X_train.columns]

    # Determine the largest feasible stratified CV fold count.
    min_class_count = int(y_train.value_counts().min())
    cv_folds = min(5, min_class_count)

    if cv_folds < 2:
        raise ValueError(
            "Not enough examples in the minority class for cross-validation."
        )

    pipeline, parameter_grid = build_pipeline(X_train)

    cross_validation = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    # MLflow tracking. Local runs are stored in the mlruns directory.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Tourism_Product_Taken")

    print("Project directory:", project_dir)
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Predictor columns:", len(X_train.columns))
    print("Cross-validation folds:", cv_folds)

    with mlflow.start_run(run_name="random_forest_grid_search") as run:
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "target": TARGET,
            "random_state": RANDOM_STATE,
            "cv_folds": cv_folds,
            "scoring": "f1",
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
        })

        print("\nStarting hyperparameter tuning...")
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
            "best_cross_validation_f1": float(search.best_score_),
        }

        # Log selected parameters and evaluation metrics to MLflow.
        mlflow.log_params({
            key: str(value)
            for key, value in search.best_params_.items()
        })
        mlflow.log_metrics(metrics)

        # Save the full fitted pipeline, including preprocessing.
        model_path = deployment_dir / "tourism_model.joblib"
        joblib.dump(best_model, model_path)

        # Save evaluation metrics.
        metrics_path = reports_dir / "evaluation_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        # Save selected hyperparameters.
        params_path = reports_dir / "best_parameters.json"
        best_params = {
            key: value if value is not None else "None"
            for key, value in search.best_params_.items()
        }
        with params_path.open("w", encoding="utf-8") as file:
            json.dump(best_params, file, indent=4)

        # Save detailed evaluation report.
        report = {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "target": TARGET,
            "model": "RandomForestClassifier",
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "feature_columns": X_train.columns.tolist(),
            "best_parameters": best_params,
            "metrics": metrics,
            "classification_report": classification_report(
                y_test, predictions,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            ).tolist(),
            "mlflow_run_id": run.info.run_id,
        }

        report_path = reports_dir / "model_training_report.json"
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        # Log the fitted pipeline and report files to MLflow.
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(params_path))
        mlflow.log_artifact(str(report_path))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="tourism_model",
        )

        print("\nMODEL TRAINING COMPLETED")
        print("\nBest parameters:")
        print(json.dumps(best_params, indent=4))
        print("\nEvaluation metrics:")
        print(json.dumps(metrics, indent=4))
        print("\nSaved model:", model_path)
        print("Saved reports:", reports_dir)
        print("MLflow run ID:", run.info.run_id)
        print("MLflow tracking URI:", mlflow.get_tracking_uri())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"MODEL TRAINING FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
