
from pathlib import Path
import json
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # Project root is the parent of the scripts directory.
    project_dir = Path(__file__).resolve().parent.parent

    data_path = project_dir / "data" / "tourism.csv"
    artifacts_dir = project_dir / "artifacts"
    reports_dir = artifacts_dir / "reports"

    reports_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    original_rows = len(df)

    # Remove the index column, if present.
    df = df.drop(
        columns=[
            col for col in ["Unnamed: 0", "index"]
            if col in df.columns
        ],
        errors="ignore"
    )

    # Remove duplicate records.
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    if "ProdTaken" not in df.columns:
        raise ValueError("Required target column 'ProdTaken' is missing.")

    # Remove records with missing values.
    missing_rows = int(df.isna().any(axis=1).sum())
    df = df.dropna().copy()

    # Validate target values.
    df["ProdTaken"] = pd.to_numeric(df["ProdTaken"], errors="raise")

    if set(df["ProdTaken"].unique()) != {0, 1}:
        raise ValueError(
            "Target column 'ProdTaken' must contain both 0 and 1 only."
        )

    if df["ProdTaken"].value_counts().min() < 2:
        raise ValueError(
            "Not enough examples in the minority target class for a "
            "stratified train/test split."
        )

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["ProdTaken"]
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(artifacts_dir / "train.csv", index=False)
    test_df.to_csv(artifacts_dir / "test.csv", index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(data_path),
        "original_rows": original_rows,
        "rows_after_cleaning": len(df),
        "duplicates_removed": duplicate_count,
        "rows_with_missing_values_removed": missing_rows,
        "training_rows": len(train_df),
        "testing_rows": len(test_df),
        "feature_count_including_target": len(df.columns),
        "target_distribution": {
            str(k): int(v)
            for k, v in df["ProdTaken"].value_counts().items()
        }
    }

    report_path = reports_dir / "preparation_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Data preparation completed successfully.")
    print(f"Training data: {artifacts_dir / 'train.csv'}")
    print(f"Testing data:  {artifacts_dir / 'test.csv'}")
    print(f"Preparation report: {report_path}")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")


if __name__ == "__main__":
    main()
