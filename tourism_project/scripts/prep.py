"""
Tourism MLOps - Data Preparation

Expected project structure:
tourism_project/
    data/
        tourism.csv
    artifacts/
        reports/
        train.csv
        test.csv
    scripts/
        register.py
        prep.py

Run from the project root:
    python scripts/prep.py

This script prepares data for model training. It does not train a model.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


def resolve_project_directory() -> Path:
    """Find the project root when run as a script or from Jupyter."""
    if "__file__" in globals():
        script_path = Path(__file__).resolve()
        if script_path.parent.name.lower() == "scripts":
            return script_path.parent.parent

    current = Path.cwd().resolve()
    for base in [current, *current.parents]:
        direct = base / "tourism_project"
        if (direct / "data").is_dir():
            return direct

        if base.name.lower() == "tourism_project" and (base / "data").is_dir():
            return base

        if base.name.lower() == "github_repo" and (base / "data").is_dir():
            return base

    return current


def find_dataset(project_dir: Path) -> Path:
    """Locate the dataset inside data/."""
    data_dir = project_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for filename in ("tourism.csv", "Tourism.csv"):
        path = data_dir / filename
        if path.is_file():
            return path

    csv_files = sorted(data_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset found in {data_dir}")

    raise ValueError(
        "Multiple CSV files found. Keep the intended dataset named tourism.csv "
        "or update find_dataset().\n" + "\n".join(map(str, csv_files))
    )


def prepare_data(project_dir: Path) -> dict:
    """Clean, split, save datasets and write a preparation report."""
    project_dir = project_dir.resolve()
    dataset_path = find_dataset(project_dir)
    artifacts_dir = project_dir / "artifacts"
    reports_dir = artifacts_dir / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project directory: {project_dir}")
    print(f"Loading dataset:   {dataset_path}")

    original_df = pd.read_csv(dataset_path)
    original_df.columns = original_df.columns.astype(str).str.strip()

    if original_df.empty:
        raise ValueError("Input dataset is empty.")

    if "ProdTaken" not in original_df.columns:
        raise ValueError(
            "Target column 'ProdTaken' is missing. "