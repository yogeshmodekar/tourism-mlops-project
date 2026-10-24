"""
Tourism MLOps - Dataset Registration

This script validates the tourism dataset and writes a JSON registration
report to artifacts/reports/data_registration_report.json.

Expected project structure:
tourism_project/
    data/
        tourism.csv
    artifacts/
        reports/
    scripts/
        register.py

Run from a terminal:
    python scripts/register.py

Or run the code cell from Jupyter; it uses the current working directory
to locate the project folder.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import pandas as pd


def resolve_project_directory() -> Path:
    """Resolve the project folder without assuming __file__ exists."""
    # When saved and executed as a .py file, use this file's project parent.
    if "__file__" in globals():
        script_path = Path(__file__).resolve()
        if script_path.parent.name.lower() == "scripts":
            return script_path.parent.parent

    # When run directly in Jupyter, search the current directory and parents.
    current = Path.cwd().resolve()
    candidates = [current] + list(current.parents)

    for base in candidates:
        direct = base / "tourism_project"
        if (direct / "data").is_dir():
            return direct

        # Also support running with the project directory itself as cwd.
        if base.name.lower() == "tourism_project" and (base / "data").is_dir():
            return base

        # Support the repository itself containing data/ directly.
        if base.name.lower() == "github_repo" and (base / "data").is_dir():
            return base

    # Fallback: assume current directory is the project root.
    return current


def find_dataset(project_dir: Path) -> Path:
    """Find a CSV dataset in the project's data directory."""
    data_dir = project_dir / "data"

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data folder not found: {data_dir}\n"
            "Set PROJECT_DIR near the bottom of this script to your actual project path."
        )

    preferred_names = ("tourism.csv", "Tourism.csv")
    for filename in preferred_names:
        candidate = data_dir / filename
        if candidate.is_file():
            return candidate

    csv_files = sorted(data_dir.glob("*.csv"))

    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset found in {data_dir}")
    raise ValueError(
        "Multiple CSV files found in the data folder. Please keep the intended "
        "dataset named tourism.csv or set the desired file explicitly.\n"
        + "\n".join(str(path) for path in csv_files)
    )


def register_dataset(project_dir: Path) -> Path:
    """Validate the dataset and write a registration report."""
    project_dir = project_dir.resolve()
    dataset_path = find_dataset(project_dir)
    report_dir = project_dir / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project directory: {project_dir}")