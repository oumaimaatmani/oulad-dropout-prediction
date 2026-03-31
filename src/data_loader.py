"""
Data loading, validation, and initial cleaning utilities for OULAD.

This module handles:
- Loading all seven OULAD CSV files with correct dtypes
- Schema validation (expected columns and types)
- Basic cleaning (missing value handling, type casting)
- Dataset summary statistics
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# Expected files and their required columns
OULAD_SCHEMA = {
    "studentInfo": [
        "code_module", "code_presentation", "id_student", "gender",
        "region", "highest_education", "imd_band", "age_band",
        "num_of_prev_attempts", "studied_credits", "disability",
        "final_result",
    ],
    "studentRegistration": [
        "code_module", "code_presentation", "id_student",
        "date_registration", "date_unregistration",
    ],
    "studentAssessment": [
        "id_assessment", "id_student", "date_submitted",
        "is_banked", "score",
    ],
    "studentVle": [
        "code_module", "code_presentation", "id_student",
        "id_site", "date", "sum_click",
    ],
    "assessments": [
        "code_module", "code_presentation", "id_assessment",
        "assessment_type", "date", "weight",
    ],
    "vle": [
        "code_module", "code_presentation", "id_site",
        "activity_type", "week_from", "week_to",
    ],
    "courses": [
        "code_module", "code_presentation", "module_presentation_length",
    ],
}


def load_oulad(data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Load all OULAD CSV files from the specified directory.

    Parameters
    ----------
    data_dir : str
        Path to directory containing the raw OULAD CSV files.

    Returns
    -------
    dict
        Dictionary mapping table names to DataFrames.

    Raises
    ------
    FileNotFoundError
        If any required CSV file is missing.
    ValueError
        If any file is missing required columns.
    """
    data_path = Path(data_dir)
    tables = {}

    for table_name, required_cols in OULAD_SCHEMA.items():
        filepath = data_path / f"{table_name}.csv"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Required file not found: {filepath}\n"
                f"Download OULAD from https://analyse.kmi.open.ac.uk/open_dataset "
                f"and place all CSV files in {data_dir}/"
            )

        df = pd.read_csv(filepath)

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Table '{table_name}' is missing columns: {missing_cols}"
            )

        tables[table_name] = df
        print(f"  Loaded {table_name}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    print(f"\nAll {len(tables)} OULAD tables loaded successfully.")
    return tables


def clean_student_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the studentInfo table.

    Operations:
    - Map final_result to binary completion status
    - Handle missing imd_band values
    - Cast categorical columns to category dtype
    - Create numeric encodings for ordered categories
    """
    df = df.copy()

    # Binary target: 1 = dropout (Withdrawn or Fail), 0 = completed (Pass or Distinction)
    df["is_dropout"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)

    # Completion status (inverse perspective)
    df["completion_status"] = df["final_result"].isin(
        ["Pass", "Distinction"]
    ).astype(int)

    # Handle missing IMD band
    df["imd_band"] = df["imd_band"].fillna("Unknown")

    # Ordered age band encoding
    age_order = {"0-35": 0, "35-55": 1, "55<=": 2}
    df["age_band_numeric"] = df["age_band"].map(age_order).fillna(1).astype(int)

    # Education level encoding (ordinal)
    edu_order = {
        "No Formal quals": 0,
        "Lower Than A Level": 1,
        "A Level or Equivalent": 2,
        "HE Qualification": 3,
        "Post Graduate Qualification": 4,
    }
    df["education_numeric"] = (
        df["highest_education"].map(edu_order).fillna(1).astype(int)
    )

    # Categorical dtypes for memory efficiency
    cat_cols = [
        "code_module", "code_presentation", "gender", "region",
        "highest_education", "imd_band", "age_band", "disability",
        "final_result",
    ]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df


def clean_student_vle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the studentVle interaction table.

    Operations:
    - Rename sum_click for clarity
    - Remove rows with zero or negative clicks
    - Cast date to integer
    """
    df = df.copy()
    df = df.rename(columns={"sum_click": "clicks"})
    df = df[df["clicks"] > 0].copy()
    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].astype(int)
    return df


def clean_assessments(
    assessments: pd.DataFrame, student_assessment: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean assessment-related tables.

    Operations:
    - Handle missing assessment dates (set to module end)
    - Cap scores at 100
    - Merge assessment metadata into student scores
    """
    assessments = assessments.copy()
    student_assessment = student_assessment.copy()

    # Fill missing assessment dates with a large value (end of module)
    assessments["date"] = assessments["date"].fillna(999)
    assessments["date"] = assessments["date"].astype(int)

    # Cap student scores at 100
    student_assessment["score"] = student_assessment["score"].clip(upper=100)

    # Fill missing scores with 0 (not submitted)
    student_assessment["score"] = student_assessment["score"].fillna(0)

    return assessments, student_assessment


def get_student_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique student identifiers (module + presentation + student id).
    """
    key_cols = ["code_module", "code_presentation", "id_student"]
    return df[key_cols].drop_duplicates().reset_index(drop=True)


def summarize_dataset(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate a summary of all loaded tables.

    Returns a DataFrame with row counts, column counts, memory usage,
    and missing value percentages for each table.
    """
    records = []
    for name, df in tables.items():
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        records.append({
            "table": name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "missing_pct": round(missing_pct, 2),
        })
    return pd.DataFrame(records)
