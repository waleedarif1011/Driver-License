"""Feature engineering for driver-license qualification classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "driver_license_sample.csv"

TARGET_COL = "Qualified"
ID_COL = "Applicant ID"

CATEGORICAL_COLS: tuple[str, ...] = (
    "Gender",
    "Age Group",
    "Race",
    "Training",
    "Reactions",
)

NUMERIC_COLS: tuple[str, ...] = (
    "Signals",
    "Yield",
    "Speed Control",
    "Night Drive",
    "Road Signs",
    "Steer Control",
    "Mirror Usage",
    "Confidence",
    "Parking",
    "Theory Test",
)

RAW_COLUMNS: tuple[str, ...] = (
    ID_COL,
    *CATEGORICAL_COLS[:3],
    "Training",
    *NUMERIC_COLS,
    "Reactions",
    TARGET_COL,
)

# Gradio / CLI interactive inputs (all raw predictors except ID).
DEMO_FEATURES: tuple[str, ...] = (
    "Gender",
    "Age Group",
    "Race",
    "Training",
    "Reactions",
    *NUMERIC_COLS,
)

GENDER_CHOICES = ("Female", "Male")
AGE_GROUP_CHOICES = ("Teenager", "Young Adult", "Middle Age")
RACE_CHOICES = ("Black", "White", "Other")
TRAINING_CHOICES = ("Basic", "Advanced")
REACTIONS_CHOICES = ("Slow", "Average", "Fast")


def load_raw(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    """Load the drivers-license CSV with the notebook schema."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dataset at {path}. Run: python scripts/fetch_data.py"
        )
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df.copy()


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Impute Training mode (notebook step) and coerce numerics."""
    out = df.copy()
    if out["Training"].isna().any():
        mode_training = out["Training"].mode(dropna=True)
        fill = str(mode_training.iloc[0]) if len(mode_training) else "Basic"
        out["Training"] = out["Training"].fillna(fill)

    num_cols = list(NUMERIC_COLS)
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[num_cols] = out[num_cols].fillna(out[num_cols].median())

    for col in list(CATEGORICAL_COLS) + [TARGET_COL]:
        out[col] = out[col].astype(str).str.strip()
    return out


def encode_features(
    df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """One-hot encode categoricals (``drop_first=True``) matching the notebook.

    Returns ``(X, y, feature_names)`` where ``y`` is binary ``Qualified_Yes``.
    """
    work = clean_raw(df)
    work = work.drop(columns=[ID_COL], errors="ignore")

    y = (work[TARGET_COL].str.lower() == "yes").astype(int)
    work = work.drop(columns=[TARGET_COL])

    encoded = pd.get_dummies(
        work,
        columns=list(CATEGORICAL_COLS),
        drop_first=True,
    )
    # Keep boolean / numeric as float for sklearn / LightGBM.
    encoded = encoded.astype(float)

    if feature_columns is None:
        feature_names = list(encoded.columns)
        X = encoded
    else:
        feature_names = list(feature_columns)
        X = encoded.reindex(columns=feature_names, fill_value=0.0)

    return X, y, feature_names


def prepare_frame(
    path: str | Path = DEFAULT_DATA,
    *,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load → clean → encode for modeling."""
    return encode_features(load_raw(path), feature_columns=feature_columns)


def raw_row_to_frame(features: dict[str, Any]) -> pd.DataFrame:
    """Build a one-row raw DataFrame from a feature dict (demo / CLI)."""
    row: dict[str, Any] = {ID_COL: "AID_PRED"}
    for col in CATEGORICAL_COLS:
        if col not in features:
            raise ValueError(f"Missing categorical feature: {col}")
        row[col] = features[col]
    for col in NUMERIC_COLS:
        if col not in features:
            raise ValueError(f"Missing numeric feature: {col}")
        row[col] = float(features[col])
    # Placeholder target so clean/encode paths stay unified.
    row[TARGET_COL] = features.get(TARGET_COL, "No")
    return pd.DataFrame([row])


def feature_defaults(df: pd.DataFrame) -> dict[str, Any]:
    """Modal categoricals + median numerics for demo defaults."""
    clean = clean_raw(df)
    defaults: dict[str, Any] = {}
    for col in CATEGORICAL_COLS:
        defaults[col] = str(clean[col].mode(dropna=True).iloc[0])
    for col in NUMERIC_COLS:
        defaults[col] = float(clean[col].median())
    return defaults
