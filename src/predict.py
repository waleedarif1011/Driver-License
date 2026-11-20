"""Single-candidate driver qualification inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.features import (
    AGE_GROUP_CHOICES,
    DEMO_FEATURES,
    GENDER_CHOICES,
    NUMERIC_COLS,
    RACE_CHOICES,
    REACTIONS_CHOICES,
    TARGET_COL,
    TRAINING_CHOICES,
    encode_features,
    raw_row_to_frame,
)
from src.train import MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]


def load_model(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing model at {path}. Run: python -m src.train")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Unexpected artifact format in {path}")
    return artifact


def predict(
    features: dict[str, Any],
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return qualification probability, badge, and top contributing factors."""
    art = artifact or load_model()
    frame = raw_row_to_frame(features)
    X, _, _ = encode_features(frame, feature_columns=list(art["features"]))
    model = art["model"]
    probability = float(np.asarray(model.predict_proba(X)[:, 1], dtype=float).ravel()[0])
    qualified = probability >= 0.5
    badge = "Qualified" if qualified else "Not Qualified"

    importances = dict(art.get("feature_importance") or {})
    # Map encoded feature importances back to human-readable drivers.
    drivers: list[dict[str, Any]] = []
    for name, imp in sorted(importances.items(), key=lambda t: t[1], reverse=True):
        drivers.append({"feature": name, "importance": round(float(imp), 4)})
        if len(drivers) >= 5:
            break

    return {
        "badge": badge,
        "qualified": qualified,
        "probability": round(probability, 4),
        "probability_pct": round(probability * 100.0, 1),
        "model_name": art.get("model_name"),
        "target": art.get("target", TARGET_COL),
        "features_used": {k: features[k] for k in DEMO_FEATURES if k in features},
        "top_factors": drivers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict driver qualification probability for one candidate."
    )
    parser.add_argument("--gender", choices=GENDER_CHOICES, required=True)
    parser.add_argument("--age-group", choices=AGE_GROUP_CHOICES, required=True)
    parser.add_argument("--race", choices=RACE_CHOICES, required=True)
    parser.add_argument("--training", choices=TRAINING_CHOICES, required=True)
    parser.add_argument("--reactions", choices=REACTIONS_CHOICES, required=True)
    for col in NUMERIC_COLS:
        flag = "--" + col.lower().replace(" ", "-")
        parser.add_argument(flag, type=float, required=True, dest=col)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    features: dict[str, Any] = {
        "Gender": args.gender,
        "Age Group": args.age_group,
        "Race": args.race,
        "Training": args.training,
        "Reactions": args.reactions,
    }
    for col in NUMERIC_COLS:
        features[col] = float(getattr(args, col))

    result = predict(features, artifact=load_model(args.model))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
