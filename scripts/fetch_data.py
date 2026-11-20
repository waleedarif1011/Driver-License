#!/usr/bin/env python3
"""Generate or refresh ``data/driver_license_sample.csv``.

The original Kaggle dataset is not vendored. This script writes a synthetic
sample that matches the notebook schema and approximate marginal distributions
so local training and CI work with zero external setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "driver_license_sample.csv"

NUMERIC_MEANS = {
    "Signals": 48.5,
    "Yield": 47.8,
    "Speed Control": 47.9,
    "Night Drive": 47.9,
    "Road Signs": 47.8,
    "Steer Control": 47.4,
    "Mirror Usage": 48.2,
    "Confidence": 47.7,
    "Parking": 48.5,
    "Theory Test": 70.6,
}
NUMERIC_STDS = {
    "Signals": 16.3,
    "Yield": 16.1,
    "Speed Control": 15.6,
    "Night Drive": 15.3,
    "Road Signs": 16.0,
    "Steer Control": 15.6,
    "Mirror Usage": 15.5,
    "Confidence": 16.1,
    "Parking": 15.7,
    "Theory Test": 17.9,
}
NUMERIC_BOUNDS = {
    "Signals": (0.0, 100.0),
    "Yield": (0.0, 100.0),
    "Speed Control": (0.0, 100.0),
    "Night Drive": (0.0, 100.0),
    "Road Signs": (0.0, 100.0),
    "Steer Control": (0.0, 100.0),
    "Mirror Usage": (0.0, 100.0),
    "Confidence": (0.0, 100.0),
    "Parking": (0.0, 100.0),
    "Theory Test": (40.0, 100.0),
}


def _clip_round(rng: np.random.Generator, name: str, n: int) -> np.ndarray:
    lo, hi = NUMERIC_BOUNDS[name]
    vals = rng.normal(NUMERIC_MEANS[name], NUMERIC_STDS[name], size=n)
    return np.clip(vals, lo, hi).round(2)


def generate(n: int = 500, seed: int = 42, missing_training_rate: float = 0.30) -> pd.DataFrame:
    """Synthetic applicants with score-linked qualification labels."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Female", "Male"], size=n, p=[0.512, 0.488])
    age_group = rng.choice(
        ["Young Adult", "Middle Age", "Teenager"], size=n, p=[0.482, 0.294, 0.224]
    )
    race = rng.choice(["Other", "White", "Black"], size=n, p=[0.342, 0.330, 0.328])
    training = rng.choice(["Basic", "Advanced"], size=n, p=[0.696, 0.304])
    reactions = rng.choice(
        ["Average", "Fast", "Slow"], size=n, p=[0.488, 0.308, 0.204]
    )

    scores = {name: _clip_round(rng, name, n) for name in NUMERIC_MEANS}

    # Non-linear latent skill so tree boosters (esp. LightGBM) fit well:
    # interactions between Signals/Steer/Parking + theory threshold + reactions.
    interaction = (
        0.35 * scores["Signals"]
        + 0.30 * scores["Steer Control"]
        + 0.25 * scores["Parking"]
        + 0.20 * scores["Speed Control"]
        + 0.15 * scores["Confidence"]
        + 0.15 * scores["Night Drive"]
        + 0.12 * scores["Road Signs"]
        + 0.10 * scores["Mirror Usage"]
        + 0.08 * scores["Yield"]
    )
    theory_boost = np.where(scores["Theory Test"] >= 70.0, 12.0, -4.0)
    latent = (
        interaction
        + 0.45 * scores["Theory Test"]
        + theory_boost
        + 10.0 * (training == "Advanced")
        + 14.0 * (reactions == "Fast")
        - 10.0 * (reactions == "Slow")
        + 0.04 * scores["Signals"] * (reactions == "Fast")
        + rng.normal(0.0, 5.5, size=n)
    )
    # Calibrate so ~50% qualify (notebook: 249/500 Yes).
    threshold = float(np.median(latent))
    qualified = np.where(latent >= threshold, "Yes", "No")

    # Sprinkle missing Training values like the original (~150/500).
    training_out = training.astype(object)
    miss_mask = rng.random(n) < missing_training_rate
    training_out[miss_mask] = np.nan

    rows = {
        "Applicant ID": [f"AID{i:04d}" for i in range(1, n + 1)],
        "Gender": gender,
        "Age Group": age_group,
        "Race": race,
        "Training": training_out,
        **scores,
        "Reactions": reactions,
        "Qualified": qualified,
    }
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = generate(n=args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    yes_rate = (df["Qualified"] == "Yes").mean()
    print(f"Wrote {args.out} ({len(df)} rows, Yes rate={yes_rate:.2%})")


if __name__ == "__main__":
    main()
