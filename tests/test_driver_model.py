"""Tests for feature pipeline, model load, and qualification probability."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.features import (
    DEMO_FEATURES,
    NUMERIC_COLS,
    PRE_TEST_CATEGORICAL_COLS,
    TARGET_COL,
    encode_features,
    load_raw,
    prepare_frame,
)
from src.predict import load_model, predict
from src.train import METRICS_PATH, MODEL_PATH, train

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "driver_license_sample.csv"


@pytest.fixture(scope="module")
def trained_metrics():
    assert DATA.is_file()
    metrics = train()
    assert MODEL_PATH.is_file()
    assert METRICS_PATH.is_file()
    assert metrics["best"]["roc_auc"] >= 0.70
    return metrics


@pytest.fixture(scope="module")
def artifact(trained_metrics):
    return load_model(MODEL_PATH)


def test_sample_schema():
    df = load_raw(DATA)
    assert len(df) >= 200
    for col in DEMO_FEATURES:
        assert col in df.columns
    assert TARGET_COL in df.columns
    assert "Applicant ID" in df.columns


def test_feature_pipeline_imputes_training():
    X, y, features = prepare_frame()
    assert TARGET_COL not in X.columns
    assert "Applicant ID" not in X.columns
    assert X.isna().sum().sum() == 0
    assert set(y.unique()).issubset({0, 1})
    assert len(features) == X.shape[1]
    # drop_first one-hot leaves these encoded columns.
    assert "Gender_Male" in features
    assert "Training_Basic" in features


def test_encode_aligns_to_training_columns(artifact):
    raw = load_raw(DATA).head(3)
    X, y, _ = encode_features(raw, feature_columns=list(artifact["features"]))
    assert list(X.columns) == list(artifact["features"])
    assert len(y) == 3


def test_metrics_include_three_models(trained_metrics):
    names = {m["name"] for m in trained_metrics["models"]}
    assert names == {"LogisticRegression", "RandomForest", "LightGBM"}
    for row in trained_metrics["models"]:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["roc_auc"] <= 1.0
    assert trained_metrics["exported_model"] == "LightGBM"


def test_pre_test_ablation_recorded(trained_metrics):
    abl = trained_metrics["pre_test_ablation"]
    assert abl["n_features"] >= 1
    assert 0.0 <= abl["roc_auc"] <= 1.0
    assert 0.0 <= abl["f1"] <= 1.0
    # Pre-test-only model should not beat the circular full model on ROC-AUC.
    full_auc = trained_metrics["best"]["roc_auc"]
    assert abl["roc_auc"] <= full_auc + 1e-9
    assert trained_metrics["pre_test_features"] == list(PRE_TEST_CATEGORICAL_COLS)
    assert trained_metrics["credibility_notes"]["synthetic_data"] is True


def test_prepare_frame_pre_test_excludes_scores():
    X, y, features = prepare_frame(include_scores=False)
    assert len(y) == len(X)
    for col in NUMERIC_COLS:
        assert col not in features
    assert X.shape[1] == len(features)
    assert X.isna().sum().sum() == 0


def test_model_file_loads(artifact):
    assert "model" in artifact
    assert hasattr(artifact["model"], "predict_proba")
    assert artifact.get("model_name") == "LightGBM"
    assert len(artifact.get("presets") or []) == 3


def test_probability_in_unit_interval(artifact):
    defaults = artifact["feature_defaults"]
    features = {c: defaults[c] for c in DEMO_FEATURES}
    result = predict(features, artifact=artifact)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["badge"] in {"Qualified", "Not Qualified"}
    assert len(result["top_factors"]) >= 1


def test_strong_scores_raise_probability(artifact):
    defaults = artifact["feature_defaults"]
    weak = {c: defaults[c] for c in DEMO_FEATURES}
    strong = dict(weak)
    weak["Training"] = "Basic"
    weak["Reactions"] = "Slow"
    for col in NUMERIC_COLS:
        weak[col] = 25.0
    weak["Theory Test"] = 45.0

    strong["Training"] = "Advanced"
    strong["Reactions"] = "Fast"
    for col in NUMERIC_COLS:
        strong[col] = 85.0
    strong["Theory Test"] = 95.0

    low = predict(weak, artifact=artifact)
    high = predict(strong, artifact=artifact)
    assert high["probability"] > low["probability"]
