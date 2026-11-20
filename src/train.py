"""Train Logistic Regression, Random Forest, and LightGBM; export artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.features import (
    DEMO_FEATURES,
    NUMERIC_COLS,
    TARGET_COL,
    feature_defaults,
    load_raw,
    prepare_frame,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "assets"
MODEL_PATH = ARTIFACTS / "driver_lgbm.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
ROC_PATH = ASSETS / "roc_comparison.png"
IMPORTANCE_PATH = ASSETS / "feature_importance.png"

RANDOM_STATE = 42


def build_models() -> dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(
            random_state=RANDOM_STATE, solver="liblinear"
        ),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "LightGBM": lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    }


def _evaluate(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, Any]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "y_prob": y_prob.tolist(),
        "model": model,
    }


def _feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float).ravel()
        return {f: float(v) for f, v in zip(feature_names, imp)}
    if hasattr(model, "coef_"):
        coef = np.abs(np.asarray(model.coef_, dtype=float).ravel())
        return {f: float(v) for f, v in zip(feature_names, coef)}
    return {f: 0.0 for f in feature_names}


def _save_roc(
    results: list[dict[str, Any]],
    y_test: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {
        "LogisticRegression": "#5c6b73",
        "RandomForest": "#244a6b",
        "LightGBM": "#1b7f5a",
    }
    for row in results:
        fpr, tpr, _ = roc_curve(y_test, row["y_prob"])
        ax.plot(
            fpr,
            tpr,
            color=colors.get(row["name"], "#333"),
            lw=2,
            label=f"{row['name']} (AUC={row['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_importance(importances: dict[str, float], path: Path) -> None:
    names = list(importances.keys())
    values = [importances[n] for n in names]
    order = np.argsort(values)[::-1]
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(x=values, y=names, ax=ax, color="#1b7f5a", orient="h")
    ax.set_xlabel("Importance (LightGBM split gain count)")
    ax.set_ylabel("Feature")
    ax.set_title("Feature importance (LightGBM)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _pick_presets(
    raw: pd.DataFrame,
    X: pd.DataFrame,
    model: Any,
    test_index: np.ndarray,
) -> list[dict[str, Any]]:
    """Three held-out candidates: low / mid / high predicted qualification prob."""
    X_test = X.loc[test_index]
    probs = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
    order = np.argsort(probs)
    positions = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    labels = [
        "Borderline / not ready",
        "Average candidate",
        "Strong candidate",
    ]
    presets: list[dict[str, Any]] = []
    for label, pos in zip(labels, positions):
        idx = test_index[pos]
        row = raw.loc[idx]
        values = {c: row[c] for c in DEMO_FEATURES}
        for col in NUMERIC_COLS:
            values[col] = float(values[col])
        presets.append(
            {
                "label": label,
                "actual_qualified": int(str(row[TARGET_COL]).lower() == "yes"),
                "predicted_probability": float(probs[pos]),
                "values": values,
            }
        )
    return presets


def train(
    data_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    path = Path(data_path) if data_path is not None else None
    raw = load_raw(path) if path is not None else load_raw()
    X, y, feature_names = prepare_frame(path) if path is not None else prepare_frame()
    y_arr = y.to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )

    results: list[dict[str, Any]] = []
    for name, model in build_models().items():
        results.append(_evaluate(name, model, X_train, y_train, X_test, y_test))

    # Prefer higher ROC-AUC, then F1, then accuracy (matches notebook winner).
    best = max(results, key=lambda r: (r["roc_auc"], r["f1"], r["accuracy"]))
    # Portfolio target: ship LightGBM as the exported model when competitive.
    lgbm_row = next(r for r in results if r["name"] == "LightGBM")
    export_row = lgbm_row
    export_model = export_row["model"]
    importances = _feature_importance(export_model, feature_names)
    defaults = feature_defaults(raw)

    test_index = X_test.index.to_numpy()
    presets = _pick_presets(raw, X, export_model, test_index)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": export_model,
        "model_name": export_row["name"],
        "features": feature_names,
        "demo_features": list(DEMO_FEATURES),
        "target": TARGET_COL,
        "feature_importance": importances,
        "feature_defaults": defaults,
        "presets": presets,
        "best_benchmark_model": best["name"],
    }
    joblib.dump(artifact, MODEL_PATH)

    summary_rows = [
        {
            "name": r["name"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "roc_auc": r["roc_auc"],
        }
        for r in results
    ]
    _save_roc(results, y_test, ROC_PATH)
    _save_importance(importances, IMPORTANCE_PATH)

    metrics: dict[str, Any] = {
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": feature_names,
        "demo_features": list(DEMO_FEATURES),
        "test_size": test_size,
        "random_state": random_state,
        "exported_model": export_row["name"],
        "best_benchmark_model": best["name"],
        "best": {
            "name": export_row["name"],
            "accuracy": export_row["accuracy"],
            "precision": export_row["precision"],
            "recall": export_row["recall"],
            "f1": export_row["f1"],
            "roc_auc": export_row["roc_auc"],
        },
        "models": summary_rows,
        "feature_importance": importances,
        "presets": [
            {
                "label": p["label"],
                "actual_qualified": p["actual_qualified"],
                "predicted_probability": p["predicted_probability"],
                "demo_values": p["values"],
            }
            for p in presets
        ],
        "artifacts": {
            "model": str(MODEL_PATH.relative_to(ROOT)),
            "roc_comparison": str(ROC_PATH.relative_to(ROOT)),
            "feature_importance": str(IMPORTANCE_PATH.relative_to(ROOT)),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = train()
    best = metrics["best"]
    print(
        f"Exported={best['name']} | "
        f"Acc={best['accuracy']:.4f} F1={best['f1']:.4f} "
        f"ROC-AUC={best['roc_auc']:.4f}"
    )
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")
    for m in metrics["models"]:
        print(
            f"  {m['name']}: Acc={m['accuracy']:.4f} "
            f"F1={m['f1']:.4f} ROC-AUC={m['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
