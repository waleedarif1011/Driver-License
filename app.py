"""Gradio Driver Qualification Predictor."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.features import (
    AGE_GROUP_CHOICES,
    GENDER_CHOICES,
    NUMERIC_COLS,
    RACE_CHOICES,
    REACTIONS_CHOICES,
    TRAINING_CHOICES,
)
from src.predict import load_model, predict
from src.train import IMPORTANCE_PATH, METRICS_PATH, MODEL_PATH, ROC_PATH

ROOT = Path(__file__).resolve().parent

_ARTIFACT = None

NUMERIC_SPECS: dict[str, dict] = {
    "Signals": {"minimum": 0, "maximum": 100, "step": 0.1},
    "Yield": {"minimum": 0, "maximum": 100, "step": 0.1},
    "Speed Control": {"minimum": 0, "maximum": 100, "step": 0.1},
    "Night Drive": {"minimum": 0, "maximum": 100, "step": 0.1},
    "Road Signs": {"minimum": 0, "maximum": 100, "step": 0.1},
    "Steer Control": {"minimum": 0, "maximum": 100, "step": 0.1},
    "Mirror Usage": {"minimum": 0, "maximum": 100, "step": 0.1},
    "Confidence": {"minimum": 0, "maximum": 100, "step": 0.1},
    "Parking": {"minimum": 0, "maximum": 100, "step": 0.1},
    "Theory Test": {"minimum": 40, "maximum": 100, "step": 0.1},
}


def _get_artifact():
    global _ARTIFACT
    if _ARTIFACT is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                "artifacts/driver_lgbm.joblib missing — run `python -m src.train` first."
            )
        _ARTIFACT = load_model(MODEL_PATH)
    return _ARTIFACT


def _preset_choices() -> list[str]:
    art = _get_artifact()
    presets = art.get("presets") or []
    if presets:
        return [p["label"] for p in presets]
    return ["Default medians"]


def _metrics_blurb() -> str:
    if not METRICS_PATH.is_file():
        return "_Train with `python -m src.train` to populate metrics._"
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    best = m.get("best", {})
    rows = "\n".join(
        f"| `{r['name']}` | {r['accuracy']:.3f} | {r['precision']:.3f} | "
        f"{r['recall']:.3f} | {r['f1']:.3f} | {r['roc_auc']:.3f} |"
        for r in m.get("models", [])
    )
    return (
        f"**Exported model:** `{best.get('name', '—')}` "
        f"(Acc={best.get('accuracy', float('nan')):.3f}, "
        f"ROC-AUC={best.get('roc_auc', float('nan')):.3f})\n\n"
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n"
        "|-------|----------|-----------|--------|----|---------|\n"
        f"{rows}"
    )


def _card_html(result: dict) -> str:
    color = "#1b7f5a" if result["qualified"] else "#8b3a3a"
    bg = "#e8f5ef" if result["qualified"] else "#f8eaea"
    factors = result.get("top_factors") or []
    factor_rows = "".join(
        f"<li><code>{d['feature']}</code> (importance {d['importance']:.1f})</li>"
        for d in factors
    )
    return f"""
<div style="border-radius:12px;padding:1.1rem 1.25rem;background:{bg};
            border:2px solid {color};font-family:system-ui,sans-serif;">
  <div style="font-size:0.85rem;letter-spacing:0.04em;text-transform:uppercase;
              opacity:0.75;">Driver qualification</div>
  <div style="font-size:2.2rem;font-weight:700;line-height:1.1;color:#111;">
    {result['badge']}
  </div>
  <div style="margin-top:0.55rem;color:#222;">
    Probability <strong>{result['probability_pct']:.1f}%</strong>
  </div>
  <div style="margin-top:0.9rem;font-size:0.9rem;color:#333;">
    <strong>Top contributing factors</strong>
    <ul style="margin:0.35rem 0 0;padding-left:1.2rem;">{factor_rows}</ul>
  </div>
</div>
"""


def estimate(gender, age_group, race, training, reactions, *scores):
    art = _get_artifact()
    features = {
        "Gender": gender,
        "Age Group": age_group,
        "Race": race,
        "Training": training,
        "Reactions": reactions,
    }
    for name, val in zip(NUMERIC_COLS, scores):
        features[name] = float(val)
    result = predict(features, artifact=art)
    summary = (
        f"### {result['badge']} — **{result['probability_pct']:.1f}%** "
        f"qualification probability\n\n"
        f"Model: `{result['model_name']}`."
    )
    plot = str(IMPORTANCE_PATH) if IMPORTANCE_PATH.is_file() else None
    return summary, _card_html(result), plot


def apply_preset(name: str):
    art = _get_artifact()
    defaults = art.get("feature_defaults") or {}
    presets = art.get("presets") or []
    match = next((p for p in presets if p.get("label") == name), None)
    values = (match or {}).get("values") or defaults

    def pick(key, fallback, choices=None):
        val = values.get(key, defaults.get(key, fallback))
        if choices and val not in choices:
            return fallback
        return val

    outs = [
        pick("Gender", "Female", GENDER_CHOICES),
        pick("Age Group", "Young Adult", AGE_GROUP_CHOICES),
        pick("Race", "Other", RACE_CHOICES),
        pick("Training", "Basic", TRAINING_CHOICES),
        pick("Reactions", "Average", REACTIONS_CHOICES),
    ]
    for col in NUMERIC_COLS:
        spec = NUMERIC_SPECS[col]
        raw = float(values.get(col, defaults.get(col, 50.0)))
        outs.append(min(max(raw, spec["minimum"]), spec["maximum"]))
    return outs


def build_ui() -> gr.Blocks:
    art = _get_artifact()
    defaults = art.get("feature_defaults") or {}
    choices = _preset_choices()

    with gr.Blocks(title="Driver Qualification Predictor") as demo:
        gr.Markdown(
            "# Driver Qualification Predictor\n\n"
            "Enter candidate demographics, training, reactions, and test scores "
            "to estimate **qualification probability** with a LightGBM model.\n\n"
            + _metrics_blurb()
        )
        with gr.Row():
            with gr.Column(scale=3):
                preset = gr.Dropdown(
                    choices=choices,
                    value=choices[0],
                    label="Candidate presets (from held-out split)",
                )
                gender = gr.Dropdown(
                    choices=list(GENDER_CHOICES),
                    value=str(defaults.get("Gender", "Female")),
                    label="Gender",
                )
                age_group = gr.Dropdown(
                    choices=list(AGE_GROUP_CHOICES),
                    value=str(defaults.get("Age Group", "Young Adult")),
                    label="Age group",
                )
                race = gr.Dropdown(
                    choices=list(RACE_CHOICES),
                    value=str(defaults.get("Race", "Other")),
                    label="Race",
                )
                training = gr.Dropdown(
                    choices=list(TRAINING_CHOICES),
                    value=str(defaults.get("Training", "Basic")),
                    label="Training",
                )
                reactions = gr.Dropdown(
                    choices=list(REACTIONS_CHOICES),
                    value=str(defaults.get("Reactions", "Average")),
                    label="Reactions",
                )
                score_inputs = []
                for col in NUMERIC_COLS:
                    spec = NUMERIC_SPECS[col]
                    value = float(defaults.get(col, 50.0))
                    value = min(max(value, spec["minimum"]), spec["maximum"])
                    score_inputs.append(
                        gr.Slider(
                            minimum=spec["minimum"],
                            maximum=spec["maximum"],
                            value=value,
                            step=spec["step"],
                            label=col,
                        )
                    )
                run = gr.Button("Score candidate", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                card = gr.HTML()
                importance = gr.Image(label="Feature importance", type="filepath")
                if ROC_PATH.is_file():
                    gr.Image(
                        value=str(ROC_PATH),
                        label="ROC comparison",
                        type="filepath",
                    )

        inputs = [gender, age_group, race, training, reactions, *score_inputs]
        outputs = [summary, card, importance]
        preset.change(fn=apply_preset, inputs=preset, outputs=inputs)
        run.click(fn=estimate, inputs=inputs, outputs=outputs)
        demo.load(fn=estimate, inputs=inputs, outputs=outputs)
        demo.load(fn=apply_preset, inputs=preset, outputs=inputs)
    return demo


def main() -> None:
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
