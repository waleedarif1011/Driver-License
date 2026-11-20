---
title: Driver Qualification Predictor
emoji: 🚗
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# Drivers License — Pre-Test Qualification Signals

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces/waleedarif1011/driver-license)
[![CI](https://github.com/waleedarif1011/driver-license/actions/workflows/ci.yml/badge.svg)](https://github.com/waleedarif1011/driver-license/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Synthetic data (read first).** All numbers in this repo come from `data/driver_license_sample.csv` — a **500-row synthetic stand-in** matching the [Kaggle Drivers License Test Scores](https://www.kaggle.com/) schema so CI and the Gradio demo need no external download. It is **not** production DMV or driving-school data. Treat metrics as portfolio demos, not operational claims.

> **Circularity caveat.** Predicting `Qualified` from the same practical / theory **test scores that largely define qualification** is partly circular. Full-model ROC-AUC ≈ 0.95 is an **upper-bound classification demo**, not evidence of a breakthrough applicant-screening system.

**Preferred framing:** estimate qualification risk from **pre-test signals** available before exams — demographics, training level, and reactions — and treat score columns as diagnostic context only. A LightGBM **ablation without exam scores** is trained in `src/train.py` and recorded under `pre_test_ablation` in [`artifacts/metrics.json`](artifacts/metrics.json).

> **Live demo:** after the Hugging Face Space is created (see [Deploy](#deploy-hugging-face-space)), open the badge URL above. Until then, run locally with `python app.py`.

## Suggested GitHub description / topics

- **Description:** Synthetic driver-license qualification demo — pre-test signals (training, reactions) + honest ablation vs circular test-score features.
- **Topics:** `machine-learning`, `lightgbm`, `classification`, `gradio`, `synthetic-data`, `ablation-study`, `portfolio`

## Results (held-out 20%, `random_state=42`)

### Full feature set (includes circular exam scores — demo upper bound)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.820 | 0.808 | 0.840 | 0.824 | 0.907 |
| Random Forest | 0.880 | 0.896 | 0.860 | 0.878 | 0.951 |
| **LightGBM** | **0.890** | **0.915** | **0.860** | **0.887** | **0.951** |

`LightGBM` is the exported Gradio/CLI model (full feature set for interactive score-card demos).

### Pre-test ablation (no practical / theory scores)

LightGBM trained on **Gender, Age Group, Race, Training, Reactions** only (same split). Expect a large drop — that gap is the point: most of the full-model lift comes from scores that near-define the label.

| Setting | Accuracy | F1 | ROC-AUC |
|---------|----------|----|---------|
| Full LightGBM (with scores) | 0.890 | 0.887 | 0.951 |
| Pre-test only LightGBM | 0.580 | 0.553 | 0.672 |

Full numbers: [`artifacts/metrics.json`](artifacts/metrics.json).

### ROC comparison & feature importance

![ROC comparison](assets/roc_comparison.png)

![Feature importance](assets/feature_importance.png)

Feature importance on the full model is dominated by Theory Test / practical scores — another signal that the circular columns drive the headline metrics.

## Quickstart

```bash
python -m pip install -r requirements.txt
python -m src.train          # train + write artifacts/ (incl. pre_test_ablation) and assets/
python app.py                # Gradio demo at http://127.0.0.1:7860
```

Refresh the bundled sample (optional):

```bash
python scripts/fetch_data.py
```

### CLI inference

```bash
python -m src.predict \
  --gender Female --age-group "Young Adult" --race Other \
  --training Advanced --reactions Fast \
  --signals 80 --yield 75 --speed-control 78 --night-drive 72 \
  --road-signs 80 --steer-control 85 --mirror-usage 70 \
  --confidence 75 --parking 82 --theory-test 92
```

Returns JSON with `badge`, `probability`, and top contributing factors.

## Method (short)

- **Data:** `data/driver_license_sample.csv` — **synthetic** 500 applicants matching the notebook schema (stand-in for the Kaggle CSV).
- **Cleaning:** impute missing `Training` with the mode; coerce numeric score columns.
- **Features (full demo model):** drop `Applicant ID`; one-hot encode `Gender`, `Age Group`, `Race`, `Training`, `Reactions` with `drop_first=True`; keep practical/theory score columns.
- **Features (preferred story / ablation):** pre-test categoricals only — `include_scores=False` in `src/features.py` / `_pre_test_ablation` in `src/train.py`.
- **Split:** 80/20 stratified train/test (`random_state=42`).
- **Models:** Logistic Regression (`liblinear`), Random Forest, LightGBM — LightGBM exported to `artifacts/driver_lgbm.joblib`.
- **Demo:** Gradio form with demographics + score sliders, synthetic-data + circularity disclosure, and 3 held-out presets.

## Exploratory findings (from the notebook)

* Applicants who qualified generally scored higher on practical components, especially Signals, Steer Control, Parking, and Theory Test — expected if those scores define the label.
* Advanced training and Fast reactions correlate with higher qualification rates (the pre-test signal worth highlighting).
* Gender and Race distributions are relatively balanced vs qualification in this sample.

## Tests & CI

```bash
python -m pytest -q
```

GitHub Actions runs the same suite on every push/PR (`.github/workflows/ci.yml`).

## Deploy (Hugging Face Space)

This repo is Space-ready (`app.py` + Gradio README frontmatter). Create and push:

```bash
# once: huggingface-cli login   # needs a write token with Spaces scope
huggingface-cli repo create driver-license --type space --space_sdk gradio -y
git remote add space https://huggingface.co/spaces/waleedarif1011/driver-license
git push space HEAD:main
```

Step-by-step: [`docs/hf-space-deploy.md`](docs/hf-space-deploy.md).

## Layout

```
app.py                 Gradio Driver Qualification Predictor (3 presets)
src/features.py        clean + one-hot encode; include_scores flag for ablation
src/train.py           LR / RF / LightGBM + pre-test ablation metrics/plots
src/predict.py         CLI + probability + top factors
data/                  synthetic driver_license_sample.csv
artifacts/             driver_lgbm.joblib + metrics.json (incl. pre_test_ablation)
assets/                roc_comparison + feature_importance plots
notebooks/             exploratory notebook (local data path)
scripts/fetch_data.py  regenerate the sample CSV
tests/                 pytest suite
```

## License

MIT — see [`LICENSE`](LICENSE).
