# Hugging Face Space deploy

The Gradio Driver Qualification Predictor is Space-ready. A live Space was **not**
created from this agent run because no Hugging Face write token (`HF_TOKEN`) was
available in the environment.

## Exact steps (Waleed)

1. Create a token at https://huggingface.co/settings/tokens with **Write** + Spaces access.
2. Locally (or in CI):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # paste the token
huggingface-cli repo create driver-license --type space --space_sdk gradio -y
git remote add space https://huggingface.co/spaces/<YOUR_HF_USER>/driver-license
git push space main
```

3. Confirm the Space builds and the three presets behave as expected:
   - Borderline / not ready → Not Qualified (low probability)
   - Average candidate → mid-range probability
   - Strong candidate → Qualified (high probability)
4. Update the Live Demo badge URL in `README.md` if the Space slug differs from
   `waleedarif1011/driver-license`.

## Files the Space needs

- `app.py`, `requirements.txt`, `README.md` (YAML frontmatter)
- `src/`
- `artifacts/` (`driver_lgbm.joblib`, `metrics.json`)
- `assets/` (plots used in the UI / README)
- `data/` (only required if re-training on the Space; inference uses `artifacts/`)
