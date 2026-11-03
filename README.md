# VitalSense — experimental health-screening prototype

VitalSense is a Streamlit educational project that explores changes in self-entered vital signs and symptoms.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m src.simulate
python -m src.features
python -m src.train
streamlit run app/Home.py
```

Run these commands from the project root. The simulation, feature generation, and training commands rebuild the synthetic dataset and model artifacts.

## Important limitations

- `src/simulate.py` generates **synthetic data**. The condition labels are simulated events, not diagnoses from real patients.
- The reported ROC-AUC / PR-AUC values measure performance on held-out simulated users. They do **not** demonstrate accuracy in real clinical settings.
- The model score and the Low / Moderate / High bands are experimental software outputs, not calibrated disease probabilities or validated clinical cut-offs.
- The symptom/history adjustments in `src/risk.py` are hand-authored heuristics, not clinically validated risk equations.
- Do not use this project to diagnose, rule out, monitor, or treat disease. Seek qualified medical advice for health concerns; urgent symptoms require urgent care regardless of app output.

## Project structure

- `app/Home.py` — patient inputs
- `app/pages/1_Risk_Report.py` — prototype report and download
- `src/simulate.py` — synthetic data generation with plausible value bounds
- `src/features.py` — feature engineering shared by model training
- `src/train.py` — model training and simulated-data evaluation
- `src/risk.py` — inference and hand-authored context adjustments
- `models/models.joblib` — serialized model bundle
- `results/` — evaluation output for the current synthetic run

## Privacy and deployment

This version does not intentionally write the submitted patient form to a database, but hosting platforms and infrastructure may have their own logging and retention behavior. Do not enter real identifiable or sensitive health information into a public demo. The app is not clinically validated or ready for real-patient use.
