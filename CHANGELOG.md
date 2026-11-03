# Improvements in this version

- Added a teal/blue dashboard theme and more readable form/report layouts.
- Hid Streamlit's Deploy button through `.streamlit/config.toml`.
- Changed the recent-readings table to initialize from the entered usual values.
- Added prominent notices that this is a prototype trained on synthetic data.
- Reworded result percentages as prototype scores, not disease probabilities.
- Added urgent-symptom caution text and improved the report download name.
- Fixed the synthetic-data generator to clamp measurements to plausible input ranges.
- Aligned training feature scaling with the inference noise scales and regenerated the synthetic dataset, model bundle, and evaluation results.
- Pinned core ML package versions to match the regenerated serialized model.
