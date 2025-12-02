# Health Predictor

An interactive Streamlit-based hybrid health prediction dashboard that combines a rule-based decision engine with optional ML model fallbacks. The app provides modular assessments for Heart Disease, Diabetes, Stress / Mental Health, and Fitness / Lifestyle and can generate a downloadable PDF report.

This README was updated to match the repository contents: the primary application entrypoint is `app.py`, a sample dataset `cleaned_health_data.csv` is included, and a pickled model artifact `health_risk_model.pkl` is present. The app also looks for model files under a `models/` directory.

Table of Contents
- Features
- Repository structure
- Requirements
- Quick start
- Usage
  - Running the Streamlit app
  - How the app predicts
  - Adding or updating ML models
  - Sample data
- Infra & development notes
- Data & privacy
- License & contact

Features
- Interactive Streamlit dashboard (single-file app: app.py).
- Rule-based prediction logic for each module (heart, diabetes, stress, fitness).
- Optional ML model integration: app attempts to load pickled models if present.
- PDF generation of the health report with a download button inside the app.
- Sample cleaned dataset included for exploration.

Repository structure (relevant files / directories)
