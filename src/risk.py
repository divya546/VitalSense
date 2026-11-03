from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURES, VITALS

ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ["infection", "cardiac", "stress", "diabetes", "apnea"]
TITLES = {"infection": "Infection / Fever", "cardiac": "Heart & Blood Pressure",
          "stress": "Stress & Burnout", "diabetes": "Blood Sugar / Diabetes",
          "apnea": "Sleep Apnea / Breathing"}

VITAL_INPUTS = {
    "rhr":     dict(label="Resting heart rate (bpm)", short="Heart rate", min=30.0, max=200.0, default=70.0, step=1.0, noise=1.8),
    "hrv":     dict(label="HRV (ms), leave 50 if unknown", short="HRV", min=5.0, max=200.0, default=50.0, step=1.0, noise=4.0),
    "spo2":    dict(label="Blood oxygen SpO2 (%)", short="SpO2", min=70.0, max=100.0, default=97.0, step=0.5, noise=0.4),
    "sleep":   dict(label="Sleep (hours)", short="Sleep", min=0.0, max=14.0, default=7.0, step=0.5, noise=0.7),
    "steps":   dict(label="Daily steps", short="Steps", min=0.0, max=40000.0, default=7000.0, step=500.0, noise=1500.0),
    "temp":    dict(label="Body temperature (°C)", short="Temp", min=34.0, max=42.0, default=36.6, step=0.1, noise=0.1),
    "sys_bp":  dict(label="Systolic BP (mmHg)", short="Systolic BP", min=70.0, max=230.0, default=118.0, step=1.0, noise=4.0),
    "glucose": dict(label="Fasting glucose (mg/dL)", short="Glucose", min=40.0, max=500.0, default=95.0, step=1.0, noise=6.0),
}

SYMPTOMS = ["Fever or chills", "Cough or sore throat", "Chest pain or tightness",
            "Shortness of breath", "Palpitations", "Frequent urination",
            "Excessive thirst", "Blurred vision", "Unusual tiredness",
            "Loud snoring or daytime sleepiness", "Headache or dizziness",
            "Anxiety or low mood"]

RELEVANT = {"infection": ["temp", "rhr", "hrv", "spo2", "sleep", "steps"],
            "cardiac": ["sys_bp", "rhr", "hrv", "spo2", "steps"],
            "stress": ["hrv", "rhr", "sleep", "steps"],
            "diabetes": ["glucose", "steps", "sleep"],
            "apnea": ["spo2", "sleep", "hrv", "sys_bp"]}

ADVICE = {
    "infection": "Rest, drink plenty of fluids and track your temperature. See a doctor if fever lasts more than 2 to 3 days or goes above 39 °C.",
    "cardiac": "Reduce salt, stay active, avoid smoking and check your blood pressure regularly. Get chest pain or breathlessness checked promptly.",
    "stress": "Protect your sleep, take short breaks, try breathing exercises and walk daily. Talk to someone you trust or a counsellor if it persists.",
    "diabetes": "Limit sugary food and drinks, stay active and get an HbA1c or fasting glucose test done at a lab.",
    "apnea": "Keep a regular sleep schedule, avoid alcohol before bed and sleep on your side. Ask a doctor about a sleep study if snoring and tiredness continue.",
}

# Experimental heuristic adjustment; not a clinically validated risk equation.
PRIOR_SHIFT = -1.7
_models = None


def _load_models():
    global _models
    if _models is None:
        _models = joblib.load(ROOT / "models" / "models.joblib")
    return _models


def bmi_of(p):
    return p["weight"] / (p["height"] / 100) ** 2


def _feature_row(usual, readings):
    zs = pd.DataFrame({v: (readings[v] - usual[v]) / VITAL_INPUTS[v]["noise"] for v in VITALS}).clip(-8, 8)
    feat = {}
    for v in VITALS:
        feat[f"{v}_z"] = zs[v].iloc[-1]
        feat[f"{v}_z3"] = zs[v].iloc[-3:].mean()
        feat[f"{v}_d1"] = readings[v].iloc[-1] - readings[v].iloc[-2]
    return pd.DataFrame([feat])[FEATURES], zs


def _context_factors(p):
    s, u, bmi, age = set(p["symptoms"]), p["usual"], bmi_of(p), p["age"]
    f = {c: [] for c in CONDITIONS}

    def add(cond, cond_true, weight, text):
        if cond_true:
            f[cond].append((text, weight))

    add("infection", "Fever or chills" in s, 1.0, "You reported fever or chills")
    add("infection", "Cough or sore throat" in s, 0.6, "You reported cough or sore throat")
    add("infection", "Unusual tiredness" in s, 0.4, "You reported unusual tiredness")
    add("infection", age >= 65, 0.3, "Age 65 or above")

    add("cardiac", age >= 45, 0.4, "Age 45 or above")
    add("cardiac", age >= 60, 0.4, "Age 60 or above")
    add("cardiac", p["smoker"] == "Current", 0.6, "Current smoker")
    add("cardiac", p["smoker"] == "Former", 0.2, "Former smoker")
    add("cardiac", bmi >= 30, 0.4, f"BMI {bmi:.1f} (obesity range)")
    add("cardiac", p["fam_heart"], 0.5, "Family history of heart disease")
    add("cardiac", p["known_bp"], 0.5, "Diagnosed high blood pressure")
    add("cardiac", p["known_diabetes"], 0.4, "Diagnosed diabetes")
    add("cardiac", "Chest pain or tightness" in s, 0.9, "You reported chest pain or tightness")
    add("cardiac", "Palpitations" in s, 0.5, "You reported palpitations")
    add("cardiac", "Shortness of breath" in s, 0.5, "You reported shortness of breath")
    add("cardiac", u["sys_bp"] >= 140, 0.6, f"Usual blood pressure is high ({u['sys_bp']:.0f})")
    add("cardiac", 130 <= u["sys_bp"] < 140, 0.3, f"Usual blood pressure is raised ({u['sys_bp']:.0f})")
    add("cardiac", u["rhr"] > 100, 0.5, f"Usual resting heart rate is high ({u['rhr']:.0f})")

    add("stress", "Anxiety or low mood" in s, 0.9, "You reported anxiety or low mood")
    add("stress", "Unusual tiredness" in s, 0.4, "You reported unusual tiredness")
    add("stress", "Headache or dizziness" in s, 0.3, "You reported headache or dizziness")
    add("stress", "Palpitations" in s, 0.3, "You reported palpitations")
    add("stress", u["sleep"] < 6, 0.4, f"Usual sleep is short ({u['sleep']:.1f} h)")

    add("diabetes", age >= 45, 0.4, "Age 45 or above")
    add("diabetes", 25 <= bmi < 30, 0.3, f"BMI {bmi:.1f} (overweight range)")
    add("diabetes", bmi >= 30, 0.7, f"BMI {bmi:.1f} (obesity range)")
    add("diabetes", p["fam_diabetes"], 0.6, "Family history of diabetes")
    add("diabetes", p["known_diabetes"], 0.8, "Diagnosed diabetes")
    add("diabetes", p["known_bp"], 0.2, "Diagnosed high blood pressure")
    add("diabetes", "Frequent urination" in s, 0.8, "You reported frequent urination")
    add("diabetes", "Excessive thirst" in s, 0.8, "You reported excessive thirst")
    add("diabetes", "Blurred vision" in s, 0.5, "You reported blurred vision")
    add("diabetes", u["glucose"] >= 126, 1.0, f"Usual fasting glucose is in the diabetic range ({u['glucose']:.0f})")
    add("diabetes", 100 <= u["glucose"] < 126, 0.4, f"Usual fasting glucose is above normal ({u['glucose']:.0f})")

    add("apnea", "Loud snoring or daytime sleepiness" in s, 1.0, "You reported loud snoring or daytime sleepiness")
    add("apnea", bmi >= 30, 0.6, f"BMI {bmi:.1f} (obesity range)")
    add("apnea", age >= 40, 0.3, "Age 40 or above")
    add("apnea", p["known_bp"], 0.3, "Diagnosed high blood pressure")
    add("apnea", p["smoker"] == "Current", 0.2, "Current smoker")
    add("apnea", "Headache or dizziness" in s, 0.2, "You reported headache or dizziness")
    add("apnea", "Unusual tiredness" in s, 0.3, "You reported unusual tiredness")
    add("apnea", u["spo2"] < 94, 0.6, f"Usual blood oxygen is low ({u['spo2']:.0f}%)")
    return f


def _level(score):
    # Demo-only thresholds. These bands are not validated clinical cut-offs.
    return "Low" if score < 0.20 else "Moderate" if score < 0.50 else "High"


def safety_alerts(p):
    t = p["readings"].iloc[-1]
    s = set(p["symptoms"])
    alerts = []
    if t["spo2"] < 92:
        alerts.append(f"Blood oxygen is {t['spo2']:.0f}%, which is low.")
    if t["rhr"] > 120 or t["rhr"] < 40:
        alerts.append(f"Resting heart rate of {t['rhr']:.0f} bpm is outside the safe range.")
    if t["sys_bp"] >= 180:
        alerts.append(f"Systolic blood pressure of {t['sys_bp']:.0f} is very high.")
    if t["glucose"] >= 300 or t["glucose"] < 55:
        alerts.append(f"Glucose of {t['glucose']:.0f} mg/dL is dangerous.")
    if t["temp"] >= 39.5:
        alerts.append(f"Temperature of {t['temp']:.1f} °C is a high fever.")
    if "Chest pain or tightness" in s and "Shortness of breath" in s:
        alerts.append("Chest pain together with shortness of breath.")
    return alerts


def assess(p):
    models = _load_models()
    usual, readings = p["usual"], p["readings"]
    row, zs = _feature_row(usual, readings)
    today, factors = readings.iloc[-1], _context_factors(p)
    out = {}
    for c in CONDITIONS:
        base = float(np.clip(models[c].predict_proba(row)[0, 1], 0.005, 0.995))
        boost = sum(w for _, w in factors[c])
        score = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + PRIOR_SHIFT + boost)))
        reasons = []
        for v in sorted(RELEVANT[c], key=lambda k: -abs(zs[k].iloc[-1])):
            z = zs[v].iloc[-1]
            if abs(z) >= 1.5:
                word = "much " if abs(z) >= 3 else ""
                reasons.append(f"{VITAL_INPUTS[v]['short']} is {word}{'higher' if z > 0 else 'lower'} than your usual "
                               f"({today[v]:.1f} vs {usual[v]:.1f})")
        reasons += [text for text, _ in sorted(factors[c], key=lambda x: -x[1])]
        out[c] = dict(score=float(score), level=_level(score), reasons=reasons[:6], advice=ADVICE[c])
    return dict(results=out, z_today=zs.iloc[-1], alerts=safety_alerts(p))
