import numpy as np
import pandas as pd

VITALS = ["rhr", "hrv", "spo2", "sleep", "steps", "temp", "sys_bp", "glucose"]

# Keep these scales aligned with src/risk.py's VITAL_INPUTS[*]["noise"].
# The rolling mean estimates a person's usual level; the fixed scale is used
# by both training and inference so feature units are more consistent.
VITAL_NOISE = {
    "rhr": 1.8,
    "hrv": 4.0,
    "spo2": 0.4,
    "sleep": 0.7,
    "steps": 1500.0,
    "temp": 0.1,
    "sys_bp": 4.0,
    "glucose": 6.0,
}
FEATURES = [f"{v}_{s}" for v in VITALS for s in ("z", "z3", "d1")]


def add_features(df, window=14):
    out = []
    for _, g in df.sort_values("date").groupby("user_id"):
        g = g.copy()
        for v in VITALS:
            usual = g[v].rolling(window, min_periods=7).mean().shift(1)
            g[f"{v}_z"] = (g[v] - usual) / VITAL_NOISE[v]
            g[f"{v}_z3"] = g[f"{v}_z"].rolling(3).mean()
            g[f"{v}_d1"] = g[v].diff()
        out.append(g)
    return pd.concat(out).dropna().reset_index(drop=True)


if __name__ == "__main__":
    f = add_features(pd.read_csv("data/raw.csv", parse_dates=["date"]))
    f.to_csv("data/features.csv", index=False)
    print("Saved data/features.csv", f.shape)
