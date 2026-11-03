import numpy as np
import pandas as pd

BASE = {
    "rhr": (65, 7, 1.8),
    "hrv": (50, 12, 4),
    "spo2": (97, 0.6, 0.4),
    "sleep": (7, 0.6, 0.7),
    "steps": (7000, 2000, 1500),
    "temp": (36.5, 0.12, 0.1),
    "sys_bp": (118, 8, 4),
    "glucose": (95, 8, 6),
}

# Synthetic demo data only. These labels are simulated events, not clinical diagnoses.
VALID_RANGES = {
    "rhr": (30, 200), "hrv": (5, 200), "spo2": (70, 100), "sleep": (0, 14),
    "steps": (0, 40000), "temp": (34, 42), "sys_bp": (70, 230), "glucose": (40, 500),
}

CONDITIONS = {
    "infection": ({"rhr": 9, "hrv": -14, "spo2": -1.5, "sleep": -1.3, "steps": -2500, "temp": 0.5}, (2, 2, 3)),
    "cardiac":   ({"rhr": 6, "hrv": -10, "sys_bp": 14, "steps": -1500, "spo2": -0.5}, (5, 4, 4)),
    "stress":    ({"hrv": -12, "rhr": 4, "sleep": -1.5, "steps": -1000}, (4, 5, 4)),
    "diabetes":  ({"glucose": 35, "steps": -2000, "sleep": -0.6, "rhr": 2}, (6, 5, 5)),
    "apnea":     ({"spo2": -2.2, "sleep": -0.8, "hrv": -6, "sys_bp": 6}, (3, 4, 3)),
}


def simulate_user(uid, days=180, seed=0):
    rng = np.random.default_rng(seed + uid)
    idx = pd.date_range("2026-01-01", periods=days, freq="D")
    d = {}
    for v, (m, sd, noise) in BASE.items():
        d[v] = rng.normal(m, sd) + rng.normal(0, noise, days)
    d["steps"] = np.clip(d["steps"], 500, None)

    labels = {}
    for cond, (effects, (pre, dur, post)) in CONDITIONS.items():
        y = np.zeros(days, int)
        shape = np.r_[np.linspace(0.3, 1, pre), np.ones(dur), np.linspace(1, 0.3, post)]
        for onset in rng.choice(np.arange(20, days - 15), size=rng.integers(1, 4), replace=False):
            start = onset - pre
            for v, delta in effects.items():
                d[v][start:start + len(shape)] += delta * shape
            y[max(0, onset - 3):onset] = 1
        labels[f"y_{cond}"] = y

    # Multiple simulated conditions can overlap, so clamp after all effects.
    # This prevents impossible negative steps/HRV and out-of-range values.
    for vital, (low, high) in VALID_RANGES.items():
        d[vital] = np.clip(d[vital], low, high)

    df = pd.DataFrame({**d, **labels}, index=idx)
    df["user_id"] = uid
    return df.reset_index(names="date")


if __name__ == "__main__":
    data = pd.concat([simulate_user(u) for u in range(80)])
    data.to_csv("data/raw.csv", index=False)
    print("Saved data/raw.csv", data.shape)
