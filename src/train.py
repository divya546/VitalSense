from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.features import FEATURES

ROOT = Path(__file__).resolve().parent.parent
CONDS = ["infection", "cardiac", "stress", "diabetes", "apnea"]
df = pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["date"])
X, groups = df[FEATURES], df["user_id"]


def make_model():
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200,
                                          class_weight="balanced", random_state=0)


models, rows = {}, []
for c in CONDS:
    y = df[f"y_{c}"]
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(X, y, groups):
        oof[te] = make_model().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    rule = df["rhr_z"] if c in ("infection", "cardiac") else (
        -df["hrv_z"] if c in ("stress", "apnea") else df["glucose_z"])
    rows.append({"condition": c,
                 "Model ROC-AUC": roc_auc_score(y, oof),
                 "Model PR-AUC": average_precision_score(y, oof),
                 "Rule ROC-AUC": roc_auc_score(y, rule.fillna(0))})
    models[c] = make_model().fit(X, y)

res = pd.DataFrame(rows).round(3)
print("NOTE: these metrics are evaluated on simulated data only; they do not measure clinical accuracy.")
print(res.to_string(index=False))
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
res.to_csv(ROOT / "results" / "results.csv", index=False)
joblib.dump(models, ROOT / "models" / "models.joblib")

ax = res.set_index("condition")[["Model ROC-AUC", "Rule ROC-AUC"]].plot.bar(figsize=(7, 4))
ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.4, 1); plt.xticks(rotation=0)
plt.title("Personalised model vs simple rule"); plt.tight_layout()
plt.savefig(ROOT / "results" / "comparison.png", dpi=150)
