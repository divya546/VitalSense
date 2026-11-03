import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from src.risk import TITLES, VITAL_INPUTS, assess, bmi_of

st.set_page_config(page_title="VitalSense | Prototype Report", page_icon="🩺", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1400px;}
  #MainMenu {visibility:hidden;}
  header [data-testid="stToolbar"] {visibility:hidden;height:0;}
  [data-testid="stAppDeployButton"] {display:none !important;}
  footer {visibility:hidden;}
  .hero {
    padding: 1.35rem 1.6rem; border-radius: 20px;
    background: linear-gradient(120deg, #0f766e 0%, #155e75 55%, #1d4ed8 100%);
    color: white; margin-bottom: 1rem;
  }
  .hero h1 {color: white; margin: 0 0 .25rem 0;}
  .hero p {color: #e0f2fe; margin: 0;}
  h2, h3 {color: #0f766e;}
  div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px;}
  div[data-testid="stMetric"] {
    background: #f0fdfa; padding: .8rem 1rem; border: 1px solid #ccfbf1; border-radius: 14px;
  }
  div.stButton > button {border-radius: 10px; min-height: 2.7rem;}
</style>
""", unsafe_allow_html=True)

p = st.session_state.get("patient")
if p is None:
    st.markdown('<div class="hero"><h1>🩺 VitalSense report</h1><p>No screening details have been submitted yet.</p></div>', unsafe_allow_html=True)
    st.info("Enter your details first to create a prototype report.")
    if st.button("Go to patient details", type="primary"):
        st.switch_page("Home.py")
    st.stop()

report = assess(p)
res = report["results"]

st.markdown(
    f'<div class="hero"><h1>🩺 VitalSense report</h1><p>For {p["name"] if p["name"] else "the person entered"} · Experimental screening summary</p></div>',
    unsafe_allow_html=True
)
st.error(
    "Important: this is an educational prototype. Its model was trained on simulated data, "
    "and the score bands use unvalidated rules. Scores are not calibrated probabilities of disease "
    "and cannot replace medical assessment."
)

for alert in report["alerts"]:
    st.warning(f"Please take this reading seriously: {alert} If you feel unwell or the reading is unexpected, repeat it correctly and contact a healthcare professional; seek urgent care for severe symptoms.")

top = max(res, key=lambda c: res[c]["score"])
c1, c2, c3 = st.columns(3)
c1.metric("Age entered", f'{p["age"]} years')
c2.metric("BMI estimate", f"{bmi_of(p):.1f}")
c3.metric("Highest prototype signal", TITLES[top] if res[top]["level"] != "Low" else "No elevated signal by demo thresholds")
st.caption("BMI is a rough screening measure and does not diagnose health conditions. The signal bands below are illustrative software thresholds, not clinical cut-offs.")

st.subheader("Screening summary by category")
for c, r in sorted(res.items(), key=lambda kv: -kv[1]["score"]):
    with st.container(border=True):
        a, b = st.columns([3, 1])
        a.markdown(f"**{TITLES[c]}**")
        b.markdown(f"**{r['level']} signal**")
        st.progress(min(max(float(r["score"]), 0.0), 1.0), text=f'Prototype score: {r["score"] * 100:.0f}/100 · not a disease probability')
        with st.expander("What influenced this score?"):
            if r["reasons"]:
                for reason in r["reasons"]:
                    st.write("•", reason)
            else:
                st.write("No large deviation from the entered usual values was detected by the prototype rules.")
            st.info("Prototype suggestion: " + r["advice"])

st.subheader("Reading trends")
trend = p["readings"].copy()
trend.index = ["2 days ago", "Yesterday", "Today"]
trend = trend.rename(columns={k: VITAL_INPUTS[k]["short"] for k in VITAL_INPUTS})
left, right = st.columns(2)
with left:
    st.markdown("**Your readings over three days**")
    st.line_chart(trend, use_container_width=True)
with right:
    st.markdown("**Today compared with your usual values**")
    dev = report["z_today"].rename(lambda k: VITAL_INPUTS[k]["short"])
    st.bar_chart(dev.rename("Relative deviation"), use_container_width=True)
st.caption("Charts show only the measurements entered in this app. Differences and screening scores are not diagnoses or calibrated disease probabilities.")

lines = [
    f"VitalSense prototype report for {p['name'] or 'patient'}",
    "IMPORTANT: Experimental model trained on simulated data. Scores are not clinical probabilities.",
    ""
]
for c, r in res.items():
    lines.append(f"{TITLES[c]}: {r['level']} signal (prototype score {r['score'] * 100:.0f}/100; not a disease probability)")
    lines += [f"  - {reason}" for reason in r["reasons"]]
lines += ["", "Not a medical diagnosis. Seek professional medical advice for health concerns."]
st.download_button("Download report (.txt)", "\n".join(lines), file_name="vitalsense_prototype_report.txt", mime="text/plain")

c1, c2 = st.columns(2)
with c1:
    if st.button("Edit details", use_container_width=True):
        st.switch_page("Home.py")
with c2:
    if st.button("Clear this session's report", use_container_width=True):
        st.session_state.pop("patient", None)
        st.rerun()

st.caption("If you have severe chest pain, difficulty breathing, fainting, confusion, or rapidly worsening symptoms, seek urgent medical care regardless of this report.")
