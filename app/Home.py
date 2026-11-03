import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.risk import SYMPTOMS, VITAL_INPUTS

st.set_page_config(page_title="VitalSense | Health Dashboard", page_icon="🩺", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1400px;}
  #MainMenu {visibility: hidden;}
  header [data-testid="stToolbar"] {visibility: hidden; height: 0;}
  [data-testid="stAppDeployButton"] {display: none !important;}
  footer {visibility: hidden;}
  section[data-testid="stSidebar"] {background: linear-gradient(180deg,#effaff,#f8fbff); border-right: 1px solid #dbeafe;}
  .hero {
    padding: 1.6rem 1.8rem; border-radius: 22px;
    background: linear-gradient(115deg, #087f8c 0%, #168aad 48%, #3978f6 100%);
    color: white; margin-bottom: 1.2rem;
  }
  .hero h1 {color: white; margin: 0 0 .35rem 0; font-size: 2.4rem;}
  .hero p {color: #e0f2fe; margin: 0; font-size: 1.05rem;}
  div[data-testid="stForm"] {
    border: 1px solid #dbeafe; border-radius: 18px; padding: 1.2rem;
    background: linear-gradient(180deg, rgba(240,249,255,.85), rgba(255,255,255,.98));
    box-shadow: 0 5px 22px rgba(30,64,175,.05);
  }
  div[data-testid="stMetric"] {background:#f0f9ff;border:1px solid #dbeafe;border-radius:14px;padding:12px;}
  div.stButton > button, div[data-testid="stFormSubmitButton"] button {background:linear-gradient(100deg,#3978f6,#10b981);color:white;border:0;}
  h2, h3 {color: #0f766e;}
  div.stButton > button, div[data-testid="stFormSubmitButton"] button {
    border-radius: 10px; font-weight: 650; min-height: 2.8rem;
  }
  div[data-testid="stAlert"] {border-radius: 12px;}
  .small-note {color: #475569; font-size: .92rem;}
</style>
<div class="hero">
  <h1>🩺 VitalSense</h1>
  <p>Your health readings, trends and screening summary in one place.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Prototype only: this app uses a model trained on simulated data and unvalidated rules. "
    "Its scores are not medical probabilities and must not be used to diagnose, rule out, or treat a condition."
)

st.markdown("### Health assessment")
st.caption("Add your details, health history, usual measurements and recent readings. Use measured values whenever possible.")

with st.form("intake"):
    st.subheader("1. Personal information")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Name (optional)")
    age = c2.number_input("Age", 5, 100, 30)
    sex = c3.selectbox("Sex", ["Female", "Male", "Other", "Prefer not to say"])
    c1, c2, c3 = st.columns(3)
    height = c1.number_input("Height (cm)", 100, 220, 165)
    weight = c2.number_input("Weight (kg)", 25, 250, 65)
    smoker = c3.selectbox("Smoking", ["Never", "Former", "Current", "Prefer not to say"])

    st.divider()
    st.subheader("2. Medical history")
    c1, c2 = st.columns(2)
    known_bp = c1.checkbox("I have been diagnosed with high blood pressure")
    known_diabetes = c1.checkbox("I have been diagnosed with diabetes")
    fam_heart = c2.checkbox("Family history of heart disease")
    fam_diabetes = c2.checkbox("Family history of diabetes")

    st.divider()
    st.subheader("3. Usual health readings")
    st.caption("Use typical readings from when you feel well. Defaults are examples, not measured personal values.")
    usual = {}
    cols = st.columns(4)
    for i, (k, cfg) in enumerate(VITAL_INPUTS.items()):
        usual[k] = cols[i % 4].number_input(
            cfg["label"], min_value=float(cfg["min"]), max_value=float(cfg["max"]),
            value=float(cfg["default"]), step=float(cfg["step"]), key=f"usual_{k}"
        )

    st.divider()
    st.subheader("4. Recent readings — last 3 days")
    st.caption("Enter measurements for each day. The table starts with your usual values as a convenience—replace them with actual readings.")
    template = pd.DataFrame(
        {k: [usual[k]] * 3 for k in VITAL_INPUTS},
        index=["2 days ago", "Yesterday", "Today"]
    )
    readings = st.data_editor(
        template,
        column_config={
            k: st.column_config.NumberColumn(
                cfg["short"], min_value=float(cfg["min"]), max_value=float(cfg["max"]),
                step=float(cfg["step"]), format="%.1f" if cfg["step"] < 1 else "%g"
            )
            for k, cfg in VITAL_INPUTS.items()
        },
        use_container_width=True,
        key="recent_readings",
    )

    st.divider()
    st.subheader("5. Symptoms")
    symptoms = st.multiselect("Select any that apply", SYMPTOMS)

    submitted = st.form_submit_button("Get health risk assessment", type="primary", use_container_width=True)

if submitted:
    if readings.isna().any().any():
        st.error("Please fill every cell in the recent-readings table.")
    else:
        st.session_state["patient"] = dict(
            name=name.strip(), age=int(age), sex=sex, height=float(height), weight=float(weight),
            smoker=smoker, known_bp=known_bp, known_diabetes=known_diabetes,
            fam_heart=fam_heart, fam_diabetes=fam_diabetes,
            usual=usual, readings=readings.astype(float).reset_index(drop=True), symptoms=symptoms
        )
        st.switch_page("pages/1_Risk_Report.py")

st.divider()
st.caption("Important: this tool is for educational screening only and is not a medical diagnosis. For severe chest pain, difficulty breathing, fainting, confusion, or rapidly worsening symptoms, seek urgent medical care.")
