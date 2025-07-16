import streamlit as st
from load_data import get_dataset
from dashboards import cost_anomalies, fraud_detection, risk_scoring, cpt_charge_audit, no_show_model



st.set_page_config(page_title="Healthcare Claims Dashboard", layout="wide")

st.title("🏥 Healthcare Claims Dashboard")

df = get_dataset()

tab = st.selectbox("Choose a dashboard:", [
    "🧍 Risk Scoring",
    "💰 Cost Anomalies",
    "🕵️ Fraud Detection",
    "💥 CPT Charge Audit",
    "🤖 No-Show Predictor"  # New tab
])

if tab == "🧍 Risk Scoring":
    risk_scoring.run(df)
elif tab == "💰 Cost Anomalies":
    cost_anomalies.run(df)
elif tab == "🕵️ Fraud Detection":
    fraud_detection.run(df)
elif tab == "💥 CPT Charge Audit":
    cpt_charge_audit.run(df)
elif tab == "🤖 No-Show Predictor":
    no_show_model.run(df) 

