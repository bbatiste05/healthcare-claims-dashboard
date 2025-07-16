import streamlit as st
from load_data import get_dataset
from dashboards import cost_anomalies, fraud_detection, risk_scoring, cpt_charge_audit, no_show_model



st.set_page_config(page_title="Healthcare Claims Dashboard", layout="wide")

st.title("🏥 Healthcare Claims Dashboard")

claims_df = get_dataset()

tab = st.selectbox("Choose a dashboard:", [
    "🧍 Risk Scoring",
    "💰 Cost Anomalies",
    "🕵️ Fraud Detection",
    "💥 CPT Charge Audit",
    "🤖 No-Show Predictor"  # New tab
])

with tab == "🧍 Risk Scoring":
    risk_scoring.run(claims_df)
with tab == "💰 Cost Anomalies":
    cost_anomalies.run(claims_df)
with tab == "🕵️ Fraud Detection":
    fraud_detection.run(claims_df)
with tab == "💥 CPT Charge Audit":
    cpt_charge_audit.run(claims_df)
with tab == "🤖 No-Show Predictor":
    no_show_model.run() 

