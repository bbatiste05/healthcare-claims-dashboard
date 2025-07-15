import streamlit as st
from dashboards import cost_anomalies, fraud_detection, risk_scoring

from load_data import get_dataset

st.set_page_config(page_title="Healthcare Claims Dashboard", layout="wide")

st.title("🏥 Healthcare Claims Dashboard")

df = get_dataset()

tab1, tab2, tab3 = st.tabs(["🧍 Risk Scoring", "💰 Cost Anomalies", "🕵️ Fraud Detection"])

with tab1:
    risk_scoring.run(df)

with tab2:
    cost_anomalies.run(df)

with tab3:
    fraud_detection.run(df)
