# Save this as app.py and run it with: streamlit run app.py

import pandas as pd
import streamlit as st

df = pd.read_csv("mock_claims.csv")
df['charge_amount'] = pd.to_numeric(df['charge_amount'], errors='coerce')
df.dropna(subset=['charge_amount'], inplace=True)

# Aggregations
top_dx = df.groupby('icd10')['charge_amount'].sum().sort_values(ascending=False)
top_cpt = df.groupby('cpt')['charge_amount'].sum().sort_values(ascending=False)
patient_totals = df.groupby('patient_id')['charge_amount'].sum()

# Dashboard UI
st.title("📊 Healthcare Claims Dashboard")

st.subheader("Top Diagnoses by Total Cost")
st.bar_chart(top_dx.head(3))

st.subheader("Top Procedures by Total Cost")
st.bar_chart(top_cpt.head(3))

# Alerts
st.subheader("⚠️ High-Cost Patients")
threshold = st.slider("Set Cost Threshold", min_value=100, max_value=1000, value=500)
alerts = patient_totals[patient_totals > threshold]
st.write(alerts)
