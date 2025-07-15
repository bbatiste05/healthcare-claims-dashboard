import streamlit as st
import plotly.express as px

def run(df):
    st.subheader("🧍 High-Risk Patient Identification")

    icd_risk = {
        "E11.9": 3,    # Type 2 Diabetes
        "I10": 2,      # Hypertension
        "E11.65": 4,   # Diabetes w/ complications
        "A41.9": 5     # Sepsis
    }

    df["risk_score"] = df["icd10"].map(icd_risk).fillna(1)
    patient_risk = df.groupby("patient_id")["risk_score"].sum().sort_values(ascending=False).head(10)

    st.write("### 🔢 Top 10 Highest-Risk Patients")
    st.dataframe(patient_risk)

    fig = px.bar(patient_risk, orientation='h', title="Top Patient Risk Scores", labels={"value": "Risk Score", "index": "Patient ID"})
    st.plotly_chart(fig, use_container_width=True)
