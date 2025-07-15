import streamlit as st
import plotly.express as px

def run(df):
    st.subheader("🕵️ Suspicious Activity Detector")

    claim_counts = df.groupby(['provider_id', 'patient_id']).size().reset_index(name='claim_count')
    high_volume = claim_counts[claim_counts['claim_count'] > 3]

    st.write("### 🚩 Providers with >3 Claims Per Patient")
    st.dataframe(high_volume)

    provider_summary = high_volume.groupby("provider_id")["claim_count"].sum()

    fig = px.bar(provider_summary, title="Suspicious Claim Volumes by Provider",
                 labels={"value": "Total Claims", "index": "Provider ID"})
    st.plotly_chart(fig, use_container_width=True)
