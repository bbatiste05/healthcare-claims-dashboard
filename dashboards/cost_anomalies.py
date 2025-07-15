import streamlit as st
import plotly.express as px
import numpy as np

def run(df):
    st.subheader("💰 Cost Anomaly Detection")

    provider_avg = df.groupby("provider_id")["charge_amount"].mean()
    z_scores = (provider_avg - provider_avg.mean()) / provider_avg.std()
    outliers = z_scores[abs(z_scores) > 2]

    st.write("### 🧾 Providers with Outlier Average Charges")
    st.dataframe(outliers)

    fig = px.bar(outliers, title="Outlier Providers by Avg Charge (Z-Score > 2)",
                 labels={"value": "Z-Score", "index": "Provider ID"})
    st.plotly_chart(fig, use_container_width=True)
