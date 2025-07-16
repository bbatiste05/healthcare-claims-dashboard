import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest

def run(df):
    st.subheader("🕵️ Fraud Detection")

    if df is None or df.empty:
        st.warning("No data available for fraud detection.")
        return

    # Inject variation if columns are missing
    if 'wait_days' not in df.columns:
        df['wait_days'] = np.random.randint(1, 15, size=len(df))
    if 'num_procedures' not in df.columns:
        df['num_procedures'] = np.random.randint(1, 6, size=len(df))

    # Aggregate data per provider
    agg = df.groupby('provider_id').agg({
        'charge_amount': 'mean',
        'wait_days': 'mean',
        'num_procedures': 'mean'
    }).reset_index()

    # Avoid training on very small groups
    provider_counts = df['provider_id'].value_counts()
    valid_providers = provider_counts[provider_counts >= 3].index
    agg = agg[agg['provider_id'].isin(valid_providers)]

    if agg.empty:
        st.info("Not enough providers with 3 or more claims for fraud detection.")
        return

    # Train Isolation Forest
    model = IsolationForest(contamination=0.2, random_state=42)
    features = agg[['charge_amount', 'wait_days', 'num_procedures']]
    agg['anomaly'] = model.fit_predict(features)
    agg['anomaly_flag'] = agg['anomaly'].apply(lambda x: "🚨 Suspicious" if x == -1 else "✅ Normal")

    # Display flagged table
    st.write("### 🚩 Flagged Providers")
    flagged = agg[agg['anomaly'] == -1]
    st.dataframe(flagged[['provider_id', 'charge_amount', 'wait_days', 'num_procedures', 'anomaly_flag']])

    # Show all providers on a chart
    st.write("### 📈 Suspicious Claim Volumes by Provider")
    fig = px.scatter(
        agg, 
        x='charge_amount', y='wait_days',
        color='anomaly_flag',
        size='num_procedures',
        hover_data=['provider_id'],
        title="Anomaly Detection for Providers"
    )
    st.plotly_chart(fig, use_container_width=True)



