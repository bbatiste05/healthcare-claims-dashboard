import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest


def run(df):
    st.subheader("🕵️ AI-Based Fraud Detection")

    if df is None or df.empty:
        st.error("🚫 No data provided.")
        return

    df = df.copy()

    # Simulate wait_days if not present
    if 'wait_days' not in df.columns:
        st.warning("⏳ 'wait_days' not found — simulating random values.")
        df['wait_days'] = pd.Series([abs(i % 10) + 1 for i in range(len(df))])

    # Simulate num_procedures if not present
    if 'num_procedures' not in df.columns:
        if 'cpt' in df.columns:
            df['num_procedures'] = df.groupby('provider_id')['cpt'].transform('count')
        else:
            df['num_procedures'] = 1  # fallback default

    # Drop rows with missing charge_amount or provider_id
    df = df.dropna(subset=['provider_id', 'charge_amount'])

    # Group by provider
    agg = df.groupby('provider_id').agg({
        'charge_amount': 'mean',
        'wait_days': 'mean',
        'num_procedures': 'mean'
    }).reset_index()

    agg.columns = ['provider_id', 'avg_charge', 'avg_wait', 'avg_procedures']

    # Run Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    agg['anomaly_score'] = model.fit_predict(agg[['avg_charge', 'avg_wait', 'avg_procedures']])
    agg['flagged'] = agg['anomaly_score'].apply(lambda x: '🚩 Anomaly' if x == -1 else '✅ Normal')

    st.write("### 📋 Provider Summary")
    st.dataframe(agg)

    st.write("### 📈 Charges vs Wait Time")
    fig = px.scatter(
        agg,
        x='avg_charge',
        y='avg_wait',
        color='flagged',
        hover_data=['provider_id', 'avg_procedures'],
        title="Provider Anomaly Detection (Isolation Forest)"
    )
    st.plotly_chart(fig, use_container_width=True)

