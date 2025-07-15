import streamlit as st
import pandas as pd
import plotly.express as px

def run(df):
    st.subheader("💥 CPT Charge Audit: Flag Overpriced Procedures by Provider")

    # Group and flag CPT avg charges per provider
    cpt_avg = df.groupby(['provider_id', 'cpt'])['charge_amount'].mean().reset_index()
    cpt_avg['avg_charge'] = cpt_avg['charge_amount'].round(2)

    # Add flags
    cpt_avg['charge_flag'] = cpt_avg['avg_charge'].apply(
        lambda x: '⚠️ Overpriced' if x > 300 else '✅ Normal'
    )

    # Merge provider name
    if 'provider_id' in df.columns and 'provider_id' in cpt_avg.columns:
        providers = df[['provider_id', 'provider_id']].drop_duplicates()
        merged = pd.merge(cpt_avg, df[['provider_id', 'provider_id', 'provider_name']].drop_duplicates(), on='provider_id', how='left')
        cpt_avg['provider_name'] = merged['provider_name']
    else:
        cpt_avg['provider_name'] = "Unknown"

    st.write("### ⚠️ Flagged CPTs by Provider (Over $300 Avg)")
    st.dataframe(cpt_avg.sort_values('avg_charge', ascending=False))

    # Plot
    fig = px.bar(
        cpt_avg,
        x='cpt',
        y='avg_charge',
        color='charge_flag',
        facet_col='provider_name',
        title="Avg CPT Charges per Provider (Flagged)",
        labels={'avg_charge': 'Average Charge', 'cpt': 'CPT Code'},
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
