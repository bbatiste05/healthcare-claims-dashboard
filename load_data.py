import pandas as pd
import streamlit as st

def clean_data(df):
    df['charge_amount'] = pd.to_numeric(df['charge_amount'], errors='coerce')
    df.dropna(subset=['charge_amount'], inplace=True)
    return df

def get_dataset():
    st.sidebar.subheader("Upload Your Claims Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Validate required columns for claims
        if 'charge_amount' in df.columns and 'provider_id' in df.columns:
            st.sidebar.success("✅ Claims dataset loaded.")
            return clean_data(df)
        else:
            st.sidebar.error("❌ Invalid dataset. Expected columns: 'charge_amount', 'provider_id'")
            st.stop()
    else:
        st.sidebar.warning("⚠️ Please upload a claims dataset to proceed.")
        st.stop()

