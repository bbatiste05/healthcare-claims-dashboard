import pandas as pd
import streamlit as st

@st.cache_data
def load_default_data():
    df = pd.read_csv("mock_claims.csv")
    return clean_data(df)

def clean_data(df):
    df['charge_amount'] = pd.to_numeric(df['charge_amount'], errors='coerce')
    df.dropna(subset=['charge_amount'], inplace=True)
    return df

def get_dataset():
    st.sidebar.subheader("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Custom dataset loaded.")
    else:
        df = load_default_data()
        st.sidebar.info("Using default mock_claims.csv")

    return clean_data(df)
