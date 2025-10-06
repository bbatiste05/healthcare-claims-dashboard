import pandas as pd
import streamlit as st

def clean_data(df):
    df['charge_amount'] = pd.to_numeric(df['charge_amount'], errors='coerce')
    df.dropna(subset=['charge_amount'], inplace=True)
    return df

def get_dataset():
    """Loads user-uploaded CSV and normalizes column names for consistency."""
    st.sidebar.subheader("📁 Upload Your Claims Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # ✅ Normalize column names (strip + lowercase)
        df.columns = df.columns.str.strip().str.lower()

        # ✅ Standardize column naming so tools work correctly
        rename_map = {
            "service_date": "claim_date",   # Used by fraud_flags()
            "member_id": "patient_id",
            "provider": "provider_id",
            "providerid": "provider_id",
        }
        df = df.rename(columns=rename_map)

        # ✅ Check for required columns
        required = {"charge_amount", "provider_id"}
        missing = required - set(df.columns)
        if missing:
            st.sidebar.error(f"❌ Missing required columns: {', '.join(missing)}")
            st.stop()

        # ✅ Optional sidebar feedback
        renamed = [f"{k} → {v}" for k, v in rename_map.items() if k in df.columns]
        if renamed:
            st.sidebar.info(f"🔄 Renamed columns: {', '.join(renamed)}")
        else:
            st.sidebar.success("✅ Column names already standardized.")

        df = clean_data(df)

        # ✅ Optional data preview
        st.sidebar.write("📊 Sample data preview:")
        st.sidebar.dataframe(df.head(), use_container_width=True)

        return df

    else:
        st.sidebar.warning("⚠️ Please upload a claims dataset to proceed.")
        st.stop()
