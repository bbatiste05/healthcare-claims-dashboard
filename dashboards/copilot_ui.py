# dashboards/copilot_ui.py
import streamlit as st
import pandas as pd

from copilot.agent import ask_gpt
from copilot.rag import SimpleRAG
from copilot import tools as copilot_tools


def run(claims_df):
    st.header("🚀 Healthcare Claims Copilot")

    user_q = st.text_input("Ask a question about claims data:")
    if not user_q:
        return

    rag = SimpleRAG("data/")

    # 1) Get structured answer
    result = ask_gpt(user_q, claims_df, rag)

    # --- Display the Copilot's response ---
    st.markdown("### 📝 Summary")
    for s in result.get("summary", []):
        formatted_summary = s.replace(". ", ". \n\n")
        st.markdown(formatted_summary)

    # ✅ Display Table (formatted if available)
    if result.get("tables"):
        st.markdown("### 📊 Results Table")
        df_table = pd.DataFrame(result["tables"])
        if not df_table.empty:
            if "Table" in df_table.columns:
                df_table = df_table[df_table["Table"] != "top_cpt_code_cost"]

            for cpt_col in ["CPT", "CPT Code", "CPTCode"]:
                if cpt_col in df_table.columns:
                    df_table[cpt_col] = df_table[cpt_col].astype(str).str.replace(r"\.0+$", "", regex=True)

            df_table = df_table.fillna("")
            # Display formatted numeric columns cleanly
            rename_map = {
                "CPT": "CPT Code",
                "CPTCode": "CPT Code",
                "icd": "ICD-10 Code",
                "cost": "Total Cost",
                "Cost": "Total Cost",
                "Cost Share (%)": "Cost Share (%)"
            }
            df_table.rename(columns=rename_map, inplace=True)

    st.dataframe(
        df_table.style.format({
            "Total Cost": "${:,.0f}",
            "Cost Share (%)": "{:.2f}%"
        }),
        use_container_width=True
    )    

    # ✅ Display Next Steps if present
    if result.get("next_steps"):
        st.markdown("### ✅ Next Steps")
        for step in result["next_steps"]:
            st.write(f"- {step}")

    # ✅ Optional: Show citations if you use RAG context
    if result.get("citations"):
        st.markdown("### 📚 Citations")
        for c in result["citations"]:
            st.write(f"- {c}")
