# dashboards/copilot_ui.py
import streamlit as st
import pandas as pd

from copilot.agent import ask_gpt as handle_query
from copilot.rag import SimpleRAG
from copilot import tools as copilot_tools

def _render_table(df: pd.DataFrame, caption: str):
    st.caption(caption)
    st.dataframe(df, use_container_width=True)

def run(claims_df):
    st.header("🚀 Healthcare Claims Copilot")

    user_q = st.text_input("Ask a question about claims data:")
    if not user_q:
        return

    rag = SimpleRAG(claims_df)

    # 1) Get structured answer
    result = ask_gpt(user_q, claims_df, rag)

    # 2) Render the summary
    st.subheader("Answer")
    summary_text = " ".join(result.get("summary", []))
    if summary_text.strip():
        st.markdown(summary_text)
    else:
        st.info("No summary available.")

    # 3) Render tables (if any)
    if result.get("tables"):
        for table in result["tables"]:
            st.subheader(f"Table: {table.get('name', 'Unnamed')}")
            if "rows" in table:
                st.table(table["rows"])
            else:
                st.write("⚠️ No rows found for this table.")

    # 4) Render next steps
    if result.get("next_steps"):
        st.subheader("Next Steps")
        for step in result["next_steps"]:
            st.write("•", step)

    # 5) Render citations
    if result.get("citations"):
        st.subheader("Citations")
        st.write(", ".join(result["citations"]))
