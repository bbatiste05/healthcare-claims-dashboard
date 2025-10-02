# dashboards/copilot_ui.py
import streamlit as st
import pandas as pd

from copilot.agent import ask_gpt
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

    rag = SimpleRAG("data/")

    # 1) Get structured answer
    result = ask_gpt(user_q, claims_df, rag)

    # 2) Render the summary
    summary_text = " ".join(result.get("summary", []))
    st.markdown(f"**Answer**\n\n{summary_text}")
    st.subheader("Answer")
    st.write(" ".join(result.get("summary", [])))




    # 3) Render tables (if any)
    if result.get("tables"):
        st.subheader("📊 Tables")
        for tbl in result["tables"]:
            # If it's a list of dicts, treat as rows
            if isinstance(tbl, list) and all(isinstance(row, dict) for row in tbl):
                st.dataframe(pd.DataFrame(tbl))
            # If it's a dict with metadata
            elif isinstance(tbl, dict) and "name" in tbl:
                st.write(f"**{tbl['name'].capitalize()}** (no row data returned)")
            else:
                st.json(tbl)  # fallback debug

    # 4) Render next steps
        st.subheader("Next Steps")
        for step in result.get("next_steps", []):
            st.write("•", step)

    # 5) Render citations
        st.subheader("Citations")
        st.write(", ".join(result.get("citations"[])))
