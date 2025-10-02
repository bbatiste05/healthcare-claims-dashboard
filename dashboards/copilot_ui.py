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
    st.subheader("Answer")
    summaries = result.get("summary", [])
    if summaries:
        for s in summaries:
            st.markdown(s)
    else:
        st.info("No summary available.")

    # 3) Render tables (if any)
    if result.get("tables"):
        st.subheader("📊 Tables")
        for tbl in result["tables"]:
            if isinstance(tbl, list):  # list of row dicts
                st.dataframe(pd.DataFrame(tbl))
            elif isinstance(tbl, dict) and "name" in tbl:
                st.markdown(f"**{tbl['name'].capitalize()}**")


    # 4) Render next steps
    if result.get("next_steps"):
        st.subheader("Next Steps")
        for step in result["next_steps"]:
            st.write("•", step)

    # 5) Render citations
    if result.get("citations"):
        st.subheader("Citations")
        st.write(", ".join(result["citations"]))
