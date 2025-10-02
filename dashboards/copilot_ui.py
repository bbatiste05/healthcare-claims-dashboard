# dashboards/copilot_ui.py
import streamlit as st
import pandas as pd

from copilot.agent import ask_gpt
from copilot.rag import SimpleRAG


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
            try:
                if isinstance(tbl, list) and all(isinstance(row, dict) for row in tbl):
                    df = pd.DataFrame(tbl)
                    df = df.applymap(lambda x: str(x) if isinstance(x, (dict, list)) else x)
                    st.dataframe(df, use_container_width=True)

                elif isinstance(tbl, dict):
                    df = pd.DataFrame([tbl])
                    df = df.applymap(lambda x: str(x) if isinstance(x, (dict, list)) else x)
                    st.dataframe(df, use_container_width=True)

                else:
                    st.json(tbl)  # fallback debug

            except Exception as e:
                st.error(f"⚠️ Could not render table: {e}")
                st.json(tbl)

    # 4) Render next steps
    if result.get("next_steps"):
        st.subheader("✅ Next Steps")
        for step in result["next_steps"]:
            st.write("•", step)

    # 5) Render citations
    if result.get("citations"):
        st.subheader("📚 Citations")
        st.write(", ".join(result["citations"]))
