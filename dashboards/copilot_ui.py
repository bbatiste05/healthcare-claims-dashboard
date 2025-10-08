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
        st.write(s)

    # ✅ Display Table (formatted if available)
    if result.get("tables"):
        st.markdown("### 📊 Results Table")
        df_table = pd.DataFrame(result["tables"])
        if not df_table.empty:
            # Display formatted numeric columns cleanly
            st.dataframe(
                df_table.style.format({
                    "Total Cost": "${:,.0f}",
                    "Cost Share (%)": "{:.2f}%"
                })
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

        tables = result["tables"]

        # If it's a list of dicts (rows) → single DataFrame
        if isinstance(tables, list) and all(isinstance(row, dict) for row in tables):
            df = pd.DataFrame(tables)
            df = df.applymap(lambda x: str(x) if isinstance(x, (dict, list)) else x)
            st.dataframe(df, use_container_width=True)

        # If it's already multiple tables, render each
        elif isinstance(tables, list):
            for tbl in tables:
                if isinstance(tbl, list) and all(isinstance(row, dict) for row in tbl):
                    df = pd.DataFrame(tbl)
                    df = df.applymap(lambda x: str(x) if isinstance(x, (dict, list)) else x)
                    st.dataframe(df, use_container_width=True)
                elif isinstance(tbl, dict):
                    df = pd.DataFrame([tbl])
                    df = df.applymap(lambda x: str(x) if isinstance(x, (dict, list)) else x)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.json(tbl)  # fallback


    # 4) Render next steps
    if result.get("next_steps"):
        st.subheader("✅ Next Steps")
        for step in result["next_steps"]:
            st.write("•", step)

    # 5) Render citations
    if result.get("citations"):
        st.subheader("📚 Citations")
        st.write(", ".join(result["citations"]))
