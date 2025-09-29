# dashboards/copilot_ui.py
import streamlit as st
import pandas as pd

from copilot.agent import handle_query
from copilot.rag import SimpleRAG
from copilot import tools as copilot_tools

def _render_table(df: pd.DataFrame, caption: str):
    st.caption(caption)
    st.dataframe(df, use_container_width=True)

def run(claims_df: pd.DataFrame):
    st.subheader("🗣️ Claims Copilot")
    st.write("Ask natural-language questions about cost drivers, risk, anomalies, and potential fraud. "
             "Answers include evidence references to tables/figures and external code/provider info.")
    
    # Safety toggle (k-anonymity guidance only; enforce in your tool functions if you like)
    phi_safe = st.toggle("PHI-safe mode (aggregate; hide small-N)", value=True)
    st.session_state["phi_safe"] = phi_safe

    # Load simple retriever over external CSVs
    rag = SimpleRAG(base_dir=".")

    user_q = st.chat_input("Try: Which CPTs drove outpatient cost in Q2 2024?")
    if not user_q:
        st.info("Ask a question to get started. Example: 'List providers with Z≥3 on CPT 99213 in TX.'")
        return

    # 1) Get a structured answer (placeholder agent; later swap with a GPT call + tools)
    result = handle_query(user_q, claims_df, rag)

    # 2) Render the summary
    with st.expander("Answer", expanded=True):
        for bullet in result.get("summary", []):
            st.write(f"- {bullet}")

    # 3) Render evidence: recompute referenced tables using your tool wrappers
    #    (This keeps UI consistent and decouples reasoning from display)
    requested_tables = [t.get("name") for t in result.get("tables", [])]
    if "cost_drivers" in requested_tables:
        out = copilot_tools.top_icd_cpt_cost(claims_df)
        _render_table(out["table"], "Cost Drivers (total paid and share)")

    if "provider_outliers" in requested_tables:
        out = copilot_tools.provider_anomalies(claims_df)
        _render_table(out["table"], "Provider Outliers by Z-score")

    if "fraud_flags" in requested_tables:
        out = copilot_tools.fraud_flags(claims_df)
        _render_table(out["table"], "Flagged Providers (excessive claims per patient)")

    if "risk_scores" in requested_tables:
        out = copilot_tools.risk_scoring(claims_df)
        _render_table(out["table"], "Risk Scores (toy logic)")

    # 4) Citations / next steps
    st.markdown("**Citations**")
    st.write(", ".join(result.get("citations", [])) or "—")
    st.markdown("**Next steps**")
    for step in result.get("next_steps", []):
        st.write(f"- {step}")
