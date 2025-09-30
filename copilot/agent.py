
# copilot/agent.py
import json
import pandas as pd
from .prompts import SYSTEM_PROMPT, FEW_SHOTS
from .tools import top_icd_cpt_cost, provider_anomalies, fraud_flags, risk_scoring
from .rag import SimpleRAG
import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPEN_API_KEY"])
model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

def handle_query(user_q: str, df: pd.DataFrame, rag: SimpleRAG):
    """Very simple intent routing without LLM calls—acts as a placeholder for your GPT."
    Returns a dict in the target JSON schema.
    """
    q = user_q.lower()
    result = {"summary": [], "tables": [], "figures": [], "citations": [], "next_steps": []}

    if any(w in q for w in ["cost", "driver", "cpt", "icd"]):
        res = top_icd_cpt_cost(df)
        result["summary"].append("Top cost drivers computed from current dataset.")
        result["tables"].append({"name": res["table_name"]})
        result["citations"].append("claims.table.cost_drivers")
        result["next_steps"].append("Filter by quarter or plan for more specificity.")
        return result

    if "z-score" in q or "outlier" in q or "anomal" in q:
        res = provider_anomalies(df)
        result["summary"].append("Provider outliers identified using Z-scores on total paid.")
        result["tables"].append({"name": res["table_name"]})
        result["citations"].append("claims.table.provider_outliers")
        result["next_steps"].append("Drill into specific CPT to validate anomaly.")
        return result

    if "fraud" in q or "excessive" in q:
        res = fraud_flags(df)
        result["summary"].append("Providers flagged for excessive claims per patient.")
        result["tables"].append({"name": res["table_name"]})
        result["citations"].append("claims.table.fraud_flags")
        result["next_steps"].append("SIU review recommended; confirm medical necessity.")
        return result

    if "risk" in q:
        res = risk_scoring(df)
        result["summary"].append("Patient risk scores calculated (toy logic).")
        result["tables"].append({"name": res["table_name"]})
        result["citations"].append("claims.table.risk_scores")
        result["next_steps"].append("Integrate HCC mappings for clinical validity.")
        return result

    # default
    result["summary"].append("No specific intent matched; try asking about cost drivers, outliers, fraud, or risk.")
    return result
