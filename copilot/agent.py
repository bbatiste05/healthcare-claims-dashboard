# copilot/agent.py
import json
import pandas as pd
import streamlit as st
import openai
from typing import Dict, Any, List
from openai import OpenAI

from .prompts import SYSTEM_PROMPT, FEW_SHOTS
from .tools import top_icd_cpt_cost, provider_anomalies, fraud_flags, risk_scoring
from .rag import SimpleRAG




# ------------------------------
# 1. Define available tool schema
# ------------------------------
def _tools_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "top_icd_cpt_cost",
                "description": "Find top cost drivers filtered by ICD/CPT/period/plan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "icd": {"type": "string", "nullable": True},
                        "cpt": {"type": "string", "nullable": True},
                        "period": {"type": "string", "nullable": True},
                        "plan": {"type": "string", "nullable": True},
                        "top_n": {"type": "integer", "default": 10}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "provider_anomalies",
                "description": "Compute provider outliers using Z-scores on charge amounts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "nullable": True},
                        "metric": {"type": "string", "enum": ["z"], "default": "z"},
                        "threshold": {"type": "number", "default": 3.0}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fraud_flags",
                "description": "Flag providers with excessive claims per patient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min_claims_per_patient": {"type": "integer", "default": 5},
                        "window_days": {"type": "integer", "default": 90}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "risk_scoring",
                "description": (
                    "Compute patient risk scores or analyze wait times by ICD, cohort, or patient. "
                    "Handles questions mentioning 'risk', 'wait', 'delay', 'ICD', or 'patients'."
                ),    
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cohort": {"type": "string", "nullable": True},
                        "top_n": {"type": "integer", "defauly": 10}
                    }
                }
            }
        }
    ]


# ------------------------------
# 2. Build messages
# ------------------------------
def _messages(user_q: str, rag: SimpleRAG) -> list:
    snippets = rag.search(user_q, k=5)
    snip_text = "\n".join([json.dumps(s, ensure_ascii=False) for s in snippets])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"External context (ICD/CPT/NPPES snippets):\n{snip_text}"}
    ]

    # few-shot examples
    for ex in FEW_SHOTS:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})

    # user question
    messages.append({"role": "user", "content": user_q})
    return messages


# ------------------------------
# 3. Call local Python tools
# ------------------------------
def _call_tool(name: str, args: Dict[str, Any], df: pd.DataFrame, user_q: str = ""):
    """
    Dynamically route user questions to the most relevant local tool.
    Supports flexible phrasing and interchangeable queries across cost, provider, fraud, and risk.
    """
    user_q_lower = user_q.lower() if isinstance(user_q, str) else ""

    # --- Cost drivers (ICD/CPT/period/plan/charge/expense) ---
    if any(word in user_q_lower for word in ["cost", "charge", "expense", "driver", "top icd", "top cpt", "revenue", "q1", "q2", "quarter"]):
        return top_icd_cpt_cost(df, **args)

    # --- Provider anomalies / comparisons ---
    if any(word in user_q_lower for word in ["provider", "anomaly", "z-score", "outlier", "billing pattern", "compare"]):
        return provider_anomalies(df, **args)

    # --- Fraud / excessive claims / abuse detection ---
    if any(word in user_q_lower for word in ["fraud", "flag", "excessive", "claims per patient", "abuse", "overbilling"]):
        return fraud_flags(df, **args)

    # --- Risk scoring / wait days / ICD or patient-level risk ---
    if any(word in user_q_lower for word in ["risk", "patient", "cohort", "wait", "delay", "icd", "severity"]):
        return risk_scoring(df, user_q=user_q, **args)


    return {"summary": f"⚠️ No matching tool found for query: {user_q}", "table": []}

# ------------------------------
# 4. Main entrypoint (final stable version)
# ------------------------------
def ask_gpt(user_q: str, df: pd.DataFrame, rag: SimpleRAG) -> Dict[str, Any]:
    key = st.secrets.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key)

    messages = _messages(user_q, rag)
    tools = _tools_schema()
    result_payload = {"summary": [], "tables": [], "figures": [], "citations": [], "next_steps": []}

    auto_tool_result = None
    auto_detected = False  # ✅ fixed typo

    try:
        # 1️⃣ Try local auto-detection first (keyword-based intent routing)
        try:
            auto_tool_result = _call_tool("", {}, df, user_q=user_q)
            if auto_tool_result and "⚠️ No matching" not in str(auto_tool_result.get("summary", "")):
                auto_detected = True
                st.info("🧠 Auto-detected tool execution based on query intent.")
        except Exception as e:
            st.warning(f"Auto-detect check failed: {e}")

        # 2️⃣ If not auto-detected, let GPT choose the tool
        if not auto_detected:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    fn = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    auto_tool_result = _call_tool(fn, args, df, user_q=user_q)
                    st.write(" tool_result preview:", auto_tool_result)
                    auto_detected = True
                    st.info(f"🧠 Tool invoked by GPT: {fn}")

        # 3️⃣ If a tool executed successfully (either auto or GPT)
        if auto_detected and auto_tool_result:
            tool_result = auto_tool_result

            # Normalize output
            if isinstance(tool_result, dict):
                if "summary" not in tool_result:
                    tool_result["summary"] = "Tool executed successfully."
                if "table" in tool_result and isinstance(tool_result["table"], pd.DataFrame):
                    tool_result["table"] = tool_result["table"].to_dict(orient="records")
            elif isinstance(tool_result, pd.DataFrame):
                tool_result = {"summary": "DataFrame result", "table": tool_result.to_dict(orient="records")}
            else:
                tool_result = {"summary": str(tool_result), "table": []}

            # Populate result payload
            result_payload["summary"] = [tool_result.get("summary", "")]
            result_payload["tables"] = tool_result.get("table", [])
            result_payload["citations"] = [tool_result.get("table_name", "tool_output")]
            result_payload["next_steps"] = [
                "Review the above data for trends and outliers.",
                "Validate ICD/CPT mappings or provider records.",
                "Explore follow-up analysis on period-over-period changes.",
            ]

        else:
            # GPT fallback if no tool invoked
            result_payload["summary"].append("No tools invoked — possible vague or unsupported query.")
            result_payload["next_steps"].append("Try rephrasing with keywords like cost, provider, risk, or fraud.")

        # 4️⃣ Clean + flatten tables for Streamlit rendering
        if "tables" in result_payload:
            clean_tables = []
            for t in result_payload["tables"]:
                if isinstance(t, str):
                    try:
                        parsed = json.loads(t.replace("'", '"'))
                        if isinstance(parsed, list):
                            clean_tables.extend(parsed)
                        elif isinstance(parsed, dict):
                            clean_tables.append(parsed)
                        else:
                            clean_tables.append({"Raw": str(parsed)})
                    except Exception:
                        clean_tables.append({"Raw": t})
                elif isinstance(t, list):
                    for row in t:
                        if isinstance(row, dict):
                            clean_tables.append(row)
                        else:
                            clean_tables.append({"Value": str(row)})
                elif isinstance(t, dict):
                    clean_tables.append(t)

            # Expand nested ICD/CPT structures if any
            final_rows = []
            for row in clean_tables:
                if isinstance(row, dict):
                    quarter = row.get("Quarter") or row.get("quarter") or ""
                    codes = row.get("Top ICD-10 Codes") or row.get("ICD10") or row.get("icd") or []
                    if isinstance(codes, list):
                        for sub in codes:
                            if isinstance(sub, dict):
                                final_rows.append({
                                    "Quarter": quarter,
                                    "ICD-10 Code": sub.get("ICD-10 Code") or sub.get("icd") or sub.get("Code"),
                                    "Total Cost": sub.get("Total Cost") or sub.get("Cost") or sub.get("Charge"),
                                    "Cost Share (%)": sub.get("Cost Share (%)") or sub.get("Share (%)"),
                                })
                            else:
                                final_rows.append({"Quarter": quarter, "ICD-10 Code": str(sub)})
                    else:
                        final_rows.append(row)

            result_payload["tables"] = pd.DataFrame(final_rows).replace({None: ""}).to_dict(orient="records")

        # 🧩 Clean up CPT/ICD column names for consistency
if "tables" in result_payload and result_payload["tables"]:
    clean_df = pd.DataFrame(result_payload["tables"])
    rename_map = {
        "cpt": "CPT Code",
        "CPT": "CPT Code",
        "icd10": "ICD-10 Code",
        "ICD10": "ICD-10 Code",
        "charge_amount": "Total Cost",
        "total_cost": "Total Cost",
    }
    clean_df.rename(columns={k: v for k, v in rename_map.items() if k in clean_df.columns}, inplace=True)
    result_payload["tables"] = clean_df.to_dict(orient="records")


        return result_payload

    except openai.RateLimitError:
        return {
            "summary": ["⚠️ Rate limit reached. Please wait a few seconds and try again."],
            "tables": [],
            "figures": [],
            "citations": [],
            "next_steps": []
        }
