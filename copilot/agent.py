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
                        "top_n": {"type": "integer", "default": 10},
                    },
                },
            },
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
                        "threshold": {"type": "number", "default": 3.0},
                    },
                },
            },
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
                        "window_days": {"type": "integer", "default": 90},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "risk_scoring",
                "description": "Compute de-identified risk scores for patients.",
                "parameters": {
                    "type": "object",
                    "properties": {"cohort": {"type": "string", "nullable": True}},
                },
            },
        },
    ]


# ------------------------------
# 2. Build messages
# ------------------------------
def _messages(user_q: str, rag: SimpleRAG) -> list:
    snippets = rag.search(user_q, k=5)
    snip_text = "\n".join([json.dumps(s, ensure_ascii=False) for s in snippets])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"External context (ICD/CPT/NPPES snippets):\n{snip_text}"},
    ]

    for ex in FEW_SHOTS:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})

    messages.append({"role": "user", "content": user_q})
    return messages


# ------------------------------
# 3. Call local Python tools
# ------------------------------
def _call_tool(name: str, args: Dict[str, Any], df: pd.DataFrame, user_q: str = ""):
    user_q_lower = user_q.lower() if isinstance(user_q, str) else ""

    if name == "top_icd_cpt_cost" or ("cost" in user_q_lower or "charge" in user_q_lower):
        return top_icd_cpt_cost(df, **args)
    elif name == "provider_anomalies" or ("provider" in user_q_lower or "quarter" in user_q_lower):
        return provider_anomalies(df, **args)
    elif name == "fraud_flags" or ("fraud" in user_q_lower or "claims per patient" in user_q_lower):
        return fraud_flags(df, **args)
    elif name == "risk_scoring" or ("risk" in user_q_lower or "cohort" in user_q_lower):
        return risk_scoring(df, **args)

    return {"summary": "No matching tool found for this query.", "table": []}


# ------------------------------
# 4. Main entrypoint (stable rollback version)
# ------------------------------
def ask_gpt(user_q: str, df: pd.DataFrame, rag: SimpleRAG) -> Dict[str, Any]:
    key = st.secrets.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key)

    messages = _messages(user_q, rag)
    tools = _tools_schema()

    result_payload = {"summary": [], "tables": [], "figures": [], "citations": [], "next_steps": []}

    try:
        # 1. Ask GPT (tool choice auto-enabled)
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2
        )

        msg = resp.choices[0].message

        auto_tool_result = None
        auto_detected = False

        # 🧠 Auto-detect likely tool before explicit tool calls (fallback safety)
        try:
            auto_tool_result = _call_tool("", {}, df, user_q=user_q)
            if isinstance(auto_tool_result, dict):
                summary_text = str(auto_tool_result.get("summary", "")).lower()
                if "no matching" not in summary_text:
                    auto_detected = True
            elif isinstance(auto_tool_result, pd.DataFrame) and not auto_tool_result.empty:
                auto_detected = True
                
        except Exception as e:
            st.warning(f"Auto-detect fallback failed: {e}")

        st.info(f"Auto-detect: using {_call_tool.__name__ if auto_detected else 'GPT tool selection'}")

        # ✅ Handle auto-detected tool result (direct run, bypassing GPT tool_call)
        if auto_detected and auto_tool_result:
            st.info("Auto-detected tool result used directly.")

            # Normalize tool result into standard format
            if isinstance(auto_tool_result, pd.DataFrame):
                result_payload["tables"] = auto_tool_result.to_dict(orient="records")
                result_payload["summary"].append("Tool executed successfully (auto-detected).")
                return result_payload

            elif isinstance(auto_tool_result, dict):
                result_payload["summary"].append(auto_tool_result.get("summary", "Tool executed successfully."))
                if "table" in auto_tool_result:
                    if isinstance(auto_tool_result["table"], pd.DataFrame):
                        result_payload["tables"] = auto_tool_result["table"].to_dict(orient="records")
                    elif isinstance(auto_tool_result["table"], list):
                        result_payload["tables"] = auto_tool_result["table"]
                return result_payload


        # 2. If GPT requested a tool → run locally
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn = tc.function.name
                args = json.loads(tc.function.arguments or "{}")

                # ✅ Run the tool locally
                tool_result = _call_tool(fn, args, df, user_q=user_q)

                # ✅ Normalize tool_result into valid JSON for GPT
                if isinstance(tool_result, dict):
                    if "summary" not in tool_result:
                        tool_result["summary"] = "Tool executed successfully."
                    if "table" in tool_result and isinstance(tool_result["table"], pd.DataFrame):
                        tool_result["table"] = tool_result["table"].to_dict(orient="records")
                elif isinstance(tool_result, pd.DataFrame):
                    tool_result = {"summary": "DataFrame result", "table": tool_result.to_dict(orient="records")}
                else:
                    tool_result = {"summary": str(tool_result), "table": []}

                # ✅ Send back tool results to GPT for formatting
                safe_tool_content = json.dumps(tool_result, default=str, indent=2)
                tool_id = getattr(tc, "id", "tool_1")

                follow_messages = [
                    *messages,
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": str(tool_id), "type": "function",
                             "function": {"name": fn, "arguments": json.dumps(args)}}
                        ],
                    },
                    {"role": "tool", "content": safe_tool_content, "tool_call_id": str(tool_id)},
                    {
                        "role": "user",
                        "content": (
                            "Format the final answer as valid JSON with keys: summary, tables, figures, citations, next_steps. "
                            "Maintain tone and structure of the Healthcare Claims CoPilot (use clinical reasoning and clear numeric summaries). "
                            "Include 1-2 key insights (e.g., which ICD/CPTcodes dominate cost, anomaly trends, or risk patterns) and quantify results. "
                            "Include 2-3 recommend next_steps that guide further analysis or compliance action. "
                            "Always include citations referencing data sources used (e.g., 'claims_df', 'icd.csv', 'nppes.csv'). "
                            "Ensure JSON syntax is correct, concise, and under 2000 tokens."
                        ),
                    },
                ]

                follow = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=follow_messages,
                    temperature=0.2,
                )

                # ✅ Parse GPT’s formatted JSON output
                final_answer = follow.choices[0].message.content
                try:
                    parsed = json.loads(final_answer)
                    for k in result_payload.keys():
                        if k in parsed:
                            if isinstance(parsed[k], str):
                                result_payload[k] = [parsed[k]]
                            elif isinstance(parsed[k], list):
                                result_payload[k].extend(parsed[k])
                            else:
                                result_payload[k] = parsed[k]
                except Exception:
                    result_payload["summary"].append(final_answer)

            # ✅ Flatten nested tables (the version that worked yesterday)
            if "tables" in result_payload:
                clean_tables = []
                for t in result_payload["tables"]:
                    if isinstance(t, str):
                        try:
                            parsed = json.loads(t.replace("'", '"'))
                            clean_tables.extend(parsed if isinstance(parsed, list) else [parsed])
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

                final_rows = []
                for row in clean_tables:
                    if isinstance(row, dict):
                        quarter = row.get("Quarter") or row.get("quarter") or ""
                        codes = row.get("Top ICD-10 Codes") or row.get("ICD10") or row.get("icd") or []
                        if isinstance(codes, list):
                            for code_entry in codes:
                                if isinstance(code_entry, dict):
                                    final_rows.append({
                                        "Quarter": quarter,
                                        "ICD-10 Code": code_entry.get("ICD-10 Code") or code_entry.get("icd") or code_entry.get("Code"),
                                        "Total Cost": code_entry.get("Total Cost") or code_entry.get("Cost") or code_entry.get("Charge"),
                                        "Cost Share (%)": code_entry.get("Cost Share (%)") or code_entry.get("Share (%)"),
                                    })
                        else:
                            final_rows.append({
                                "Quarter": quarter,
                                "ICD-10 Code": row.get("ICD-10 Code"),
                                "Total Cost": row.get("Total Cost"),
                                "Cost Share (%)": row.get("Cost Share (%)"),
                            })

            df_final = pd.DataFrame(final_rows).dropna(how="all").fillna("")
            result_payload["tables"] = df_final.to_dict(orient="records")

        # 3. Fallback if no tools invoked
        result_payload["summary"].append(msg.content or "No tools invoked.")

          # ✅ Ensure next_steps and citations are always present
        if not result_payload.get("next_steps") or not result_payload["next_steps"]:
            result_payload["next_steps"] = [
                "Review high-cost codes to identify potential cost concentration.",
                "Cross-reference top CPTs with provider utilization data.",
                "Validate claim data completeness before deeper analysis.",
            ]

        if not result_payload.get("citations") or not result_payload["citations"]:
            result_payload["citations"] = ["claims_df", "icd10_reference.csv"]        
        
        return result_payload

    except openai.RateLimitError:
        return {
            "summary": ["⚠️ Rate limit reached. Please wait a few seconds and try again."],
            "tables": [],
            "figures": [],
            "citations": [],
            "next_steps": []
        }
