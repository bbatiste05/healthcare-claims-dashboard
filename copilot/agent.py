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

# --- add this helper ---
def _route_intent(user_q: str) -> str:
    q = (user_q or "").lower()
    # cost drivers / ICD / CPT
    if any(k in q for k in ["cpt", "cost driver", "top cost", "highest cost", "charge amount", "icd"]):
        return "top_icd_cpt_cost"
    # anomalies & comparisons
    if any(k in q for k in ["anomal", "outlier", "z-score", "z score", "compare", "q1 vs q2", "quarter"]):
        return "provider_anomalies"
    # fraud / frequency
    if any(k in q for k in ["fraud", "excessive claim", "claims per patient"]):
        return "fraud_flags"
    # risk
    if any(k in q for k in ["risk", "risk score"]):
        return "risk_scoring"
    return ""    

# ------------------------------
# 3. Call local Python tools
# ------------------------------
def _call_tool(name: str, args: Dict[str, Any], df: pd.DataFrame, user_q: str = ""):
    """Route by explicit tool name OR by question intent."""
    pick = name or _route_intent(user_q)
    if pick == "top_icd_cpt_cost":
        return top_icd_cpt_cost(df, **args)
    if pick == "provider_anomalies":
        # support 'compare' if user asks e.g. "Q1 vs Q2 2024"
        if "q1" in user_q.lower() and "q2" in user_q.lower():
            args = {**args, "compare": "2024Q1_vs_2024Q2"}
        return provider_anomalies(df, **args)
    if pick == "fraud_flags":
        return fraud_flags(df, **args)
    if pick == "risk_scoring":
        # if user includes “by icd”
        if "by icd" in user_q.lower() or "icd" in user_q.lower():
            args = {**args, "by_icd": True}
        return risk_scoring(df, **args)
    return {"summary": "No matching tool.", "table_name": "none", "table": []}


# ------------------------------
# 4. Main entrypoint (stable rollback version)
# ------------------------------
def ask_gpt(user_q: str, df: pd.DataFrame, rag: SimpleRAG) -> Dict[str, Any]:
    key = st.secrets.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key)

    messages = _messages(user_q, rag)
    tools = _tools_schema()

    result_payload = {"summary": [], "tables": [], "figures": [], "citations": [], "next_steps": []}

    # --- 0) One-pass local autoroute (cheap & reliable) ---
    try:
        auto = _call_tool("", {}, df, user_q=user_q)
        rows = auto.get("table", [])
        if isinstance(rows, list) and len(rows) > 0:
            # we already have a good, structured answer → format and return
            result_payload["summary"] = [auto.get("summary", "")]
            # normalize table: list[dict]
            result_payload["tables"] = rows if isinstance(rows, list) else [rows]
            if auto.get("table_name"):
                table_rows = rows if isinstance(rows, list) else [rows]
                # prepend a name row to keep your UI compatible
               result_payload["tables"] = table_rows
            else:
                result_payload["tables"] = rows if isinstance(rows, list) else [rows]
            result_payload["citations"] = auto.get("citations", [])
            result_payload["next_steps"] = auto.get("next_steps", [])

              # ✅ Ensure fallback citations exist even if autoroute returns early
            if not result_payload.get("citations") or len(result_payload["citations"]) == 0:
                if "provider" in user_q.lower():
                    result_payload["citations"] = ["provider_anomalies.csv"]
                elif "risk" in user_q.lower():
                    result_payload["citations"] = ["patient_risk_scores.csv"]
                elif "fraud" in user_q.lower():
                    result_payload["citations"] = ["fraud_flags.csv"]
                elif any(k in user_q.lower() for k in ["cpt", "icd", "cost", "charge"]):
                    result_payload["citations"] = ["claims_df.csv", "icd10_reference.csv"]
                else:
                    result_payload["citations"] = ["claims_df.csv"]            
            return result_payload
    except Exception:
        pass

    # --- (the rest of your existing ask_gpt logic can remain as-is) ---
    # If you keep your current /chat.completions follow-up section, it will run
    # when the LLM decides to call tools. The autoroute above simply gives you
    # a fast, robust answer when the question is straightforward.
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

                # ✅ Ensure fallback citations exist before returning from GPT tool_call section
                if not result_payload.get("citations") or len(result_payload["citations"]) == 0:
                    if "provider" in user_q.lower():
                        result_payload["citations"] = ["provider_anomalies.csv"]
                    elif "risk" in user_q.lower():
                        result_payload["citations"] = ["patient_risk_scores.csv"]
                    elif "fraud" in user_q.lower():
                        result_payload["citations"] = ["fraud_flags.csv"]
                    elif any(k in user_q.lower() for k in ["cpt", "icd", "cost", "charge"]):
                        result_payload["citations"] = ["claims_df.csv", "icd10_reference.csv"]
                    else:
                        result_payload["citations"] = ["claims_df.csv"]
                return result_payload


        # 2. If GPT requested a tool → run locally
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn = tc.function.name
                args = json.loads(tc.function.arguments or "{}")

                # ✅ Run the tool locally
                tool_result = _call_tool(fn, args, df, user_q=user_q)
                
               # 🩹 Ensure a visible table even if tool didn't return one
                try:
                    if isinstance(tool_result, dict):
                        table_data = tool_result.get("table", None)

                        # Normalize DataFrame → dict
                        if isinstance(table_data, pd.DataFrame):
                            tool_result["table"] = table_data.to_dict(orient="records")

                        # If no valid table present, scan for potential table-like data
                        elif not isinstance(table_data, list):
                            detected = None
                            for key, val in tool_result.items():
                                if isinstance(val, pd.DataFrame):
                                    detected = val.to_dict(orient="records")
                                    break
                                elif isinstance(val, list) and val and isinstance(val[0], dict):
                                    detected = val
                                    break
                                elif isinstance(val, dict):
                                    detected = [val]
                                    break

                            # Fallback: no tabular data detected
                            tool_result["table"] = detected or [{"message": "No table data returned by this function."}]

                    elif isinstance(tool_result, pd.DataFrame):
                        tool_result = {"summary": "DataFrame result", "table": tool_result.to_dict(orient="records")}
                    else:
                        tool_result = {"summary": str(tool_result), "table": [{"message": "Unrecognized tool output."}]}

                except Exception as e:
                    st.warning(f"⚠️ Table normalization failed: {e}")
                    tool_result = {
                        "summary": f"Tool ran, but table normalization failed: {e}",
                        "table": [{"error": str(e)}],
                    }

                # If still empty, show a diagnostic message
                if not tool_result.get("table"):
                    tool_result["table"] = [{"message": "No table data returned by this function."}]

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

#                ✅ Send back tool results to GPT for formatting
                try:
                    safe_tool_content = json.dumps(tool_result, default=str, indent=2)
                except Exception as e:
                    safe_tool_content = json.dumps({"error": f"Serialization failed: {str(e)}"}, indent=2)

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
                        val = parsed.get(k)
                        if isinstance(val, str):
                            result_payload[k] = [val]
                        elif val is not None:
                            result_payload[k] = val
                      
                    if not result_payload.get("next_steps"):   
                        result_payload["next_steps"] = ["Review top cost driversand invesigate outliers by provider or diagnosis."]

                    # ✅ Flatten table structure if nested under 'table' key
                    if "tables" in result_payload:
                        flattened = []
                        
                        for entry in result_payload["tables"]:
                            if isinstance(entry, dict) and "table" in entry:
                                table_data = entry["table"]
                                if isinstance(table_data, list):
                                    flattened.extend(table_data)
                                elif isinstance(table_data, dict):
                                    flattened.append(table_data)
                                else:
                                    flattened.append({"Value": str(table_data)})
                            elif isinstance(entry, list):
                                flattened.extend(entry)
                            elif isinstance(entry, dict):
                                flattened.append(entry)
                            else:
                                flattened.append({"Value": str(entry)})

                        # Drop blank rows and normalize types
                        df_flat = pd.DataFrame(flattened)
                        if not df_flat.empty:
                            df_flat = df_flat.fillna("")
                        result_payload["tables"] = df_flat.to_dict(orient="records")
                except Exception as e:
                    st.warning(f" error parsing GPT response: {e}")
                    result_payload["summary"].append(final_answer)    

        # ✅ Ensure tables are preserved before fallback or overwrite
        if "tables" not in result_payload or not result_payload["tables"]:
            result_payload["tables"] = df_final.to_dict(orient="records")

        # ✅ If DataFrame accidentally serialized as string, fix it
        elif isinstance(result_payload["tables"], str):
            try:
                parsed_table = json.loads(result_payload["tables"].replace("'", '"'))
                if isinstance(parsed_table, list):
                    result_payload["tables"] = parsed_table
            except Exception:
                pass

        # ✅ Force reattach any parsed table results before returning
        if "table" in locals() and isinstance(tool_result, dict) and tool_result.get("table"):
            if not result_payload.get("tables"):
                result_payload["tables"] = tool_result["table"]
            elif isinstance(result_payload["tables"], list) and isinstance(tool_result["table"], list):
                result_payload["tables"].extend(tool_result["table"])

        # ✅ Ensure fallback citations exist even if autoroute returns early
        if not result_payload.get("citations") or len(result_payload["citations"]) == 0:
            if "provider" in user_q.lower():
                result_payload["citations"] = ["provider_anomalies.csv"]
            elif "risk" in user_q.lower():
                result_payload["citations"] = ["patient_risk_scores.csv"]
            elif "fraud" in user_q.lower():
                result_payload["citations"] = ["fraud_flags.csv"]
            elif any(k in user_q.lower() for k in ["cpt", "icd", "cost", "charge"]):
                result_payload["citations"] = ["claims_df.csv", "icd10_reference.csv"]
            else:
                result_payload["citations"] = ["claims_df.csv"]

        # 🧠 If GPT didn’t call any tool but query clearly matches a known intent → run manually
        if not msg.tool_calls and not auto_detected:
            user_q_lower = user_q.lower()
            st.info("⚡ No GPT tool call detected — attempting auto-fallback match.")
            if any(k in user_q_lower for k in ["cost", "charge", "cpt", "icd"]):
                tool_result = _call_tool("top_icd_cpt_cost", {}, df, user_q=user_q)
                auto_detected = True
            elif any(k in user_q_lower for k in ["provider", "outlier", "anomaly", "quarter"]):
                tool_result = _call_tool("provider_anomalies", {}, df, user_q=user_q)
                auto_detected = True
            elif any(k in user_q_lower for k in ["fraud", "claims per patient", "flag"]):
                tool_result = _call_tool("fraud_flags", {}, df, user_q=user_q)
                auto_detected = True
            elif any(k in user_q_lower for k in ["risk", "cohort", "patient risk"]):
                tool_result = _call_tool("risk_scoring", {}, df, user_q=user_q)
                auto_detected = True

            if auto_detected:
                st.success(f"✅ Auto-ran tool based on query intent: {list(tool_result.keys())}")

                if isinstance(tool_result, pd.DataFrame):
                    tool_result = {
                        "summary": "Auto-detected DataFrame result",
                        "table": tool_result.to_dict(orient="records")
                    }

                result_payload["summary"].append(tool_result.get("summary", "Tool executed automatically."))
                result_payload["tables"] = tool_result.get("table", [])
                return result_payload
                                                                          

        # 3. Fallback if no tools invoked
        if not result_payload.get("summary"):
            result_payload["summary"] = [msg.content or "No tools invoked."]
        if not result_payload.get("tables"):
            result_payload["tables"] = []

        # ✅ Remove any placeholder 'Table' column or empty header row
        try:
            df_temp = pd.DataFrame(result_payload["tables"])
            if "Table" in df_temp.columns:
                df_temp.drop(columns=["Table"], inplace=True)
            df_temp = df_temp.fillna("")
            result_payload["tables"] = df_temp.to_dict(orient="records")
        except Exception as e:
            st.warning(f"⚠️ Could not clean table column: {e}")

        # ✅ Enrich short summaries into analytical insights
        if result_payload.get("summary"):
            enriched_summary = []
            for s in result_payload["summary"]:
                if len(s) < 120:
                    if "cpt" in s.lower() or "cost" in s.lower():
                        enriched_summary.append(
                            s + " This indicates that a small group of procedure codes are responsible for a large share of total expenses, "
                            "suggesting a need to review utilization patterns, pricing strategies, or potential overuse."
                        )
                    elif "provider" in s.lower():
                        enriched_summary.append(
                            s + " The identified providers show significant deviation from peer averages, which may indicate specialty concentration "
                            "or possible billing inconsistencies that warrant review."
                        )
                    elif "risk" in s.lower():
                        enriched_summary.append(
                            s + " Elevated risk levels may correlate with complex patient conditions or chronic comorbidities, "
                            "highlighting candidates for care coordination or intervention."
                        )
                    elif "fraud" in s.lower():
                        enriched_summary.append(
                            s + " These patterns could suggest atypical billing behavior or repeated claim anomalies worth investigating."
                        )
                    else:
                        enriched_summary.append(s)
                else:
                    enriched_summary.append(s)
            result_payload["summary"] = enriched_summary

          

        # ✅ Ensure next_steps and citations always exist
        if not result_payload.get("next_steps") or len(result_payload["next_steps"]) == 0:
            if "provider" in user_q.lower():
                result_payload["next_steps"] = [
                    "Review providers with abnormal Z-scores for potential billing errors.",
                    "Check whether flagged providers have overlapping claims across multiple CPTs.",
                ]
            elif "risk" in user_q.lower():
                result_payload["next_steps"] = [
                    "Review patient cohorts with highest computed risk scores.",
                    "Validate input features such as charge_amount and wait_days.",
                    "Consider adding comorbidity weights for refined risk estimation.",
                ]
            elif "fraud" in user_q.lower():
                result_payload["next_steps"] = [
                    "Investigate provider clusters with excessive claim counts.",
                    "Run peer comparison by specialty or region.",
                ]
            else:
                result_payload["next_steps"] = [
                    "Investigate high-cost or high-risk areas identified.",
                    "Validate data completeness for quarterly comparisons.",
                    "Drill into provider or CPT breakdowns for insight.",
                ]

        if not result_payload.get("citations") or len(result_payload["citations"]) == 0:
            if "provider" in user_q.lower():
                result_payload["citations"] = ["provider_anomalies.csv"]
            elif "risk" in user_q.lower():
                result_payload["citations"] = ["patient_risk_scores.csv"]
            elif "fraud" in user_q.lower():
                result_payload["citations"] = ["fraud_flags.csv"]
            elif "cpt" in user_q.lower():
                result_payload["citations"] = ["claims_df.csv", "cpt.csv"]
            elif "icd" in user_q.lower():
                result_payload["citations"] = ["claims_df.csv", "icd10.csv"]
            else:
                result_payload["citations"] = ["claims_df.csv"]

        return result_payload

    except openai.RateLimitError:
        return {
            "summary": ["⚠️ Rate limit reached. Please wait a few seconds and try again."],
            "tables": [],
            "figures": [],
            "citations": [],
            "next_steps": []
        }
