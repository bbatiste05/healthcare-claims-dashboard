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
                "description": "Compute de-identified risk scores for patients.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cohort": {"type": "string", "nullable": True}
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
    user_q_lower = user_q.lower() if isinstance(user_q, str) else ""

  # 🧠 Cost driver detection
    if (
        name == "top_icd_cpt_cost"
        or "cost" in user_q_lower
        or "charge" in user_q_lower
        or "driver" in user_q_lower
    ):
        return top_icd_cpt_cost(df, **args)

    # 🧠 Provider anomaly detection
    if (
        name == "provider_anomalies"
        or "provider" in user_q_lower
        or "z-score" in user_q_lower
        or "quarter" in user_q_lower
        or "compare" in user_q_lower
    ):
        return provider_anomalies(df, **args)

    # 🧠 Fraud detection
    if (
        name == "fraud_flags"
        or "fraud" in user_q_lower
        or "claims per patient" in user_q_lower
        or "overbilling" in user_q_lower
    ):
        return fraud_flags(df, **args)

    # 🧠 Risk scoring or wait-time analysis
    if (
        name == "risk_scoring"
        or "risk" in user_q_lower
        or "cohort" in user_q_lower
        or "wait" in user_q_lower
        or "delay" in user_q_lower
        or "time to" in user_q_lower
        or "patient" in user_q_lower
    ):
        return risk_scoring(df, user_q=user_q, **args)

    return {"summary": f"⚠️ Unknown tool: {name}", "table": []}


# ------------------------------
# 4. Main entrypoint
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

                # ✅ Safe version for /chat/completions endpoint
                import traceback
                try:
                    safe_tool_content = json.dumps(tool_result, default=str, indent=2)
                except Exception as e:
                    safe_tool_content = json.dumps({"error": f"Serialization failed: {str(e)}"}, indent=2)

                tool_id = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_id = getattr(msg.tool_calls[0], "id", "tool_1")
                else:
                    tool_id = getattr(tc, "id", "tool_1")

                def _clean_msg(m):
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if not isinstance(content, str):
                        try:
                            content = json.dumps(content, default=str)
                        except Exception:
                            content = str(content)
                    return {"role": role, "content": content[:4000]}

                clean_messages = [_clean_msg(m) for m in messages]

                follow_messages = [
                    *clean_messages,
                    {"role": "assistant", "content": None, "tool_calls": [
                         {
                            "id": str(tool_id),
                            "type": "function",
                            "function": {"name": fn, "arguments": json.dumps(args)},
                         }
                       ] 
                    }, 
                    {"role": "tool", "content": safe_tool_content, "tool_call_id": str(tool_id)},
                    {   
                        "role": "user",
                        "content": (
                            "Format the final answer as valid JSON with keys: "
                            "summary, tables, figures, citations, next_steps. "
                            "Ensure JSON syntax is correct, concise, and under 2000 tokens."
                        ),
                    }
                ]

                try:
                    st.write("🧩 follow_messages length:", len(follow_messages))
                    st.write("✅ All content types stringified:", all(isinstance(m['content'], str) for m in follow_messages))
                except Exception:
                    pass

                try:
                    follow = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=follow_messages,
                        temperature=0.2,
                    )
                except Exception as e:
                    st.error(f"❌ GPT follow-up failed: {e}")
                    st.write(traceback.format_exc())
                    return {
                        "summary": [f"Error during GPT follow-up: {str(e)}"],
                        "tables": [],
                        "figures": [],
                        "citations": [],
                        "next_steps": []
                    }

                # ✅ Parse GPT’s formatted JSON output
                final_answer = follow.choices[0].message.content
                try:
                    parsed = json.loads(final_answer)
                    for k in result_payload.keys():
                        if isinstance(parsed.get(k), str):
                            result_payload[k] = [parsed.get(k)]
                        else:
                            result_payload[k] = parsed.get(k, result_payload[k])

                    # Normalize table structure
                    if "tables" in result_payload:
                        fixed_tables = []
                        for t in result_payload["tables"]:
                            if isinstance(t, dict):
                                fixed_tables.append(t)
                            elif isinstance(t, list):
                                fixed_tables.extend(t)
                        result_payload["tables"] = fixed_tables

                except Exception:
                    result_payload["summary"].append(final_answer)

                 # ✅ Normalize nested tables into a flat, readable structure
            if "tables" in result_payload:
                clean_tables = []

                for t in result_payload["tables"]:
                    # Case 1: If table is a stringified list of dicts
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

                    # Case 2: Already a list of dicts
                    elif isinstance(t, list):
                        for row in t:
                            if isinstance(row, dict):
                                clean_tables.append(row)
                            else:
                                clean_tables.append({"Value": str(row)})

                    # Case 3: Single dict
                    elif isinstance(t, dict):
                        clean_tables.append(t)

                # Optional: if quarters detected inside list, flatten that too
                # e.g. [{"Quarter": "Q1 2024", "Top ICD-10 Codes": [...]}, ...]
                final_rows = []
                for row in clean_tables:
                    if isinstance(row, dict):
                        quarter = row.get("Quarter") or row.get("quarter") or ""
                        codes = row.get("Top ICD-10 Codes") or row.get("ICD10") or row.get("icd") or []

                        # Case: nested list of ICD dictionaries
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
                                        "ICD-10 Code": str(code_entry),
                                        "Total Cost": None,
                                        "Cost Share (%)": None,
                                    })
                        else:
                            # Single ICD entry already flattened
                            final_rows.append({
                                "Quarter": quarter,
                                "ICD-10 Code": row.get("ICD-10 Code"),
                                "Total Cost": row.get("Total Cost"),
                                "Cost Share (%)": row.get("Cost Share (%)"),
                            })

            # Clean out empty rows and replace None with blanks
            df_final = pd.DataFrame(final_rows)
            df_final = df_final.dropna(how="all")
            df_final = df_final.fillna("")

                            

            result_payload["tables"] = df_final.to_dict(orient="records")


        # 3. Fallback if no tools invoked
        result_payload["summary"].append(msg.content or "No tools invoked.")
        return result_payload

    except openai.RateLimitError:
        return {
            "summary": ["⚠️ Rate limit reached. Please wait a few seconds and try again."],
            "tables": [],
            "figures": [],
            "citations": [],
            "next_steps": []
        }
