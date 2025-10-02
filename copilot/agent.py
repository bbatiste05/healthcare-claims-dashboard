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
def _call_tool(name: str, args: Dict[str, Any], df: pd.DataFrame):
    if name == "top_icd_cpt_cost":
        return top_icd_cpt_cost(df, **args)
    if name == "provider_anomalies":
        return provider_anomalies(df, **args)
    if name == "fraud_flags":
        return fraud_flags(df, **args)
    if name == "risk_scoring":
        return risk_scoring(df, **args)
    return {"error": f"Unknown tool: {name}"}


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
                tool_result = _call_tool(fn, args, df)

                # 🔧 Patch: Normalize DataFrame → list[dict]
                if isinstance(tool_result, pd.DataFrame):
                    tool_result = tool_result.to_dict(orient="records")
                elif isinstance(tool_result, dict):
                    # If it's a dict with rows inside, normalize too
                    if any(isinstance(v, (list, dict)) for v in tool_result.values()):
                        try:
                            tool_result = pd.DataFrame(tool_result).to_dict(orient="records")
                        except Exception:
                            tool_result = [tool_result]
                    else:
                        tool_result = [tool_result]


                # 3. Feed tool results back to GPT
                follow = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        *messages,
                        {"role": "assistant", "content": None, "tool_calls": msg.tool_calls},
                        {"role": "tool", "content": json.dumps(tool_result, default=str), "tool_call_id": tc.id},
                        {"role": "user", "content": "Format the final answer as JSON with keys: summary, tables, figures, citations, next_steps."}
                    ],
                    temperature=0.2
                )

                final_answer = follow.choices[0].message.content
                try:
                    parsed = json.loads(final_answer)

                    # Normalize each key
                    for k in result_payload.keys():
                        if isinstance(parsed.get(k), str):
                            result_payload[k] = [parsed.get(k)]
                        else:
                            result_payload[k] = parsed.get(k, result_payload[k])

                    # 🔧 Extra patch: normalize tables so they’re always list[dict]
                    if "tables" in result_payload:
                        fixed_tables = []
                        for t in result_payload["tables"]:
                            if isinstance(t, dict):
                                fixed_tables.append(t)
                            elif isinstance(t, list):
                                fixed_tables.extend(t)
                        result_payload["tables"] = fixed_tables

                    return result_payload


        # 4. Fallback if no tools invoked
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

