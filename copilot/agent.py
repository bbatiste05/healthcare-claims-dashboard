
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
                        "period": {"type": "string", "description": "e.g., '2024Q2' or '2024-01:2024-06'", "nullable": True},
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


def _messages(user_q: str, rag: SimpleRAG) -> list:
    snippets = rag.search(user_q, k=5)
    snip_text = "\n".join([json.dumps(s, ensure_ascii=False) for s in snippets])

    msgs = [
        {"role": "system", "content": [{"type": "output_text", "text": SYSTEM_PROMPT}]}
    ]

    for ex in FEW_SHOTS:
        msgs.append({
            "role": "user",
            "content": [{"type": "input_text", "text": ex["user"]}]
        })
        msgs.append({
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Use tools as needed. Return structured JSON."}]
        })

    msgs.append({
        "role": "system",
        "content": [{"type": "output_text", "text": f"External context (ICD/CPT/NPPES snippets):\n{snip_text}"}]
    })

    msgs.append({
        "role": "user",
        "content": [{"type": "input_text", "text": user_q}]
    })

    return msgs



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


def ask_gpt(user_q: str, df: pd.DataFrame, rag: SimpleRAG) -> Dict[str, Any]:
    """Experiment 4: Tool + RAG + structured JSON."""

    key = st.secrets.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key)

    # 1. Retrieval augmentation (snippets from ICD/CPT/NPPES)
    snippets = rag.search(user_q, k=5)
    snip_text = "\n".join([json.dumps(s, ensure_ascii=False) for s in snippets])

    # 2. Messages: system + few-shot + context + user
    messages = [
        {
            "role": "system",
            "content": (
                "You are Healthcare Claims Copilot. "
                "Always return valid JSON with keys: summary, tables, figures, citations, next_steps. "
                "Use tools when possible to compute real results. "
                "Ground external references using the provided context snippets."
            )
        },
        {"role": "system", "content": f"External context (ICD/CPT/NPPES snippets):\n{snip_text}"}
    ]

    for ex in FEW_SHOTS:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})

    messages.append({"role": "user", "content": user_q})

    try:
        # 3. Call GPT with tool schemas
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=_tools_schema(),   # <-- from your agent.py
            tool_choice="auto",
            temperature=0.2
        )

        result_payload = {"summary": [], "tables": [], "figures": [], "citations": [], "next_steps": []}

        # 4. Check if a tool was invoked
        for choice in resp.choices:
            msg = choice.message
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    fn = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    tool_result = _call_tool(fn, args, df)  # <-- runs Python function

                    # 5. Feed tool result back into GPT for final formatting
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

                    # Parse final JSON
                    final_answer = follow.choices[0].message.content
                    try:
                        parsed = json.loads(final_answer)
                        for k in result_payload.keys():
                            if isinstance(parsed.get(k), str):
                                result_payload[k] = [parsed.get(k)]
                            else:
                                result_payload[k] = parsed.get(k, result_payload[k])

        # Fallback if no tools used
        result_payload["summary"].append("No tools invoked. Try specifying CPT/ICD/time window.")
        return result_payload

    except openai.RateLimitError:
        return {
            "summary": ["⚠️ Rate limit reached. Please wait a few seconds and try again."],
            "tables": [],
            "figures": [],
            "citations": [],
            "next_steps": []
        }
