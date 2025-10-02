
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
    """Experiment 2: Few-shot prompting to enforce JSON-style outputs."""

    key = st.secrets.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key)

    # Build messages with few-shots
    messages = [
        {"role": "system", "content": "You are Healthcare Claims Copilot. Answer questions using structured JSON output with keys: summary, tables, figures, citations, next_steps."}
    ]

    for ex in FEW_SHOTS:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})

    messages.append({"role": "user", "content": user_q})

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.2
        )

        answer = resp.choices[0].message.content or "{}"

        # Try parsing JSON-like content
        try:
            parsed = json.loads(answer)
        except Exception:
            parsed = {"summary": [answer]}

        # Return structured payload
        return {
            "summary": parsed.get("summary", []),
            "tables": parsed.get("tables", []),
            "figures": parsed.get("figures", []),
            "citations": parsed.get("citations", []),
            "next_steps": parsed.get("next_steps", [])
        }

    except openai.RateLimitError:
        return {
            "summary": ["⚠️ Rate limit reached. Please wait a few seconds and try again."],
            "tables": [],
            "figures": [],
            "citations": [],
            "next_steps": []
        }
