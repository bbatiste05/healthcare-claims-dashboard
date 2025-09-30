
# GPT Wiring Pack

## Files
- copilot/agent_gpt.py → OpenAI function-calling agent that calls your tools.
- data/external/icd10.csv, cpt.csv, nppes.csv → tiny subsets for RAG (replace later with official sources).

## How to use
1) pip install openai>=1.40.0
2) Set env vars:
   - OPENAI_API_KEY=...
   - OPENAI_MODEL=gpt-4.1-mini  (or another Responses-compatible model)
3) In dashboards/copilot_ui.py, switch the import:
   from copilot.agent import handle_query
   → from copilot.agent_gpt import ask_gpt as handle_query
4) Keep your existing tools in copilot/tools.py; the GPT will call them.
5) Ensure external CSVs live at data/external/ so rag.SimpleRAG can load them.
