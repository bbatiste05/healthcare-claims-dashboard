# === FEW-SHOT EXAMPLES ===
FEW_SHOTS = [
    {
        "user": "Which CPT codes drove the most cost in Q2 2024?",
        "assistant": """{
            "summary": ["Top CPT codes for Q2 2024 were 99213 and 93000, accounting for 45% of costs."],
            "tables": [[
                {"CPT Code": "99213", "Total Cost": 30000, "Cost Share (%)": 30.0},
                {"CPT Code": "93000", "Total Cost": 15000, "Cost Share (%)": 15.0}
            ]],
            "figures": [],
            "citations": ["cpt.csv"],
            "next_steps": ["Review justification for CPT 99213", "Audit providers with high utilization"]
        }"""
    },
    {
        "user": "Which ICD codes account for the highest total charge amounts?",
        "assistant": """{
            "summary": ["The top ICD-10 codes driving cost are L5789, 41401, and 0389, representing 45% of total charges."],
            "tables": [[
                {"ICD-10 Code": "L5789", "Total Cost": 18449000, "Cost Share (%)": 17.18},
                {"ICD-10 Code": "41401", "Total Cost": 16713880, "Cost Share (%)": 15.57},
                {"ICD-10 Code": "0389", "Total Cost": 14227040, "Cost Share (%)": 13.25}
            ]],
            "figures": [],
            "citations": ["icd.csv"],
            "next_steps": ["Investigate cost drivers for top ICD-10 codes to identify trends."]
        }"""
    }
]

# === ANNOTATION ===
# Each few-shot does 3 things:
# 1. Shows GPT the JSON schema you expect (summary, tables, figures, citations, next_steps).
# 2. Demonstrates tool use: first example → CPT cost drivers, second → ICD cost drivers.
# 3. Provides realistic next steps an analyst would take, improving actionability.
