# copilot/tools.py
import pandas as pd
import numpy as np

def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def top_icd_cpt_cost(
    df: pd.DataFrame,
    icd: str = None,
    cpt: str = None,
    period: str = None,
    plan: str = None,
    top_n: int = 10,
    **kwargs
):
    """
    Compute top ICD-10 or CPT codes by total cost, filtered by optional parameters.
    Automatically detects ICD vs CPT mode if called with guardrail.
    """

    # --- 1️⃣ Validate and prepare
    if df.empty:
        return {"error": "No data available."}

    # Normalize columns
    df = df.rename(columns={c.lower(): c for c in df.columns})

    # Determine which column to use
    icd_col = next((c for c in df.columns if "icd" in c.lower()), None)
    cpt_col = next((c for c in df.columns if "cpt" in c.lower()), None)
    cost_col = next((c for c in df.columns if "charge" in c.lower() or "cost" in c.lower()), None)
    date_col = next((c for c in df.columns if "date" in c.lower()), None)

    if not cost_col:
        return {"error": "No cost or charge column found."}

    # --- 2️⃣ Apply time filter (if period like '2024Q2')
    if period and date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            if "Q" in period:
                year, q = period[:4], int(period[-1])
                q_start = pd.Timestamp(f"{year}-{3 * (q - 1) + 1:02d}-01")
                q_end = (q_start + pd.offsets.QuarterEnd())
                df = df[(df[date_col] >= q_start) & (df[date_col] <= q_end)]
        except Exception:
            pass  # quietly skip bad period filters

    # --- 3️⃣ Choose grouping column (guardrail behavior)
    mode = "ICD"
    group_col = icd_col

    if cpt and not icd:
        mode = "CPT"
        group_col = cpt_col

    if group_col is None:
        return {"error": f"No {mode} column found in dataset."}

    # --- 4️⃣ Aggregate top cost drivers
    grouped = (
        df.groupby(group_col)[cost_col]
        .sum()
        .reset_index()
        .rename(columns={group_col: f"{mode} Code", cost_col: "Total Cost"})
        .sort_values("Total Cost", ascending=False)
        .head(top_n)
    )

    # Compute cost share
    total_cost = grouped["Total Cost"].sum()
    grouped["Cost Share (%)"] = (grouped["Total Cost"] / total_cost * 100).round(2)

    # --- 5️⃣ Return structured output for UI
    summary = [
        f"Top {mode} codes by cost in {period or 'the selected period'} "
        f"with {mode} {grouped.iloc[0][f'{mode} Code']} leading at "
        f"${grouped.iloc[0]['Total Cost']:,} ({grouped.iloc[0]['Cost Share (%)']}%)."
    ]

    return {
        "summary": summary,
        "tables": [grouped.to_dict(orient="records")],
        "citations": [f"{mode.lower()}.csv"],
        "next_steps": [f"Review high-cost {mode} codes for anomalies or overutilization."]
    }




def provider_anomalies(df: pd.DataFrame, code=None, metric='z', threshold=3.0):
    _require_cols(df, ["charge_amount", "provider_id"])
    d = df.copy()
    if code:
        d = d[(d["cp"].astype(str) == str(code)) | (d["icd10"] == code)]

    agg = d.groupby("provider_id")["charge_amount"].sum().to_frame("total_charge")
    mu, sigma = agg["total_charge"].mean(), agg["total_charge"].std(ddof=0)
    agg["zscore"] = (agg["total_charge"] - mu) / (sigma if sigma != 0 else 1.0)
    outliers = agg[agg["zscore"] >= float(threshold)].reset_index().sort_values("zscore", ascending=False)
    return {"table_name": "provider_outliers", "table": outliers}

def fraud_flags(df: pd.DataFrame, min_claims_per_patient=5, window_days=90):
    _require_cols(df, ["service_date", "provider_id", "patient_id"])
    d = df.copy()
    d["dt"] = pd.to_datetime(d["service_date"])
    cutoff = d["dt"].max() - pd.Timedelta(days=window_days)
    d = d[d["dt"] >= cutoff]

    gp = d.groupby(["provider_id", "patient_id"]).size().to_frame("claims_per_patient")
    suspicious = gp[gp["claims_per_patient"] >= min_claims_per_patient].reset_index()
    prov = suspicious.groupby("provider_id")["claims_per_patient"].mean().to_frame("avg_claims_per_patient").reset_index()
    prov["flag_reason"] = f"avg >= {min_claims_per_patient} claims/patient in last {window_days}d"
    return {"table_name": "fraud_flags", "table": prov.sort_values("avg_claims_per_patient", ascending=False)}

def risk_scoring(df: pd.DataFrame):
    _require_cols(df, ["patient_id", "icd10"])
    chronic_prefixes = ["I1", "E11", "J4", "N18"]  # example: HTN, Diabetes, COPD, CKD
    d = df.copy()
    d["chronic_flag"] = d["icd10"].astype(str).str[:3].isin([p[:3] for p in chronic_prefixes])
    score = d.groupby("patient_id")["chronic_flag"].sum().to_frame("risk_score").reset_index()
    return {"table_name": "risk_scores", "table": score.sort_values("risk_score", ascending=False)}
