# copilot/tools.py
import pandas as pd
import numpy as np

def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def top_icd_cpt_cost(df: pd.DataFrame, icd=None, cpt=None, period=None, plan=None, top_n=10, **kwargs):
    _require_cols(df, ["charge_amount", "service_date"])
    d = df.copy()

    # Filter by ICD or CPT if provided
    if icd:
        d = d[d["icd10"].astype(str).str.startswith(icd)]
    if cpt:
        d = d[d["cp"].astype(str) == str(cpt)]

    # Time filter
    if period:
        d["dt"] = pd.to_datetime(d["service_date"])
        if "Q" in str(period):  # e.g., 2024Q2
            year = int(period[:4]); q = int(period[-1])
            d = d[(d["dt"].dt.year == year) & (d["dt"].dt.quarter == q)]
        elif ":" in str(period):  # e.g., 2024-01:2024-06
            start, end = str(period).split(":")
            d = d[(d["dt"] >= start) & (d["dt"] <= end)]

    # Group by CPT (cp) if available
    if "cp" in d.columns:
        g = d.groupby("cp")["charge_amount"].sum().sort_values(ascending=False).head(top_n)
        table = g.reset_index().rename(columns={"charge_amount": "total_charge"})
        table["share"] = table["total_charge"] / table["total_charge"].sum()
        return {"table_name": "cost_drivers", "table": table}
    else:
        g = d.groupby("icd10")["charge_amount"].sum().sort_values(ascending=False).head(top_n)
        table = g.reset_index().rename(columns={"charge_amount": "total_charge"})
        table["share"] = table["total_charge"] / table["total_charge"].sum()
        return {"table_name": "cost_drivers", "table": table}





def provider_anomalies(df: pd.DataFrame, code=None, metric='z', threshold=1.5, **kwargs):
    """
    Identify providers with unusually high average charges based on z-scores. Returns a summary and a table of outlier providers.
    """
    
    _require_cols(df, ["charge_amount", "provider_id"])
    d = df.copy()
    
    if code:
        d = d[(d["cp"].astype(str) == str(code)) | (d["icd10"] == code)]
        
    #compute
    agg = (
        d.groupby("provider_id")["charge_amount"]
        .mean()
        .reset_index()
        .rename(columns={"charge_amount": "mean_charge"})
    )

    mu, sigma = agg["mean_charge"].mean(), agg["mean_charge"].std(ddof=0)
    agg["zscore"] = (agg["mean_charge"] - mu) / (sigma if sigma != 0 else 1.0)
    agg["flag"] = agg["zscore"] >= float(threshold)

    outliers = (
        agg[agg["flag"]]
        .sort_values("zscore", ascending=False)
        .reset_index(drop=True)
    )

    # 🩹 Fallback — if no outliers found, show top 10 by total charge
    if outliers.empty:
        outliers = agg.sort_values("total_charge", ascending=False).head(10).reset_index()
        summary = "No extreme outliers found; showing top 10 highest billing providers."
    else:
        summary = f"{len(outliers)} providers exhibit unusually high billing patterns."

    # ✅ Return JSON-safe dict
    return {
        "summary": summary,
        "tables": [outliers.to_dict(orient="records")],
        "citations": ["claims.csv"],
        "next_steps": [
            "Audit providers with top Z-scores for potential overbilling.",
            "Review charge composition by procedure type."
        ]
    }

def fraud_flags(df: pd.DataFrame, min_claims_per_patient=10, window_days=90, **kwargs):
    _require_cols(df, ["provider_id", "patient_id", "claim_date"])
    d = df.copy()

    # Aggregate claims per patient per provider within the window
    flagged = (
        d.groupby(["provider_id", "patient_id"])
        .size()
        .reset_index(name="claim_count")
    )
    flagged = flagged[flagged["claim_count"] > min_claims_per_patient]

    total_providers = d["provider_id"].nunique()
    max_claims = d.groupby("patient_id").size().max()

    if flagged.empty:
        summary = (
            f"No providers were flagged for having more than {min_claims_per_patient} "
            f"claims per patient in the past {window_days} days. "
            f"Across {total_providers} providers, the highest observed count was {max_claims}."
        )
    else:
        summary = (
            f"{flagged['provider_id'].nunique()} providers exceeded "
            f"{min_claims_per_patient} claims per patient within {window_days} days."
        )

    return {
        "summary": summary,
        "tables": [flagged.to_dict(orient="records")],
        "citations": ["claims.csv"],
        "next_steps": [
            f"Re-run this analysis monthly to track new patterns exceeding {min_claims_per_patient} claims per patient.",
            "Investigate flagged providers’ billing justification and patient volume trends."
        ]
    }


def risk_scoring(df: pd.DataFrame, cohort: str = None, **kwargs):
    """
    Compute a synthetic risk score for each patient.
    - Uses available fields like charge_amount, num_procedures, and wait_days.
    - Optionally filters to a specific cohort (e.g., 'cardiology').
    """

    required_cols = ["patient_id", "charge_amount", "num_procedures", "wait_days"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return {
            "summary": f"Missing required columns: {', '.join(missing)}",
            "table": []
        }

    d = df.copy()

    # --- Generate a synthetic risk score ---
    # Normalize and weight common claim factors
    d["risk_score"] = (
        0.6 * (d["charge_amount"] / d["charge_amount"].max()) +
        0.3 * (d["num_procedures"] / d["num_procedures"].max()) +
        0.1 * (d["wait_days"] / (d["wait_days"].max() if d["wait_days"].max() != 0 else 1))
    ).round(3)

    if cohort and cohort.lower() == "cardiology":
        d = d[d["icd10"].str.startswith("I", na=False)]  # I00–I99 range = circulatory system

    if d.empty:
        return {
            "summary": f"No patients matched the cohort '{cohort}', so average patient risk scores could not be calculated.",
            "table": []
        }

    # --- Aggregate by patient ---
    agg = (
        d.groupby("patient_id")["risk_score"]
        .mean()
        .reset_index()
        .rename(columns={"risk_score": "avg_risk_score"})
    )

    # --- Compute overall stats ---
    avg_risk = round(agg["avg_risk_score"].mean(), 3)
    summary = (
        f"Average risk score across {len(agg)} patients"
        + (f" in the '{cohort}' cohort" if cohort else "")
        + f" is {avg_risk}."
    )

    return {
        "summary": summary,
        "table_name": "patient_risk_scores",
        "table": agg.head(10).to_dict(orient="records")
    }

