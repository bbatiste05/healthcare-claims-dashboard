# copilot/tools.py
import pandas as pd
import numpy as np

def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def top_icd_cpt_cost(df: pd.DataFrame, icd=None, cpt=None, period=None, plan=None, top_n=10):
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





def provider_anomalies(df: pd.DataFrame, code=None, metric='z', threshold=1.5):
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

def fraud_flags(df: pd.DataFrame, min_claims_per_patient=10, window_days=90):
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


def risk_scoring(df: pd.DataFrame):
    _require_cols(df, ["patient_id", "icd10"])
    chronic_prefixes = ["I1", "E11", "J4", "N18"]  # example: HTN, Diabetes, COPD, CKD
    d = df.copy()
    d["chronic_flag"] = d["icd10"].astype(str).str[:3].isin([p[:3] for p in chronic_prefixes])
    score = d.groupby("patient_id")["chronic_flag"].sum().to_frame("risk_score").reset_index()
    return {"table_name": "risk_scores", "table": score.sort_values("risk_score", ascending=False)}
