
# copilot/tools.py
import pandas as pd
import numpy as np

def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def top_icd_cpt_cost(df: pd.DataFrame, icd=None, cpt=None, period=None, plan=None, top_n=10):
    """Return top cost drivers by CPT/ICD filtered by optional params."""
    _require_cols(df, ["paid_amount", "service_date"])
    d = df.copy()
    if plan and "plan_id" in d.columns:
        d = d[d["plan_id"] == plan]
    if icd and "icd10" in d.columns:
        d = d[d["icd10"].str.startswith(icd)]
    if cpt and "cpt" in d.columns:
        d = d[d["cpt"].astype(str) == str(cpt)]
    if period:
        # period like '2024Q2' or '2024-01:2024-06'
        if "Q" in str(period):
            year = int(str(period)[:4]); q = int(str(period)[-1])
            d["dt"] = pd.to_datetime(d["service_date"])
            qmask = (d["dt"].dt.year == year) & (d["dt"].dt.quarter == q)
            d = d[qmask]
        elif ":" in str(period):
            start, end = str(period).split(":")
            d["dt"] = pd.to_datetime(d["service_date"])
            d = d[(d["dt"] >= start) & (d["dt"] <= end)]
    # Group by CPT if present, else ICD
    if "cpt" in d.columns:
        g = d.groupby("cpt", dropna=False)["paid_amount"].sum().sort_values(ascending=False).head(top_n)
        table = g.reset_index().rename(columns={"paid_amount":"total_paid"})
        table["share"] = table["total_paid"] / table["total_paid"].sum()
        return {"table_name": "cost_drivers", "table": table}
    elif "icd10" in d.columns:
        g = d.groupby("icd10", dropna=False)["paid_amount"].sum().sort_values(ascending=False).head(top_n)
        table = g.reset_index().rename(columns={"paid_amount":"total_paid"})
        table["share"] = table["total_paid"] / table["total_paid"].sum()
        return {"table_name": "cost_drivers", "table": table}
    else:
        raise ValueError("Need cpt or icd10 columns.")

def provider_anomalies(df: pd.DataFrame, code=None, metric='z', threshold=3.0, by="provider_npi"):
    _require_cols(df, ["paid_amount", by])
    d = df.copy()
    if code:
        if "cpt" in d.columns:
            d = d[d["cpt"].astype(str) == str(code)]
        elif "icd10" in d.columns:
            d = d[d["icd10"] == code]
    agg = d.groupby(by)["paid_amount"].sum().to_frame("total_paid")
    mu, sigma = agg["total_paid"].mean(), agg["total_paid"].std(ddof=0)
    agg["zscore"] = (agg["total_paid"] - mu) / (sigma if sigma != 0 else 1.0)
    outliers = agg[agg["zscore"] >= float(threshold)].reset_index().sort_values("zscore", ascending=False)
    return {"table_name": "provider_outliers", "table": outliers}

def fraud_flags(df: pd.DataFrame, min_claims_per_patient=5, window_days=90, by="provider_npi"):
    _require_cols(df, ["service_date", by, "patient_id"])
    d = df.copy()
    d["dt"] = pd.to_datetime(d["service_date"])
    cutoff = d["dt"].max() - pd.Timedelta(days=window_days)
    d = d[d["dt"] >= cutoff]
    gp = d.groupby([by, "patient_id"]).size().to_frame("claims_per_patient")
    suspicious = gp[gp["claims_per_patient"] >= int(min_claims_per_patient)].reset_index()
    prov = suspicious.groupby(by)["claims_per_patient"].mean().to_frame("avg_claims_per_patient").reset_index()
    prov["flag_reason"] = f"avg >= {min_claims_per_patient} claims/patient over last {window_days}d"
    return {"table_name": "fraud_flags", "table": prov.sort_values("avg_claims_per_patient", ascending=False)}

def risk_scoring(df: pd.DataFrame, cohort=None, icd_col="icd10"):
    """Toy risk score: count chronic ICD groups per patient."""
    _require_cols(df, ["patient_id"])
    chronic_prefixes = ["I1", "E11", "J4", "N18"]  # HTN, DM2, COPD-ish, CKD-ish
    d = df.copy()
    if cohort and "plan_id" in d.columns:
        d = d[d["plan_id"] == cohort]
    if icd_col in d.columns:
        d["chronic_flag"] = d[icd_col].astype(str).str[:3].isin([p[:3] for p in chronic_prefixes])
        score = d.groupby("patient_id")["chronic_flag"].sum().to_frame("risk_score").reset_index()
    else:
        score = d.groupby("patient_id").size().to_frame("risk_score").reset_index()
    return {"table_name": "risk_scores", "table": score.sort_values("risk_score", ascending=False)}
