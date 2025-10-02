# copilot/tools.py
import pandas as pd
import numpy as np

def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def top_icd_cpt_cost(df, icd=None, cpt=None, period=None, plan=None, top_n=10):
    # If CPT column has data, prefer it
    if "cpt" in df.columns and df["cpt"].notna().any():
        group_col = "cpt"
    else:
        group_col = "icd10"

    grouped = (
        df.groupby(group_col)["charge_amount"]
          .sum()
          .reset_index()
          .sort_values("charge_amount", ascending=False)
          .head(top_n)
    )
    grouped["Cost Share (%)"] = grouped["charge_amount"] / grouped["charge_amount"].sum() * 100
    return grouped.to_dict(orient="records")




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
