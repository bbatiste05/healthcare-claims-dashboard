# copilot/tools.py
import pandas as pd
import numpy as np

def _safe_run(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {
                "summary": f"⚠️ Tool execution failed: {e}",
                "table_name": "error",
                "table": [],
                "citations": [],
                "next_steps": ["Verify data format and column names."]
            }
    return wrapper

def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

@_safe_run        
def top_icd_cpt_cost(df: pd.DataFrame, icd=None, cpt=None, period=None, plan=None, top_n=10):
    # ✅ Flexible column name mapping
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Try to normalize key column names
    if 'service_date' not in df.columns:
        for alt in ['claim_date', 'date_of_service', 'dos']:
            if alt in df.columns:
                df.rename(columns={alt: 'service_date'}, inplace=True)
                break

    # Same for charge_amount
    if 'charge_amount' not in df.columns:
        for alt in ['billed_amount', 'total_charge', 'amount']:
            if alt in df.columns:
                df.rename(columns={alt: 'charge_amount'}, inplace=True)
                break

    # Final validation
    _require_cols(df, ["charge_amount", "service_date"])

    # ✅ Main logic
    if icd:
        group_field = "icd10"
    elif cpt:
        group_field = "cpt"
    else:
        # fallback: auto-detect ICD vs CPT columns
        if "icd10" in df.columns:
            group_field = "icd10"
        elif "cpt" in df.columns:
            group_field = "cpt"
        else:
            raise ValueError("No ICD or CPT code columns found.")

    summary = (
        df.groupby(group_field)["charge_amount"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    summary.columns = [group_field.upper(), "Total Cost"]
    summary["Cost Share (%)"] = (
        (summary["Total Cost"] / summary["Total Cost"].sum()) * 100
    ).round(2)

    return {"table_name": "cost_drivers", "table": summary}





@_safe_run 
def provider_anomalies(df: pd.DataFrame, code=None, metric='z', threshold=1.5, period=None, **kwargs):
    """
    Detect anomalies
    - If 'period' is given with 'compare' (e.g. '2024Q1_vs_2024Q2'), compares quarters.
    - Otherwise computes z-score outliers for total or mean charges.
    """
    
    required = ["provider_id", "charge_amount"]
    missing = [c for c in required if c not in df.colums]
    if missing:
       raise ValueError(f"Missing required columns: {missing}") 
    
                
    d = df.copy()
    d["charge_amount"] = pd.to_numeric(d["charge_amount"], error="coerce")
    d.dropna(subset=["charge_amount"], inplace=True)
    
   # --- Optional filter by code (ICD or CPT) ---
    if code:
        for c in ["cpt", "icd10"]:
            if c in d.columns:
                d = d[d[c].astype(str) == str(code)]

    # --- Detect comparison type ---
    if period and "_vs_" in str(period):
        # Compare between two quarters (e.g., '2024Q1_vs_2024Q2')
        try:
            q1, q2 = period.split("_vs_")
            qmap = {"Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)}

            def _get_q(month):
                for q, (start, end) in qmap.items():
                    if start <= month <= end:
                        return q
                return None

            d["quarter"] = pd.to_datetime(d["service_date"]).dt.month.map(_get_q)
            d["year"] = pd.to_datetime(d["service_date"]).dt.year

            d["period"] = d["year"].astype(str) + "Q" + d["quarter"].str[-1]
            pivot = (
                d.groupby(["provider_id", "period"])["charge_amount"]
                .sum()
                .unstack(fill_value=0)
            )

            if q1 in pivot.columns and q2 in pivot.columns:
                pivot["Δ_Charge"] = pivot[q2] - pivot[q1]
                pivot["Δ_%"] = (
                    (pivot[q2] - pivot[q1]) / pivot[q1].replace(0, pd.NA)
                ) * 100
                pivot = pivot.reset_index()
                pivot["Flagged"] = pivot["Δ_%"].abs() >= 20  # mark large changes
                summary = (
                    f"Compared provider billing between {q1} and {q2}. "
                    f"{(pivot['Flagged']).sum()} providers showed ≥20% change."
                )
                return {
                    "summary": summary,
                    "table_name": "provider_quarter_comparison",
                    "table": pivot[["provider_id", q1, q2, "Δ_Charge", "Δ_%", "Flagged"]],
                }

        except Exception as e:
            return {
                "summary": f"Error comparing quarters: {str(e)}",
                "table_name": "provider_quarter_comparison",
                "table": [],
            }

    # --- Default z-score anomaly detection ---
    agg = d.groupby("provider_id")["charge_amount"].sum().to_frame("total_charge")
    mu, sigma = agg["total_charge"].mean(), agg["total_charge"].std(ddof=0)
    agg["zscore"] = (agg["total_charge"] - mu) / (sigma if sigma != 0 else 1.0)
    agg["Flagged"] = agg["zscore"] >= float(threshold)

    outliers = agg[agg["Flagged"]].reset_index().sort_values("zscore", ascending=False)
    summary = (
        f"{len(outliers)} providers exhibit unusually high billing patterns "
        f"(Z ≥ {threshold}). Highest total charge: ${agg['total_charge'].max():,.0f}."
    )

    return {
        "summary": summary,
        "table_name": "provider_outliers",
        "table": outliers[["provider_id", "total_charge", "zscore", "Flagged"]],
    }
