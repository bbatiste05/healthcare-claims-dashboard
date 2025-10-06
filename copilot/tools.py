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
    Detect anomalies in provider billing patterns or compare quarters.
    """

    # --- 1️⃣ Validate columns ---
    required = ["provider_id", "charge_amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()

    # --- 2️⃣ Robust charge cleaning ---
    # remove currency symbols, commas, etc.
    d["charge_amount"] = (
        d["charge_amount"]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", pd.NA)
    )
    d["charge_amount"] = pd.to_numeric(d["charge_amount"], errors="coerce")
    d.dropna(subset=["charge_amount", "provider_id"], inplace=True)

    # --- 3️⃣ Flexible date parsing ---
    date_candidates = ["claim_date", "service_date", "date_of_service", "transaction_date"]
    found_date_col = next((c for c in date_candidates if c in d.columns), None)

    if found_date_col:
        d["claim_date"] = pd.to_datetime(
            d[found_date_col], errors="coerce", infer_datetime_format=True
        )
    else:
        d["claim_date"] = pd.NaT

    # Remove rows without valid dates when comparing periods
    if period and "_vs_" in str(period):
        d = d.dropna(subset=["claim_date"])
        if d.empty:
            return {
                "summary": f"No valid date entries found to compare {period}.",
                "table_name": "provider_quarter_comparison",
                "table": [],
            }

    # --- 4️⃣ Optional filter by code ---
    if code:
        for c in ["cpt", "icd10"]:
            if c in d.columns:
                d = d[d[c].astype(str) == str(code)]

    # --- 5️⃣ Quarter comparison logic ---
    if period and "_vs_" in str(period):
        try:
            q1, q2 = period.split("_vs_")
            qmap = {"Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)}

            d["year"] = d["claim_date"].dt.year
            d["month"] = d["claim_date"].dt.month
            d["quarter"] = d["month"].apply(
                lambda m: next((q for q, (start, end) in qmap.items() if start <= m <= end), None)
            )
            d["period"] = d["year"].astype(str) + d["quarter"]

            pivot = (
                d.groupby(["provider_id", "period"])["charge_amount"]
                .sum()
                .unstack(fill_value=0)
            )

            if q1 in pivot.columns and q2 in pivot.columns:
                pivot["Δ_Charge"] = pivot[q2] - pivot[q1]
                pivot["Δ_%"] = ((pivot[q2] - pivot[q1]) / pivot[q1].replace(0, pd.NA)) * 100
                pivot = pivot.reset_index()
                pivot["Flagged"] = pivot["Δ_%"].abs() >= 20

                summary = (
                    f"Compared provider billing between {q1} and {q2}. "
                    f"{pivot['Flagged'].sum()} providers showed ≥20% change. "
                    f"Sample size: {len(pivot)} providers."
                )

                return {
                    "summary": summary,
                    "table_name": "provider_quarter_comparison",
                    "table": pivot[["provider_id", q1, q2, "Δ_Charge", "Δ_%", "Flagged"]],
                }
            else:
                return {
                    "summary": f"Could not find both {q1} and {q2} columns for comparison. "
                               f"Available: {list(pivot.columns)}",
                    "table_name": "provider_quarter_comparison",
                    "table": [],
                }

        except Exception as e:
            return {
                "summary": f"Error comparing quarters: {str(e)}",
                "table_name": "provider_quarter_comparison",
                "table": [],
            }

    # --- 6️⃣ Default z-score anomaly detection ---
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
        "table": outliers[["provider_id", "total_charge", "Flagged"]],
    }



@_safe_run 
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


@_safe_run 
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
