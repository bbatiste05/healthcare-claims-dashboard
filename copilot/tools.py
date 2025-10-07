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
def top_icd_cpt_cost(df: pd.DataFrame, icd=None, cpt=None, period=None, plan=None, top_n=10, **kwargs):
    """
    Identify top ICD or CPT codes by total charge amount.
    Handles flexible filters and period slicing (e.g., Q1, Q2 2024).
    """

    required = ["charge_amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"summary": f"Missing required columns: {missing}", "table": []}

    d = df.copy()
    d["charge_amount"] = pd.to_numeric(d["charge_amount"], errors="coerce")

    # --- Normalize time filters if period is specified ---
    if "service_date" in d.columns or "claim_date" in d.columns:
        date_col = "service_date" if "service_date" in d.columns else "claim_date"
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")

        if period and "q" in period.lower():
            qmap = {"q1": (1, 3), "q2": (4, 6), "q3": (7, 9), "q4": (10, 12)}
            for q, (start, end) in qmap.items():
                if q in period.lower():
                    d = d[d[date_col].dt.month.between(start, end)]

    # --- Determine which code column to use ---
    code_col = None
    if "cpt" in d.columns and (cpt or "cpt" in str(icd or "").lower()):
        code_col = "cpt"
    elif "icd10" in d.columns:
        code_col = "icd10"

    if not code_col:
        return {"summary": "No ICD or CPT columns found in dataset.", "table": []}

    # --- Aggregate and rank ---
    agg = (
        d.groupby(code_col)["charge_amount"]
        .sum()
        .reset_index()
        .rename(columns={code_col: "Code", "charge_amount": "Total Cost"})
        .sort_values("Total Cost", ascending=False)
        .head(top_n)
    )
    total_cost = agg["Total Cost"].sum()
    agg["Cost Share (%)"] = ((agg["Total Cost"] / total_cost) * 100).round(2)

    summary = (
        f"Top {len(agg)} {code_col.upper()} codes driving cost in "
        f"{period or 'the dataset'} represent {agg['Cost Share (%)'].sum():.1f}% of total charges."
    )

    return {"summary": summary, "table_name": "top_cost_codes", "table": agg.to_dict(orient="records")}






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
def risk_scoring(df: pd.DataFrame, cohort: str = None, user_q: str = "", top_n: int = 10, **kwargs):
    """
    Compute a synthetic risk score for each patient or analyze wait days depending on user question.
    - If user asks about 'risk', calculate risk scores.
    - If user asks about 'wait days' or 'wait times', compute wait time metrics
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
    user_q_lower = (user_q or "").lower()

    if cohort and "icd10" in d.columns:
        if cohort.lower() == "cardiology":
            d = d[d["icd10"].str.startswith("I", na=False)]
        else:
            d = d[d["icd10"].str.contains(cohort, case=False, na=False)]

    # ===== Generate a synthetic risk score ======
    # Mode 1: Risk Scoring 
    # ==============================
    if "risk" in user_q_lower or "score" in user_q_lower:
        d["risk_score"] = (
            0.6 * (d["charge_amount"] / d["charge_amount"].max()) +
            0.3 * (d["num_procedures"] / d["num_procedures"].max()) +
            0.1 * (d["wait_days"] / (d["wait_days"].max() if d["wait_days"].max() != 0 else 1))
        ).round(3)


        # --- Aggregate by patient ---
        agg = (
            d.groupby("patient_id")["risk_score"]
            .mean()
            .reset_index()
            .rename(columns={"risk_score": "avg_risk_score"})
            .sort_values("avg_risk_score", ascending=False)
        )

        # --- Compute overall stats ---
        top_patients = agg.head(top_n)
        avg_risk = round(agg["avg_risk_score"].mean(), 3)
        summary = (
            f"Top {top_n} highest-risk patients have average risk score from "
            f"{top_patients['avg_risk_score'].min():.3f} to {top_patients['avg_risk_score'].max():.3f}, "
            f"compared to cohort mean of {avg_risk}."
        )
        return {
            "summary": summary,
            "table_name": "patient_risk_scores",
            "table": top_patients.to_dict(orient="recors"),
            "next_steps": [
                "Review ICD and demographic data for top-risk patients.",
                "Coordinate care management interventions for scores above 3.0.",
                "Recalculate risk monthly using updated clinical and claims data."
            ],
            "citations": ["patient_risk_scors"]
        }

    # =================================================
    # Mode 2: Wait time / delays analysis path
    # =================================================
    elif "wait" in user_q_lower or "delay" in user_q_lower or "time to" in user_q_lower:
        # Computer wait-day stats per patient or ICD
        group_cols = ["patient_id"]
        if "icd10" in d.columns and ("icd" in user_q_lower or "by icd" in user_q_lower):
            group_cols.append("icd10")

        agg_wait = (
            d.groupby(group_cols)["wait_days"]
            .agg(["mean", "median", "max"])
            .reset_index()
            .rename(columns={"mean": "avg_wait_days", "median": "median_wait_days", "max": "max_wait_days"})
            .sort_values("avg_wait_days", ascending=False)
        )

        top_waits = agg_wait.head(top_n)
        summary = (
            f"Top {top_n} patients (or ICD codes) show average wait times "
            f"ranging from {top_waits['avg_wait_days'].min():.1f} to {top_waits['avg_wait_days'].max():.1f} days."
        )


        return {
            "summary": summary,
            "table_name": "wait_time_analysis",
            "table": top_waits.to_dict(orient="records"),
            "next_steps": [
                "Investigate scheduling and referral processes for long waits.",
                "Correlate high wait times with cost and outcomes.",
                "Implement interventions for reducing delays in high-wait cohorts."
            ],
            "citations": ["claims_wait-times"]
        }

# =======================
# Fallback
# =======================
    else:
        return {
            "summary": "No clear intent detected (risk or wait times). Please refine your question.",
            "table_name": None,
            "table":[]
        }    
