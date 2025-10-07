import pandas as pd
import numpy as np

# === SAFE WRAPPER ===
def _safe_run(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {"summary": f"⚠️ Tool error: {str(e)}", "table": []}
    return wrapper


# === HELPER ===
def _require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ------------------------------
# 1️⃣ TOP ICD/CPT COST TOOL
# ------------------------------
@_safe_run
def top_icd_cpt_cost(df: pd.DataFrame, icd=None, cpt=None, period=None, plan=None, top_n=10):
    """
    Summarize top ICD or CPT codes driving total costs.
    """
    _require_cols(df, ["charge_amount"])

    d = df.copy()
    # Handle naming inconsistencies
    for alt in ["service_date", "claim_date", "date"]:
        if alt in d.columns:
            d["date"] = pd.to_datetime(d[alt], errors="coerce")
            break

    d["charge_amount"] = pd.to_numeric(d["charge_amount"], errors="coerce")
    d.dropna(subset=["charge_amount"], inplace=True)

    # Select grouping level
    group_col = "icd10" if "icd10" in d.columns else "cpt" if "cpt" in d.columns else None
    if not group_col:
        return {"summary": "No ICD or CPT column found in dataset.", "table": []}

    # Period filtering (e.g. "Q2 2024")
    if period:
        if "Q" in period:
            year, quarter = period.split("Q")
            year = int(year)
            q_num = int(quarter)
            q_start = (q_num - 1) * 3 + 1
            q_end = q_start + 2
            d = d[d["date"].dt.year == year]
            d = d[d["date"].dt.month.between(q_start, q_end)]

    # Aggregate totals
    agg = (
        d.groupby(group_col)["charge_amount"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    agg["Cost Share (%)"] = (agg["charge_amount"] / agg["charge_amount"].sum() * 100).round(2)
    agg.rename(columns={group_col: group_col.upper(), "charge_amount": "Total Cost"}, inplace=True)

    summary = (
        f"Top {len(agg)} {group_col.upper()} codes accounted for "
        f"{agg['Cost Share (%)'].sum():.1f}% of all charges"
        + (f" in {period}" if period else "")
        + "."
    )

    return {"summary": summary, "table_name": "top_cost_drivers", "table": agg}


# ------------------------------
# 2️⃣ PROVIDER ANOMALIES TOOL
# ------------------------------
@_safe_run
def provider_anomalies(df: pd.DataFrame, code=None, metric='z', threshold=3.0, period=None, **kwargs):
    """
    Detect provider outliers or compare quarter-over-quarter anomalies.
    """
    _require_cols(df, ["provider_id", "charge_amount"])
    d = df.copy()

    # --- Normalize charge column ---
    d["charge_amount"] = pd.to_numeric(d["charge_amount"], errors="coerce")
    d.dropna(subset=["charge_amount"], inplace=True)

    # --- Optional filter by code ---
    if code:
        for c in ["cpt", "icd10"]:
            if c in d.columns:
                d = d[d[c].astype(str) == str(code)]

    # --- Detect comparison mode ---
    if period and "_vs_" in str(period):
        try:
            q1, q2 = period.split("_vs_")
            qmap = {"Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)}

            for alt in ["claim_date", "service_date", "date"]:
                if alt in d.columns:
                    d["date"] = pd.to_datetime(d[alt], errors="coerce")
                    break

            d["month"] = d["date"].dt.month
            d["year"] = d["date"].dt.year

            def get_q(m): 
                for q, (s, e) in qmap.items():
                    if s <= m <= e: return q
                return None

            d["quarter"] = d["month"].map(get_q)
            d["period"] = d["year"].astype(str) + d["quarter"]

            pivot = (
                d.groupby(["provider_id", "period"])["charge_amount"]
                .sum()
                .unstack(fill_value=0)
            )

            if q1 in pivot.columns and q2 in pivot.columns:
                pivot["Δ_Charge"] = pivot[q2] - pivot[q1]
                pivot["Δ_%"] = ((pivot[q2] - pivot[q1]) / pivot[q1].replace(0, np.nan)) * 100
                pivot = pivot.reset_index()
                pivot["Flagged"] = pivot["Δ_%"].abs() >= 20

                summary = (
                    f"Compared billing between {q1} and {q2}: "
                    f"{pivot['Flagged'].sum()} providers showed ≥20% change."
                )

                return {
                    "summary": summary,
                    "table_name": "provider_quarter_comparison",
                    "table": pivot[["provider_id", q1, q2, "Δ_Charge", "Δ_%", "Flagged"]],
                }
        except Exception as e:
            return {"summary": f"Quarter comparison failed: {str(e)}", "table": []}

    # --- Standard Z-score anomaly detection ---
    agg = d.groupby("provider_id")["charge_amount"].sum().to_frame("total_charge")
    mu, sigma = agg["total_charge"].mean(), agg["total_charge"].std(ddof=0)
    agg["zscore"] = (agg["total_charge"] - mu) / (sigma if sigma != 0 else 1.0)
    agg["Flagged"] = agg["zscore"] >= float(threshold)

    outliers = agg[agg["Flagged"]].reset_index().sort_values("zscore", ascending=False)
    summary = (
        f"{len(outliers)} providers exhibit unusually high billing patterns (Z ≥ {threshold}). "
        f"Highest total charge: ${agg['total_charge'].max():,.0f}."
    )

    return {
        "summary": summary,
        "table_name": "provider_outliers",
        "table": outliers[["provider_id", "total_charge", "zscore", "Flagged"]],
    }


# ------------------------------
# 3️⃣ FRAUD FLAGS TOOL
# ------------------------------
@_safe_run
def fraud_flags(df: pd.DataFrame, min_claims_per_patient=5, window_days=90):
    """
    Flag providers with excessive claims per patient within a window.
    """
    _require_cols(df, ["provider_id", "patient_id", "claim_date"])
    d = df.copy()
    d["claim_date"] = pd.to_datetime(d["claim_date"], errors="coerce")

    recent = d.groupby(["provider_id", "patient_id"])["claim_date"].agg(["count", "min", "max"]).reset_index()
    recent["days_span"] = (recent["max"] - recent["min"]).dt.days
    flagged = recent[recent["count"] >= min_claims_per_patient]
    flagged = flagged[flagged["days_span"] <= window_days]

    summary = (
        f"{len(flagged)} providers flagged for ≥{min_claims_per_patient} claims per patient "
        f"within {window_days} days."
    )

    return {
        "summary": summary,
        "table_name": "fraud_flagged_providers",
        "table": flagged.to_dict(orient="records"),
    }


# ------------------------------
# 4️⃣ RISK SCORING TOOL
# ------------------------------
@_safe_run
def risk_scoring(df: pd.DataFrame, cohort: str = None, **kwargs):
    """
    Compute synthetic risk scores for each patient.
    - Uses available fields like charge_amount, num_procedures, and wait_days.
    - Optionally filters to cohort or ICD code subset.
    """
    available = df.columns.tolist()
    factors = [c for c in ["charge_amount", "num_procedures", "wait_days"] if c in available]
    if not factors:
        return {"summary": "No numeric risk factors available for scoring.", "table": []}

    d = df.copy()
    for f in factors:
        d[f] = pd.to_numeric(d[f], errors="coerce")

    if "patient_id" not in d.columns:
        return {"summary": "Missing patient_id column for risk computation.", "table": []}

    # Synthetic weighted score
    d["risk_score"] = (
        0.6 * (d["charge_amount"] / d["charge_amount"].max()) +
        0.3 * (d["num_procedures"] / d["num_procedures"].max() if "num_procedures" in d else 0) +
        0.1 * (d["wait_days"] / (d["wait_days"].max() if "wait_days" in d else 1))
    ).round(3)

    # Filter cohort if applicable
    if cohort and "icd10" in d.columns:
        if cohort.lower() == "cardiology":
            d = d[d["icd10"].str.startswith("I", na=False)]
        elif cohort.lower() == "oncology":
            d = d[d["icd10"].str.startswith("C", na=False)]

    if d.empty:
        return {"summary": f"No patients matched cohort '{cohort}'.", "table": []}

    agg = (
        d.groupby("patient_id")["risk_score"]
        .mean()
        .reset_index()
        .rename(columns={"risk_score": "avg_risk_score"})
        .sort_values("avg_risk_score", ascending=False)
    )

    summary = (
        f"Top {len(agg.head(10))} highest-risk patients identified "
        f"with mean risk scores between {agg['avg_risk_score'].head(10).min():.3f}–{agg['avg_risk_score'].head(10).max():.3f}."
    )

    return {"summary": summary, "table_name": "patient_risk_scores", "table": agg.head(10).to_dict(orient="records")}
