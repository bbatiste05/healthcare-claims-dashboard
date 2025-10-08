# copilot/tools.py
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

# ---------- small helpers ----------

_COL_ALIASES = {
    "cpt": ["cpt", "cpt_code", "procedure_code"],
    "icd10": ["icd10", "icd", "icd_code", "diagnosis_code"],
    "provider_id": ["provider_id", "npi", "provider"],
    "patient_id": ["patient_id", "member_id", "person_id"],
    "charge_amount": ["charge_amount", "amount", "charge", "total_charge_amount", "cost"],
    "service_date": ["service_date", "claim_date", "date_of_service", "dos", "date"]
}

def _coerce_col(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Find or create a canonical column; return its actual name in df or None."""
    for c in _COL_ALIASES.get(canonical, []):
        if c in df.columns:
            return c
    return None

def _find_date_col(df: pd.DataFrame) -> Optional[str]:
    c = _coerce_col(df, "service_date")
    if not c: 
        return None
    # make sure it's datetime
    if not pd.api.types.is_datetime64_any_dtype(df[c]):
        with pd.option_context("mode.chained_assignment", None):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return c

def _quarter_label(dt: pd.Timestamp) -> str:
    q = (dt.month - 1)//3 + 1
    return f"Q{q} {dt.year}"

def _top_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.head(n).reset_index(drop=True)

def _pct(x, total):
    return round(100 * (x / total), 2) if total else 0.0

def _require_cols(df: pd.DataFrame, needed: List[str]):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

# ---------- TOOLS ----------

def top_icd_cpt_cost(df: pd.DataFrame, icd=None, cpt=None,
                     period=None, plan=None, top_n: int = 10, **kwargs):
    """
    Identify top cost drivers by CPT or ICD code. 
    Falls back intelligently between CPT and ICD if one is missing.
    """
    d = df.copy()

    required = ["charge_amount"]
    if not any(col in d.columns for col in ["cpt", "icd10"]):
        raise ValueError("Dataset must include either 'cpt' or 'icd10' column.")

    # --- Optional period filter ---
    if period:
        # try to detect a usable date column
        for c in ["claim_date", "service_date", "date_of_service", "dos"]:
            if c in d.columns:
                d[c] = pd.to_datetime(d[c], errors="coerce")
                break
        else:
            c = None

        if c:
            year, q = None, None
            if "Q" in period:
                try:
                    year = int(period.split("Q")[0])
                    q = int(period.split("Q")[1])
                    month_range = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}[q]
                    d = d[d[c].dt.year == year]
                    d = d[d[c].dt.month.between(*month_range)]
                except Exception:
                    pass

    # --- CPT path preferred ---
    code_col = "cpt" if "cpt" in d.columns and d["cpt"].notna().any() else "icd10"
    label = "CPT Code" if code_col == "cpt" else "ICD-10 Code"

    # --- Aggregate ---
    cost_df = (
        d.groupby(code_col)["charge_amount"]
        .sum()
        .reset_index()
        .rename(columns={code_col: label, "charge_amount": "Total Cost"})
    )

    total_sum = cost_df["Total Cost"].sum()
    cost_df["Cost Share (%)"] = (cost_df["Total Cost"] / total_sum * 100).round(2)

    top = cost_df.sort_values("Total Cost", ascending=False).head(top_n)

    # --- Compose summary ---
    try:
        top_code = top.iloc[0][label] if label in top.columns else None
        top_share = top.iloc[0]["Cost Share (%)"]
        total_cost = top["Total Cost"].sum()
        top_mean = top["Total Cost"].mean()

        summary = (
            f"Top {top_n} {label}s drove approximately ${total_cost:,.0f} in total billed costs"
            + (f" during {period}" if period else "")
            + f". The highest individual {label} "
            + (f"({top_code}) " if top_code else "")
            + f"accounts for about {top_share:.1f}% of all costs. "
            f"Average cost among the top {top_n} {label}s is roughly ${top_mean:,.0f}. "
            "These results indicate that a small subset of procedure or diagnosis codes "
            "dominate overall expenditures, suggesting potential targets for cost-containment review."
        )

        context_note = (
            "Procedural (CPT) spending patterns may reflect utilization intensity or payer pricing differences."
            if label.lower() == "cpt"
            else "Diagnosis (ICD-10) patterns may highlight chronic or high-acuity case groups driving higher costs."
        )
        summary += " " + context_note
    except Exception:
        summary = (
            f"Top {top_n} {label}s driving total costs"
            + (f" in {period}" if period else "")
            + ". Summary generation limited due to missing data."
        )

    next_steps = [
        f"Review utilization and billing volume for top {label}s.",
        f"Cross-check with claim frequency by provider and period {period or 'overall'}.",
        "Investigate if unusually high costs align with expected clinical mix.",
    ]

    return {
        "summary": summary,
        "table_name": f"top_{label.lower().replace(' ', '_')}_cost",
        "table": top[[label, "Total Cost", "Cost Share (%)"]].to_dict(orient="records"),
        "next_steps": next_steps,
    }


def provider_anomalies(df: pd.DataFrame, code: str = None, metric: str = "z",
                       threshold: float = 3.0, period: str = None, compare: str = None) -> Dict[str, Any]:
    """
    - Default: z-score outliers on provider total charge.
    - If compare like '2024Q1_vs_2024Q2' → produce quarter-over-quarter table with deltas.
    - Supports filter by ICD/CPT code if provided.
    """
    d = df.copy()
    col_charge = _coerce_col(d, "charge_amount")
    col_prov   = _coerce_col(d, "provider_id")
    col_cpt    = _coerce_col(d, "cpt")
    col_icd    = _coerce_col(d, "icd10")
    col_date   = _find_date_col(d)

    _require_cols(d, [col_charge, col_prov])

    # optional filter by code
    if code and (col_cpt or col_icd):
        if col_cpt and d[col_cpt].astype(str).str.contains(str(code)).any():
            d = d[d[col_cpt].astype(str) == str(code)]
        elif col_icd:
            d = d[d[col_icd].astype(str) == str(code)]

    if period and "_vs_" in str(period):
        try:
            q1, q2 = period.split("_vs_")
            qmap = {"Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)}

            # --- Robust date column detection ---
            possible_dates = [c for c in ["claim_date", "service_date", "date_of_service", "dos"] if c in df.columns]
            if not possible_dates:
                raise ValueError("No date column found for quarter comparison.")
            date_col = possible_dates[0]

            d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
            d["month"] = d[date_col].dt.month
            d["year"] = d[date_col].dt.year

            # --- Map month to quarter ---
            def _get_quarter(m):
                for q, (start, end) in qmap.items():
                    if start <= m <= end:
                        return q
                return None

            d["quarter"] = d["month"].map(_get_quarter)
            d["period"] = d["year"].astype(str) + d["quarter"]

            # --- Aggregate charge amounts per provider per period ---
            pivot = (
                d.groupby(["provider_id", "period"])["charge_amount"]
                .sum()
                .unstack(fill_value=0)
            )

            # --- Flexible matching (handles “2024Q1” vs “Q1 2024”) ---
            pivot_cols = [c.replace(" ", "").replace("_", "") for c in pivot.columns]
            colmap = dict(zip(pivot.columns, pivot_cols))

            q1_clean = q1.replace(" ", "").replace("_", "")
            q2_clean = q2.replace(" ", "").replace("_", "")

            match_q1 = next((k for k, v in colmap.items() if q1_clean.lower() in v.lower()), None)
            match_q2 = next((k for k, v in colmap.items() if q2_clean.lower() in v.lower()), None)

            if not match_q1 or not match_q2:
                raise ValueError(f"Could not find both quarters ({q1} and {q2}) in data.")

            # --- Compute changes ---
            pivot["Δ_Charge"] = pivot[match_q2] - pivot[match_q1]
            pivot["Δ_%"] = (pivot["Δ_Charge"] / pivot[match_q1].replace(0, pd.NA)) * 100
            pivot["Flagged"] = pivot["Δ_%"].abs() >= 20

            pivot = pivot.reset_index()
            summary = (
                f"Compared provider billing between {match_q1} and {match_q2}. "
                f"{pivot['Flagged'].sum()} providers showed ≥20% change."
            )

            return {
                "summary": summary,
                "table_name": "provider_quarter_comparison",
                "table": pivot[["provider_id", match_q1, match_q2, "Δ_Charge", "Δ_%", "Flagged"]],
            }

        except Exception as e:
            return {
                "summary": f"Error comparing quarters: {str(e)}",
                "table_name": "provider_quarter_comparison",
                "table": [],
            }

    # default z-score on total charge
    agg = d.groupby(col_prov)[col_charge].sum().to_frame("total_charge")
    mu = float(agg["total_charge"].mean())
    sd = float(agg["total_charge"].std(ddof=0)) or 1.0
    agg["zscore"] = (agg["total_charge"] - mu) / sd
    agg["Flagged"] = agg["zscore"] >= float(threshold)
    outliers = agg[agg["Flagged"]].reset_index().rename(columns={col_prov: "provider_id"})
    outliers = outliers.sort_values("zscore", ascending=False)

    summary = f"{len(outliers)} providers with Z≥{threshold} on total charge."
    return {
        "summary": summary,
        "table_name": "provider_outliers",
        "table": outliers.to_dict(orient="records")
    }


def fraud_flags(df: pd.DataFrame, min_claims_per_patient: int = 10, window_days: int = 90) -> Dict[str, Any]:
    """Flag providers exceeding a threshold of claims per patient within a rolling window."""
    d = df.copy()
    col_prov = _coerce_col(d, "provider_id")
    col_pt   = _coerce_col(d, "patient_id")
    col_date = _find_date_col(d)

    _require_cols(d, [col_prov, col_pt])
    if not col_date:
        return {"summary": "Missing service/claim date column.", "table_name": "fraud_flags", "table": []}

    d = d.sort_values([col_pt, col_date])
    # rolling count of claims per patient within window
    d["window_start"] = d[col_date] - pd.Timedelta(days=int(window_days))
    # count claims per patient within window by provider
    out = (
        d.merge(
            d[[col_pt, col_date]].rename(columns={col_date: "date2"}),
            on=col_pt, how="left"
        )
        .query("date2 >= window_start and date2 <= @d[col_date]")
        .groupby([col_prov, col_pt, col_date]).size().reset_index(name="claims_in_window")
    )
    flagged = (
        out[out["claims_in_window"] > int(min_claims_per_patient)]
        .groupby(col_prov)["claims_in_window"].max()
        .reset_index()
        .rename(columns={col_prov: "provider_id"})
        .sort_values("claims_in_window", ascending=False)
    )
    summary = f"{len(flagged)} providers exceeded {min_claims_per_patient} claims/patient in {window_days} days."
    return {"summary": summary, "table_name": "fraud_flags", "table": flagged.to_dict(orient="records")}


def risk_scoring(df: pd.DataFrame, cohort: str = None, top_n: int = 10, by_icd: bool = False) -> Dict[str, Any]:
    """
    Synthetic risk score combining charge_amount + num_procedures + wait_days (if present).
    If by_icd=True, returns top patients with their dominant ICD.
    """
    d = df.copy()
    col_pt   = _coerce_col(d, "patient_id")
    col_amt  = _coerce_col(d, "charge_amount")
    col_proc = "num_procedures" if "num_procedures" in d.columns else None
    col_wait = "wait_days" if "wait_days" in d.columns else None
    col_icd  = _coerce_col(d, "icd10")

    _require_cols(d, [col_pt, col_amt])

    # fill missing optional columns
    if col_proc is None:
        d["num_procedures"] = 1
        col_proc = "num_procedures"
    if col_wait is None:
        d["wait_days"] = 0
        col_wait = "wait_days"

    # simple normalization
    for c in [col_amt, col_proc, col_wait]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    d["risk_score"] = (
        0.6 * (d[col_amt]  / (d[col_amt].max()  or 1)) +
        0.3 * (d[col_proc] / (d[col_proc].max() or 1)) +
        0.1 * (d[col_wait] / (d[col_wait].max() or 1))
    ).round(3)

    if cohort and cohort.lower() == "cardiology" and col_icd:
        d = d[d[col_icd].astype(str).str.upper().str.startswith("I")]  # I00–I99
        if d.empty:
            return {"summary": "No cardiology patients found.", "table_name": "patient_risk_scores", "table": []}

    # patient-level average
    patient_avg = (
        d.groupby(col_pt)["risk_score"]
         .mean()
         .reset_index()
         .rename(columns={col_pt: "patient_id", "risk_score": "avg_risk_score"})
    )

    if by_icd and col_icd:
        # dominant ICD for each patient (by total charges)
        icd_total = (
            d.groupby([col_pt, col_icd])[col_amt]
             .sum()
             .reset_index()
        )
        idx = icd_total.groupby(col_pt)[col_amt].idxmax()
        dom = icd_total.loc[idx].rename(columns={col_icd: "dominant_icd", col_amt: "icd_total_cost"})
        out = patient_avg.merge(dom[[col_pt, "dominant_icd", "icd_total_cost"]], on=col_pt, how="left") \
                         .rename(columns={col_pt: "patient_id"})
    else:
        out = patient_avg

    out = out.sort_values("avg_risk_score", ascending=False).head(int(top_n))
    summary = f"Top {len(out)} patients by risk score" + (" (with dominant ICD)" if by_icd and col_icd else "") + "."
    return {"summary": summary, "table_name": "patient_risk_scores", "table": out.to_dict(orient="records")}
