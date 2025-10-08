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

def top_icd_cpt_cost(df: pd.DataFrame, icd: str = None, cpt: str = None,
                     period: str = None, plan: str = None, top_n: int = 10) -> Dict[str, Any]:
    """
    Return top cost drivers. If CPT data/column is missing or question asks ICD, falls back to ICD.
    period accepts formats:
      - '2024Q2'
      - '2024-01:2024-06'  (inclusive range)
      - None  (all data)
    """
    d = df.copy()

    # column resolution
    col_charge = _coerce_col(d, "charge_amount")
    col_cpt    = _coerce_col(d, "cpt")
    col_icd    = _coerce_col(d, "icd10")
    col_date   = _find_date_col(d)

    _require_cols(d, [col_charge])  # at minimum we need charge

    # optional period filter
    if col_date:
        if period:
            period = str(period).strip()
            if "Q" in period and len(period) in (6, 7): # e.g. 2024Q2 / Q2 2024
                year = int(period[:4]) if period[0].isdigit() else int(period[-4:])
                q = int(period[-1])
                start_month = (q-1)*3 + 1
                end_month = start_month + 2
                mask = (d[col_date].dt.year == year) & (d[col_date].dt.month.between(start_month, end_month))
                d = d[mask]
            elif ":" in period:
                start_s, end_s = period.split(":")
                try:
                    start = pd.to_datetime(start_s, errors="coerce")
                    end   = pd.to_datetime(end_s, errors="coerce")
                    d = d[(d[col_date] >= start) & (d[col_date] <= end)]
                except Exception:
                    pass

    # which axis to use?
    prefer_cpt = bool(cpt) or ("cpt" in (icd or "").lower() and col_cpt)
    use_cpt = bool(col_cpt) and prefer_cpt
    use_icd = bool(col_icd) and not use_cpt

    # if the user asked for CPT but we don't have CPT column → fallback to ICD
    axis = None
    if use_cpt:
        axis = ("CPT Code", col_cpt)
    elif use_icd:
        axis = ("ICD-10 Code", col_icd)
    elif col_cpt:
        axis = ("CPT Code", col_cpt)
    elif col_icd:
        axis = ("ICD-10 Code", col_icd)
    else:
        # nothing to group by, return empty
        total = float(d[col_charge].sum())
        return {
            "summary": "No CPT or ICD column found; cannot compute cost drivers.",
            "table_name": "cost_drivers",
            "table": []
        }

    label, col_group = axis

    # aggregate
    g = (
        d.groupby(col_group)[col_charge]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={col_group: label, col_charge: "Total Cost"})
    )
    total_cost = float(g["Total Cost"].sum())
    g["Cost Share (%)"] = g["Total Cost"].apply(lambda x: _pct(x, total_cost))
    g = _top_n(g, top_n)

    # summary
    if period:
        summary = f"Top {label.split()[0]} cost drivers for {period}."
    else:
        summary = f"Top {label.split()[0]} cost drivers across all claims."

    return {
        "summary": summary,
        "table_name": "cost_drivers",
        "table": g.to_dict(orient="records"),
        "citations": ["claims_df"]
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

    # compare quarters
    if (compare or (period and "_vs_" in str(period))) and col_date:
        pair = (compare or period)
        try:
            a, b = str(pair).split("_vs_")
            d["Quarter"] = d[col_date].apply(_quarter_label)
            pivot = (
                d.groupby([col_prov, "Quarter"])[col_charge]
                 .sum()
                 .unstack(fill_value=0)
                 .reset_index()
                 .rename(columns={col_prov: "provider_id"})
            )
            if a not in pivot.columns or b not in pivot.columns:
                return {
                    "summary": f"Could not find both quarters ({a} and {b}) in data.",
                    "table_name": "provider_quarter_comparison",
                    "table": []
                }
            pivot["Δ_Charge"] = pivot[b] - pivot[a]
            pivot["Δ_%"] = ((pivot[b] - pivot[a]) / pivot[a].replace(0, pd.NA) * 100).round(2)
            pivot["Flagged"] = pivot["Δ_%"].abs() >= 20
            out = pivot.sort_values("Δ_%", ascending=False)
            return {
                "summary": f"Compared provider billing between {a} and {b}. {int(out['Flagged'].sum())} providers showed ≥20% change.",
                "table_name": "provider_quarter_comparison",
                "table": out.to_dict(orient="records")
            }
        except Exception:
            pass  # fall through to z-score

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
