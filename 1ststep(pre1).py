
import pandas as pd
import numpy as np
import re

FILEPATH = r"C:\Users\Tuấn Minh\OneDrive\CREDIT\accepted_2007_to_2018Q4.csv.gz"
CHUNKSIZE = 300_000

USECOLS = [
    "id", "issue_d", "loan_status", "application_type",
    "dti", "annual_inc", "annual_inc_joint", "dti_joint",
    "fico_range_low", "fico_range_high", "term",
    "funded_amnt",      # ← THÊM: denominator cho Loss Given Default (LGD)
    "recoveries",       # ← THÊM: numerator cho Recovery Given Default (RGD)
]

def clean_term(term_series: pd.Series) -> pd.Series:
    s = term_series.astype(str).str.strip()
    return pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def parse_issue_d(series: pd.Series) -> pd.Series:
    # handles "Dec-2017", "Dec 2017", "12/2017", "2017-12"
    x = series.astype("string").str.strip().str.lower()
    x = x.replace({"nan": pd.NA, "none": pd.NA, "": pd.NA})

    month_map = {
        "jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
        "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"
    }

    out = pd.Series(pd.NA, index=x.index, dtype="string")

    m1 = x.str.extract(r"^([a-z]{3})[-\s]+(\d{4})$")
    mask1 = m1[0].notna() & m1[1].notna()
    out.loc[mask1] = m1.loc[mask1, 1] + "-" + m1.loc[mask1, 0].map(month_map) + "-01"

    m2 = x.str.extract(r"^(\d{1,2})[/-](\d{4})$")
    mask2 = out.isna() & m2[0].notna() & m2[1].notna()
    out.loc[mask2] = m2.loc[mask2, 1] + "-" + m2.loc[mask2, 0].astype(int).astype(str).str.zfill(2) + "-01"

    m3 = x.str.extract(r"^(\d{4})-(\d{1,2})$")
    mask3 = out.isna() & m3[0].notna() & m3[1].notna()
    out.loc[mask3] = m3.loc[mask3, 0] + "-" + m3.loc[mask3, 1].astype(int).astype(str).str.zfill(2) + "-01"

    out = out.fillna(series.astype("string"))
    return pd.to_datetime(out, errors="coerce")

# =============================================================================
#  COUNTERS
# =============================================================================
total_rows       = 0
dti_missing_rows = 0

# Explainers inside dti-missing subset
miss_annual_inc = 0
annual_inc_le0  = 0
miss_issue_d    = 0
joint_app       = 0
miss_fico       = 0
miss_term       = 0

# Distributions
by_loan_status = {}
by_year        = {}

# LGD accumulators — tính trên TOÀN BỘ high_risk loans (không chỉ DTI-missing)
HIGH_STATUSES = {"Late (31-120 days)", "Default", "Charged Off"}
lgd_recovery_sum   = 0.0   # Σ (recoveries / funded_amnt) per high_risk loan
lgd_n              = 0     # số high_risk loans có funded_amnt > 0

# Null counters cho recoveries + funded_amnt
null_recoveries_total   = 0
null_funded_amnt_total  = 0
null_recoveries_hr      = 0   # trong high_risk loans
null_funded_amnt_hr     = 0

# Sample IDs
sample_ids = []

# =============================================================================
#  MAIN LOOP
# =============================================================================
for chunk in pd.read_csv(FILEPATH, usecols=USECOLS, chunksize=CHUNKSIZE, low_memory=False):
    total_rows += len(chunk)

    # ── recoveries + funded_amnt: ép kiểu số ─────────────────────────────────
    chunk["funded_amnt"] = to_num(chunk["funded_amnt"])
    chunk["recoveries"]  = to_num(chunk["recoveries"])

    # ── Null count toàn chunk ─────────────────────────────────────────────────
    null_recoveries_total  += int(chunk["recoveries"].isna().sum())
    null_funded_amnt_total += int(chunk["funded_amnt"].isna().sum())

    # ── Tính LGD trên high_risk loans ────────────────────────────────────────
    ls_raw = chunk["loan_status"].astype("string").str.strip()
    mask_hr = ls_raw.isin(HIGH_STATUSES)

    null_recoveries_hr  += int((mask_hr & chunk["recoveries"].isna()).sum())
    null_funded_amnt_hr += int((mask_hr & chunk["funded_amnt"].isna()).sum())

    hr_valid = chunk.loc[
        mask_hr &
        chunk["recoveries"].notna() &
        chunk["funded_amnt"].notna() &
        (chunk["funded_amnt"] > 0)
    ].copy()

    if len(hr_valid) > 0:
        rr = (hr_valid["recoveries"] / hr_valid["funded_amnt"]).clip(0, 1)
        lgd_recovery_sum += float(rr.sum())
        lgd_n            += len(hr_valid)

    # ── DTI missing analysis (giữ nguyên logic gốc) ──────────────────────────
    dti  = to_num(chunk["dti"])
    mask = dti.isna()

    if mask.any():
        sub = chunk.loc[mask].copy()
        dti_missing_rows += len(sub)

        ann      = to_num(sub["annual_inc"])
        issue_dt = parse_issue_d(sub["issue_d"])
        fico_low = to_num(sub["fico_range_low"])
        fico_high= to_num(sub["fico_range_high"])
        term_m   = clean_term(sub["term"])

        miss_annual_inc += int(ann.isna().sum())
        annual_inc_le0  += int((ann.notna() & (ann <= 0)).sum())
        miss_issue_d    += int(issue_dt.isna().sum())
        joint_app       += int((sub["application_type"].astype("string").str.strip() == "Joint App").sum())
        miss_fico       += int((fico_low.isna() | fico_high.isna()).sum())
        miss_term       += int(term_m.isna().sum())

        ls2 = sub["loan_status"].astype("string").fillna("<NA>").str.strip()
        for k, v in ls2.value_counts().items():
            by_loan_status[k] = by_loan_status.get(k, 0) + int(v)

        yrs = issue_dt.dt.year.dropna().astype(int)
        for k, v in yrs.value_counts().items():
            by_year[int(k)] = by_year.get(int(k), 0) + int(v)

        take = sub["id"].dropna().head(20).tolist()
        sample_ids.extend(take)
        sample_ids = sample_ids[:50]

# =============================================================================
#  RESULTS
# =============================================================================
def pct(x, denom):
    return round(x / denom * 100, 4) if denom else 0

print("=" * 60)
print("  DTI DIAGNOSTIC + LGD VALIDATION REPORT")
print("=" * 60)

print(f"\nTotal rows          : {total_rows:,}")
print(f"DTI missing rows    : {dti_missing_rows:,}  ({pct(dti_missing_rows, total_rows):.4f}%)")

print("\n--- Inside DTI-missing rows (DTI is NA) ---")
print(f"annual_inc missing  : {miss_annual_inc:,}  ({pct(miss_annual_inc, dti_missing_rows)}%)")
print(f"annual_inc <= 0     : {annual_inc_le0:,}  ({pct(annual_inc_le0, dti_missing_rows)}%)")
print(f"issue_d missing     : {miss_issue_d:,}  ({pct(miss_issue_d, dti_missing_rows)}%)")
print(f"joint_app rows      : {joint_app:,}  ({pct(joint_app, dti_missing_rows)}%)")
print(f"missing fico        : {miss_fico:,}  ({pct(miss_fico, dti_missing_rows)}%)")
print(f"missing term        : {miss_term:,}  ({pct(miss_term, dti_missing_rows)}%)")

print("\n--- loan_status distribution inside DTI-missing ---")
for k, v in sorted(by_loan_status.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {k:25s} {v:8,d}  ({pct(v, dti_missing_rows)}%)")

print("\n--- issue_d year distribution inside DTI-missing (top 15) ---")
for k, v in sorted(by_year.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {k}: {v:,}  ({pct(v, dti_missing_rows)}%)")

print(f"\nSample IDs to manually inspect: {sample_ids[:20]}")

# ── LGD Section ───────────────────────────────────────────────────────────────
mean_rgd = lgd_recovery_sum / lgd_n if lgd_n > 0 else 0.0
lgd_actual = 1 - mean_rgd

print("\n" + "=" * 60)
print("  LGD VALIDATION (Recovery Given Default)")
print("=" * 60)
print(f"\nrecoveries null (all rows)        : {null_recoveries_total:,}  ({pct(null_recoveries_total, total_rows):.4f}%)")
print(f"funded_amnt null (all rows)       : {null_funded_amnt_total:,}  ({pct(null_funded_amnt_total, total_rows):.4f}%)")
print(f"recoveries null (high_risk only)  : {null_recoveries_hr:,}  ({pct(null_recoveries_hr, lgd_n):.4f}% of high_risk)")
print(f"funded_amnt null (high_risk only) : {null_funded_amnt_hr:,}  ({pct(null_funded_amnt_hr, lgd_n):.4f}% of high_risk)")
print(f"\nHigh_risk loans used for LGD      : {lgd_n:,}")
print(f"Mean Recovery Given Default (RGD) : {mean_rgd*100:.2f}%")
print(f"Loss Given Default (LGD) actual   : {lgd_actual*100:.2f}%")
print(f"LGD assumed (flat)                : 65.00%")
print(f"Delta LGD (actual − assumed)      : {(lgd_actual - 0.65)*100:+.2f}pp")
print(f"\nFormula: LGD = 1 − mean(recoveries / funded_amnt)")
print(f"         RGD = recoveries / funded_amnt  (clipped to [0, 1])")