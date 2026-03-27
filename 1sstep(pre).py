import pandas as pd
import numpy as np

FILE = r"C:\Users\Tuấn Minh\OneDrive\CREDIT\accepted_2007_to_2018Q4.csv.gz"

def to_numeric_safe(s):
    return pd.to_numeric(s, errors="coerce")

def clean_term(s):
    return pd.to_numeric(
        s.astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

def normalize_loan_status(x):
    s = x.astype("string").str.strip()
    s = s.replace({"nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "": pd.NA})
    return s

def make_risk_label_3class(ls):
    low  = ls.isin(["Fully Paid", "Current"])
    med  = ls.isin(["In Grace Period", "Late (16-30 days)"])
    high = ls.isin(["Late (31-120 days)", "Default", "Charged Off"])

    out = np.select(
        [low, med, high],
        ["low_risk", "medium_risk", "high_risk"],
        default="unknown"
    )
    return pd.Series(out, index=ls.index)

usecols = [
    "loan_status",
    "loan_amnt",
    "funded_amnt",       # ← THÊM: denominator cho LGD
    "recoveries",        # ← THÊM: numerator cho Recovery Given Default (RGD)
    "fico_range_low",
    "fico_range_high",
    "dti",
    "term"
]

df = pd.read_csv(FILE, usecols=usecols, low_memory=False)

# Feature engineering
df["fico_avg"]    = (
    to_numeric_safe(df["fico_range_low"])
    + to_numeric_safe(df["fico_range_high"])
) / 2
df["dti"]         = to_numeric_safe(df["dti"])
df["term_m"]      = clean_term(df["term"])
df["loan_amnt"]   = to_numeric_safe(df["loan_amnt"])
df["funded_amnt"] = to_numeric_safe(df["funded_amnt"])
df["recoveries"]  = to_numeric_safe(df["recoveries"])

ls = normalize_loan_status(df["loan_status"])
df["risk_label"] = make_risk_label_3class(ls)

total = len(df)

mask_missing_fico = df["fico_avg"].isna()
mask_missing_dti  = df["dti"].isna()
mask_missing_term = df["term_m"].isna()
mask_unknown_risk = df["risk_label"].eq("unknown")

# High-risk mask — chỉ tính LGD preview trên high_risk loans
mask_high_risk = df["risk_label"].eq("high_risk")

print("Total rows:", total)

print("\n--- Reason counts (NOT exclusive) ---")
print(f"Missing fico_avg : {mask_missing_fico.sum():,}")
print(f"Missing dti      : {mask_missing_dti.sum():,}")
print(f"Missing term_m   : {mask_missing_term.sum():,}")
print(f"Unknown risk     : {mask_unknown_risk.sum():,}")

print("\n--- recoveries + funded_amnt null check ---")
print(f"Total high_risk loans            : {mask_high_risk.sum():,}")
print(f"recoveries null (all rows)       : {df['recoveries'].isna().sum():,}  ({df['recoveries'].isna().mean()*100:.2f}%)")
print(f"recoveries null (high_risk only) : {(mask_high_risk & df['recoveries'].isna()).sum():,}  ({(mask_high_risk & df['recoveries'].isna()).sum() / max(mask_high_risk.sum(),1)*100:.2f}% of high_risk)")
print(f"funded_amnt null (all rows)      : {df['funded_amnt'].isna().sum():,}  ({df['funded_amnt'].isna().mean()*100:.2f}%)")
print(f"funded_amnt null (high_risk)     : {(mask_high_risk & df['funded_amnt'].isna()).sum():,}  ({(mask_high_risk & df['funded_amnt'].isna()).sum() / max(mask_high_risk.sum(),1)*100:.2f}% of high_risk)")

# LGD preview: RGD = recoveries / funded_amnt → LGD = 1 - RGD
hr = df[mask_high_risk & df["recoveries"].notna() & df["funded_amnt"].notna() & (df["funded_amnt"] > 0)].copy()
hr["recovery_rate"] = (hr["recoveries"] / hr["funded_amnt"]).clip(0, 1)
lgd_actual = 1 - hr["recovery_rate"].mean()

print(f"\n--- LGD actual preview ---")
print(f"N high_risk loans with valid data : {len(hr):,}")
print(f"Mean Recovery Given Default (RGD) : {hr['recovery_rate'].mean()*100:.2f}%")
print(f"Mean Loss Given Default (LGD)     : {lgd_actual*100:.2f}%  (vs assumed flat 65%)")

# Final filter — đúng logic lọc của file chính
filtered_out = (
    mask_missing_fico
    | mask_missing_dti
    | mask_missing_term
    | mask_unknown_risk
)

print("\n--- Final ---")
print(f"Total rows removed by step (3): {filtered_out.sum():,}")
print(f"Percent removed               : {round(filtered_out.mean() * 100, 4)} %")
print(f"Rows kept (usable)            : {(~filtered_out).sum():,}")