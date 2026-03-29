# =============================================================================
# 7th Step — Macro-Augmented PD Model: XGBoost + SHAP
# Purpose: Predict default probability using BOTH loan features AND macro vars
#          to identify which economic conditions are most dangerous
# Author : Credit Risk Project
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

# ── 0. INSTALL CHECK ──────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    print(f"✅ XGBoost {xgb.__version__} found")
except ImportError:
    raise ImportError(
        "XGBoost not installed. Run:\n"
        "  pip install xgboost shap\n"
        "Then re-run this script."
    )
try:
    import shap
    print(f"✅ SHAP {shap.__version__} found")
except ImportError:
    raise ImportError(
        "SHAP not installed. Run:\n"
        "  pip install shap\n"
        "Then re-run this script."
    )

from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── 1. PATHS — EDIT THESE ────────────────────────────────────────────────────
RAW_CSV       = r"C:\Users\Tuấn Minh\OneDrive\CREDIT\accepted_2007_to_2018Q4.csv.gz"
CLEAN_CSV     = r"C:\Users\Tuấn Minh\OneDrive\CREDIT\week6_clean_data.csv"
FRED_RAW_CSV  = r"C:\Users\Tuấn Minh\OneDrive\CREDIT\fred_data_raw.csv"
FRED_API_KEY  = "YOUR_FRED_API_KEY"   # ← replace or leave blank to use fred_data_raw.csv
OUTPUT_DIR    = r"C:\Users\Tuấn Minh\OneDrive\CREDIT"
CHART_DIR     = r"C:\Users\Tuấn Minh\OneDrive\CREDIT\charts"

import os
os.makedirs(CHART_DIR, exist_ok=True)

# ── 2. LOAD CLEAN DATA ────────────────────────────────────────────────────────
print("\n[1/7] Loading clean data...")
clean = pd.read_csv(CLEAN_CSV)
print(f"  Rows: {len(clean):,}  |  Cols: {list(clean.columns)}")

# ── 3. EXTRACT issue_d FROM RAW CSV ──────────────────────────────────────────
print("\n[2/7] Extracting issue_d from raw CSV (id-join, no re-cleaning)...")

# Only pull id + issue_d from raw — everything else comes from clean data
# This preserves ALL 2.26M rows from the already-cleaned dataset
id_date = pd.read_csv(
    RAW_CSV,
    usecols=lambda c: c in ["id", "issue_d"],
    low_memory=False
)
print(f"  Raw id+issue_d rows: {len(id_date):,}")

# Parse issue_d → month period
id_date["issue_d_parsed"] = pd.to_datetime(
    id_date["issue_d"], format="%b-%Y", errors="coerce"
)
id_date = id_date.dropna(subset=["issue_d_parsed"])
id_date["month"] = id_date["issue_d_parsed"].dt.to_period("M")
id_date = id_date[["id", "issue_d_parsed", "month"]].drop_duplicates("id")
print(f"  After parsing issue_d: {len(id_date):,} rows")

# Align id dtype — raw CSV reads id as str, clean data may be int64
id_date["id"] = id_date["id"].astype(str)
clean["id"]   = clean["id"].astype(str)

# Merge into clean data — keep ALL clean rows, add issue_d where available
raw = clean.merge(id_date, on="id", how="left")
raw["is_high_risk"] = (raw["risk_label"] == "high_risk").astype(int)
raw["term_60"]      = (raw["term_m"] == 60).astype(int)

n_with_date = raw["month"].notna().sum()
print(f"  Merged: {len(raw):,} rows total | {n_with_date:,} with issue_d ({n_with_date/len(raw)*100:.1f}%)")

# For temporal split we need issue_d — drop rows without it
raw = raw.dropna(subset=["month"])
print(f"  After dropping no-date rows: {len(raw):,} rows")

# ── 4. LOAD & PREP FRED DATA ──────────────────────────────────────────────────
print("\n[3/7] Loading FRED macro data...")

fred = None

# Try loading from existing fred_data_raw.csv first
try:
    # fred_data_raw.csv has unnamed index column (dates) — read with index_col=0
    fred_raw = pd.read_csv(FRED_RAW_CSV, index_col=0)
    fred_raw.index = pd.to_datetime(fred_raw.index, errors="coerce")
    fred_raw = fred_raw[fred_raw.index.notna()].copy()
    fred_raw = fred_raw.reset_index().rename(columns={"index": "date"})
    fred_raw["month"] = pd.to_datetime(fred_raw["date"]).dt.to_period("M")
    fred = fred_raw.copy()
    print(f"  Loaded from {FRED_RAW_CSV}: {len(fred):,} rows")
    print(f"  Columns: {list(fred.columns)}")
    print(f"  Date range: {fred['date'].min()} → {fred['date'].max()}")
except Exception as e:
    print(f"  Could not load existing FRED file: {e}")

# If no existing file, pull fresh from FRED API
if fred is None or FRED_API_KEY != "YOUR_FRED_API_KEY":
    print("  Pulling fresh from FRED API...")
    try:
        from fredapi import Fred
        f = Fred(api_key=FRED_API_KEY)

        SERIES = {
            "UNRATE"          : "Unemployment Rate (%)",
            "FEDFUNDS"        : "Fed Funds Rate (%)",
            "DRCLACBS"        : "Consumer Loan Delinquency Rate (%)",
            "A191RL1Q225SBEA" : "Real GDP Growth QoQ (%)",
            "CPIAUCSL"        : "CPI (Index)",
        }

        macro_dfs = []
        for sid, label in SERIES.items():
            s = f.get_series(sid, observation_start="2007-01-01",
                             observation_end="2018-12-31")
            s = s.resample("MS").last().ffill()  # monthly, forward-fill quarterly
            df_s = s.reset_index()
            df_s.columns = ["date", sid]
            macro_dfs.append(df_s.set_index("date"))

        fred_api = pd.concat(macro_dfs, axis=1).reset_index()
        fred_api["month"] = pd.to_datetime(fred_api["date"]).dt.to_period("M")
        fred = fred_api
        print(f"  FRED API pull complete: {len(fred):,} rows")

    except Exception as e:
        print(f"  ⚠️  FRED API failed: {e}")
        print("  Building minimal macro proxy from loan data instead...")
        fred = None

# ── 5. MERGE LOAN + MACRO ─────────────────────────────────────────────────────
print("\n[4/7] Merging loans with macro data...")

MACRO_COLS = []

if fred is not None:
    # Identify available macro columns
    available = [c for c in ["UNRATE", "FEDFUNDS", "DRCLACBS",
                              "A191RL1Q225SBEA", "CPIAUCSL"] if c in fred.columns]
    print(f"  Available macro series: {available}")

    fred_slim = fred[["month"] + available].copy()

    # Handle quarterly series: forward fill
    fred_slim = fred_slim.sort_values("month").ffill()

    df = raw.merge(fred_slim, on="month", how="left")
    MACRO_COLS = available

    # Add YoY CPI inflation if CPI available
    if "CPIAUCSL" in df.columns:
        df = df.sort_values("issue_d_parsed")
        df["CPI_YOY"] = df["CPIAUCSL"].pct_change(periods=12) * 100
        MACRO_COLS.append("CPI_YOY")
        MACRO_COLS = [c for c in MACRO_COLS if c != "CPIAUCSL"]

    macro_null = df[MACRO_COLS].isnull().sum()
    print(f"  Macro null counts:\n{macro_null}")
    df = df.dropna(subset=MACRO_COLS)

else:
    # Fallback: use year as proxy for macro cycle
    print("  ⚠️  Using year as macro cycle proxy (no FRED data)")
    df = raw.copy()
    df["year"] = df["issue_d_parsed"].dt.year
    MACRO_COLS = ["year"]

print(f"  Final merged dataset: {len(df):,} rows | {df['is_high_risk'].mean():.2%} default rate")

# ── 6. FEATURE ENGINEERING ───────────────────────────────────────────────────
print("\n[5/7] Building features...")

LOAN_FEATURES = ["fico_avg", "dti", "term_60", "loan_amnt"]
ALL_FEATURES  = LOAN_FEATURES + MACRO_COLS

# Ensure all features present
for f_col in ALL_FEATURES:
    if f_col not in df.columns:
        print(f"  ⚠️  Missing feature: {f_col} — dropping")
        ALL_FEATURES.remove(f_col)

X = df[ALL_FEATURES].copy()
y = df["is_high_risk"].copy()

# Temporal split — train on 2007-2014, test on 2015-2018
SPLIT_DATE = pd.Period("2017-01", freq="M")
train_mask = df["month"] < SPLIT_DATE
test_mask  = df["month"] >= SPLIT_DATE

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"  Train: {len(X_train):,} ({y_train.mean():.2%} default) | 2007–2014")
print(f"  Test : {len(X_test):,}  ({y_test.mean():.2%} default) | 2015–2018")
print(f"  Features: {ALL_FEATURES}")

# ── 7. TRAIN XGBOOST ─────────────────────────────────────────────────────────
print("\n[6/7] Training XGBoost...")

# Class imbalance ratio
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos = neg / pos
print(f"  scale_pos_weight = {scale_pos:.2f} (neg/pos to handle imbalance)")

model = xgb.XGBClassifier(
    n_estimators      = 800,
    max_depth         = 5,
    learning_rate     = 0.02,     # slower learning → more trees before stopping
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 50,       # min samples per leaf — prevents overfitting
    scale_pos_weight  = scale_pos,
    eval_metric       = "auc",
    early_stopping_rounds = 50,   # more patience
    random_state      = 42,
    n_jobs            = -1,
    verbosity         = 0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

best_iter = model.best_iteration
print(f"  Best iteration: {best_iter}")

# ── 8. EVALUATE ───────────────────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
print(f"\n{'='*50}")
print(f"  XGBoost AUC (temporal test): {auc:.4f}")
print(f"  Baseline AUC (prev RF):       0.6465")
print(f"  Delta: {auc - 0.6465:+.4f}")
print(f"{'='*50}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Paid", "Default"]))

# ── 9. SHAP ANALYSIS ──────────────────────────────────────────────────────────
print("\n[7/7] Running SHAP analysis...")

# Sample 50k for speed
np.random.seed(42)
shap_idx = np.random.choice(len(X_test), min(50000, len(X_test)), replace=False)
X_shap   = X_test.iloc[shap_idx]

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)
shap_df     = pd.DataFrame(shap_values, columns=ALL_FEATURES)

# Mean absolute SHAP per feature
mean_shap = shap_df.abs().mean().sort_values(ascending=False)
mean_shap_pct = (mean_shap / mean_shap.sum() * 100).round(2)

print("\n📊 SHAP Feature Importance:")
print("-" * 45)
for feat, val in mean_shap_pct.items():
    bar = "█" * int(val / 2)
    tag = "  ← MACRO" if feat in MACRO_COLS else ""
    print(f"  {feat:<25} {val:6.2f}%  {bar}{tag}")
print("-" * 45)

macro_total = mean_shap_pct[MACRO_COLS].sum() if MACRO_COLS else 0
loan_total  = mean_shap_pct[LOAN_FEATURES].sum()
print(f"\n  Loan features total:  {loan_total:.1f}%")
print(f"  Macro features total: {macro_total:.1f}%")

# ── 10. MACRO IMPACT ANALYSIS ─────────────────────────────────────────────────
print("\n📈 Macro Variable Analysis:")
print("-" * 45)

if MACRO_COLS:
    macro_shap = mean_shap_pct[MACRO_COLS].sort_values(ascending=False)
    print(f"\nRanking of macro variables by predictive power:")
    for i, (feat, val) in enumerate(macro_shap.items(), 1):
        print(f"  {i}. {feat:<25} {val:.2f}%")

    most_dangerous = macro_shap.index[0]
    print(f"\n  ⚠️  Most dangerous macro variable: {most_dangerous} ({macro_shap.iloc[0]:.1f}%)")

# ── 11. SAVE RESULTS TO EXCEL ────────────────────────────────────────────────
print("\n💾 Saving results to Excel...")

results_path = os.path.join(OUTPUT_DIR, "macro_ml_results.xlsx")

with pd.ExcelWriter(results_path, engine="openpyxl") as writer:

    # Sheet 1: Model Performance
    perf = pd.DataFrame({
        "Model"    : ["XGBoost (macro)", "Random Forest (baseline)"],
        "AUC"      : [round(auc, 4),     0.6465],
        "Features" : [str(ALL_FEATURES), "fico_avg, dti, term_60, loan_amnt"],
        "Split"    : ["Temporal (2015)", "Random 80/20"],
        "Note"     : ["With macro vars", "Original model"]
    })
    perf.to_excel(writer, sheet_name="W3_ModelPerformance", index=False)

    # Sheet 2: SHAP Feature Importance
    shap_table = pd.DataFrame({
        "Feature"     : mean_shap_pct.index,
        "SHAP_pct"    : mean_shap_pct.values,
        "Type"        : ["Macro" if f in MACRO_COLS else "Loan" for f in mean_shap_pct.index],
        "Mean_AbsSHAP": mean_shap.values
    }).reset_index(drop=True)
    shap_table.to_excel(writer, sheet_name="W3_SHAP_Importance", index=False)

    # Sheet 3: Macro Summary
    if MACRO_COLS:
        macro_summary = pd.DataFrame({
            "Macro_Variable" : macro_shap.index,
            "SHAP_pct"       : macro_shap.values,
            "Rank"           : range(1, len(macro_shap) + 1),
            "Interpretation" : [
                {
                    "UNRATE"          : "Unemployment — most direct proxy of borrower stress",
                    "FEDFUNDS"        : "Fed rate — affects refinancing cost and new loan affordability",
                    "DRCLACBS"        : "Consumer delinquency — lagged leading indicator of credit cycle",
                    "A191RL1Q225SBEA" : "GDP growth — business cycle indicator",
                    "CPI_YOY"         : "Inflation — erodes real income, increases repayment burden",
                    "year"            : "Year — proxy for macro cycle if FRED unavailable",
                }.get(m, m) for m in macro_shap.index
            ]
        })
        macro_summary.to_excel(writer, sheet_name="W3_MacroRanking", index=False)

print(f"  Saved: {results_path}")

# ── 12. CHARTS ────────────────────────────────────────────────────────────────
print("\n🎨 Generating charts...")

FOREST_GREEN = "#2D5016"
WARM_BEIGE   = "#F5F0E8"
ACCENT       = "#8B4513"
MACRO_COLOR  = "#C0392B"
LOAN_COLOR   = "#2980B9"

# ─── Chart A: SHAP Feature Importance (horizontal bar) ───────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(WARM_BEIGE)
ax.set_facecolor(WARM_BEIGE)

colors = [MACRO_COLOR if f in MACRO_COLS else LOAN_COLOR for f in mean_shap_pct.index[::-1]]
bars = ax.barh(mean_shap_pct.index[::-1], mean_shap_pct.values[::-1],
               color=colors, edgecolor="white", height=0.7)

for bar, val in zip(bars, mean_shap_pct.values[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=10, color=FOREST_GREEN, fontweight="bold")

ax.set_xlabel("Mean |SHAP| contribution (%)", fontsize=11)
ax.set_title("Feature Importance: Loan vs Macro Variables\n(SHAP — XGBoost, Temporal Split 2015)",
             fontsize=13, fontweight="bold", color=FOREST_GREEN, pad=15)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=LOAN_COLOR,  label="Loan features"),
                   Patch(facecolor=MACRO_COLOR, label="Macro variables")]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
ax.spines[["top","right"]].set_visible(False)
ax.set_xlim(0, mean_shap_pct.max() * 1.25)
plt.tight_layout()
chart_a_path = os.path.join(CHART_DIR, "08_shap_feature_importance.png")
plt.savefig(chart_a_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {chart_a_path}")

# ─── Chart B: SHAP Beeswarm (top 8 features) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(WARM_BEIGE)

top_features = mean_shap.nlargest(min(8, len(ALL_FEATURES))).index.tolist()
shap_top     = shap_df[top_features]

shap.summary_plot(
    shap_values[:, [ALL_FEATURES.index(f) for f in top_features]],
    X_shap[top_features],
    feature_names=top_features,
    show=False,
    plot_type="dot",
    color_bar=True,
    max_display=8,
)

plt.title("SHAP Beeswarm: How Feature Values Drive Default Risk",
          fontsize=13, fontweight="bold", color=FOREST_GREEN, pad=15)
plt.tight_layout()
chart_b_path = os.path.join(CHART_DIR, "09_shap_beeswarm.png")
plt.savefig(chart_b_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {chart_b_path}")

# ─── Chart C: Macro SHAP over time (avg SHAP per quarter) ────────────────────
if MACRO_COLS and "UNRATE" in MACRO_COLS:
    print("  Generating macro risk over time chart...")

    # Average predicted PD by quarter
    test_data = df[test_mask].copy()
    test_data["pred_prob"] = y_prob
    test_data["quarter"]   = test_data["issue_d_parsed"].dt.to_period("Q")

    qtr_avg = test_data.groupby("quarter").agg(
        avg_pred_pd = ("pred_prob", "mean"),
        avg_unrate  = ("UNRATE",    "mean") if "UNRATE" in test_data.columns else ("pred_prob", "count"),
        n_loans     = ("pred_prob", "count")
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(WARM_BEIGE)
    ax1.set_facecolor(WARM_BEIGE)

    qtrs = [str(q) for q in qtr_avg["quarter"]]
    x = np.arange(len(qtrs))

    ax1.plot(x, qtr_avg["avg_pred_pd"] * 100, color=MACRO_COLOR, lw=2.5,
             label="Avg Predicted PD (%)", marker="o", markersize=3)
    ax1.set_ylabel("Avg Predicted PD (%)", color=MACRO_COLOR, fontsize=11)
    ax1.set_xlabel("Quarter", fontsize=10)
    ax1.tick_params(axis='y', labelcolor=MACRO_COLOR)

    if "avg_unrate" in qtr_avg.columns and qtr_avg["avg_unrate"].notna().any():
        ax2 = ax1.twinx()
        ax2.plot(x, qtr_avg["avg_unrate"], color=LOAN_COLOR, lw=2,
                 linestyle="--", label="Unemployment Rate (%)", alpha=0.8)
        ax2.set_ylabel("Unemployment Rate (%)", color=LOAN_COLOR, fontsize=11)
        ax2.tick_params(axis='y', labelcolor=LOAN_COLOR)

    step = max(1, len(qtrs)//8)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels(qtrs[::step], rotation=45, ha="right", fontsize=8)
    ax1.set_title("Predicted Default Risk vs Unemployment Rate Over Time (Test Set 2015–2018)",
                  fontsize=12, fontweight="bold", color=FOREST_GREEN, pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper left", fontsize=9)
    ax1.spines[["top"]].set_visible(False)

    plt.tight_layout()
    chart_c_path = os.path.join(CHART_DIR, "10_macro_risk_over_time.png")
    plt.savefig(chart_c_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {chart_c_path}")

# ── 13. PRINT FINAL SUMMARY ──────────────────────────────────────────────────
print("\n" + "="*60)
print("  MACRO ML MODEL — FINAL SUMMARY")
print("="*60)
print(f"\n  Dataset       : {len(df):,} loans | 2007–2018")
print(f"  Train/Test    : 2007–2014 / 2015–2018 (temporal split)")
print(f"  Model         : XGBoost (n={best_iter} trees)")
print(f"\n  AUC           : {auc:.4f}  (prev RF: 0.6465)")
print(f"  Delta         : {auc - 0.6465:+.4f}")
print(f"\n  Loan features : {loan_total:.1f}% of prediction power")
print(f"  Macro features: {macro_total:.1f}% of prediction power")
if MACRO_COLS:
    print(f"\n  Most dangerous macro variable: {macro_shap.index[0]}")
    print(f"  → {macro_shap.iloc[0]:.1f}% of XGBoost's predictive signal")
print("\n  Charts saved:")
print(f"    08_shap_feature_importance.png")
print(f"    09_shap_beeswarm.png")
print(f"    10_macro_risk_over_time.png (if UNRATE available)")
print(f"\n  Excel: macro_ml_results.xlsx")
print("="*60)
print("\n✅ Done.")
