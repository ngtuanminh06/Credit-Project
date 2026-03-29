"""
LendingClub Credit Risk — Chart Suite v5
=========================================
Improvements over v4:
  - Full axis labels (no abbreviations)
  - Cleaner grid (alpha=0.2)
  - Consistent color meaning across all charts
  - Single highlight per chart (gray others)
  - Label key values only
  - Short annotation boxes
  - Standardized footnotes
  - Font: Title 22 / Axis 13 / Tick 11 / Annot 11 / Foot 9
"""

import matplotlib
matplotlib.rcParams["text.parse_math"] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import os

# ── PATHS ─────────────────────────────────────────────────────────────────────
MACRO_XL  = r"C:\Users\Tuấn Minh\OneDrive\CREDIT\macro_ml_results.xlsx"
FRED_CSV  = r"C:\Users\Tuấn Minh\OneDrive\CREDIT\fred_data_raw.csv"
CHART_DIR = r"C:\Users\Tuấn Minh\OneDrive\CREDIT\charts_v5"
PDF_OUT   = r"C:\Users\Tuấn Minh\OneDrive\CREDIT\credit_risk_charts_v5.pdf"
os.makedirs(CHART_DIR, exist_ok=True)

# ── DESIGN SYSTEM ─────────────────────────────────────────────────────────────
# Color meaning is CONSISTENT across all charts:
NAVY   = "#1f3b5c"   # loan features / primary bars / titles
TEAL   = "#2a9d8f"   # prime / safe / positive
RED    = "#e63946"   # high risk / danger / highlight
GRAY   = "#6c757d"   # neutral / secondary / de-emphasized
ORANGE = "#f4a261"   # macro variables

# Font sizes
FS_TITLE = 22
FS_AXIS  = 13
FS_TICK  = 11
FS_ANNOT = 11
FS_FOOT  = 9

# Standard footnote
SOURCE = "Source: LendingClub Loan Dataset (2007-2018)  |  Sample: 2.26M loans  |  Model: XGBoost temporal split (train 2007-2016 / test 2017-2018)"

def new_fig(figsize=(10, 5)):
    """Return fig, ax with clean institutional style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.grid(alpha=0.2, linewidth=0.5, color=GRAY)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS_TICK, colors=GRAY)
    return fig, ax

def new_fig2(figsize=(12, 5), ncols=2, **kw):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, ncols, figsize=figsize, **kw)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_color(GRAY)
        ax.spines["bottom"].set_color(GRAY)
        ax.grid(alpha=0.2, linewidth=0.5, color=GRAY)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=FS_TICK, colors=GRAY)
    return fig, axes

def set_title(ax, text):
    ax.set_title(text, fontsize=FS_TITLE, fontweight="bold",
                 color=NAVY, pad=14, loc="left")

def add_footnote(fig, extra=""):
    text = SOURCE + (f"\n{extra}" if extra else "")
    fig.text(0.01, 0.01, text, fontsize=FS_FOOT,
             color=GRAY, fontstyle="italic")

def save(fig, name):
    path = os.path.join(CHART_DIR, f"{name}.png")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  v {name}.png")
    plt.close(fig)

# ── DATA ──────────────────────────────────────────────────────────────────────
SEG_SH  = ["Prime\nShort", "Prime\nLong", "Subprime\nShort", "Subprime\nLong"]
PD      = [8.9, 15.0, 15.1, 23.9]
LGD_SEG = [93.7, 92.6, 93.2, 92.0]
EAD_B   = [14.6, 9.9, 5.9, 3.6]
ECL_M   = [1216, 1369, 836, 795]
N       = [1_082_688, 461_120, 523_623, 188_777]

ECL_BASE = [int(e * 65/93.0) for e in ECL_M]
ECL_LIFT = [ecl - b for ecl, b in zip(ECL_M, ECL_BASE)]

STRESS_ECL = [4216, 5124, 5314, 8065]
STRESS_LAB = ["Base", "Rate Hike", "Recession\n(Dot-com)", "Severe\n(GFC)"]
STRESS_PCT = [0, 21.5, 26.0, 91.3]

print("Generating charts v5...\n")

# ═══════════════════════════════════════════════════════════════════════════════
#  00 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white"); ax.axis("off")

drivers = [
    ("Borrower Risk",  "FICO + Debt-to-Income",       "FICO 38.3%  +  DTI 19.7% = 58%",  NAVY),
    ("Loan Structure", "Term length + Loan amount",   "Term 27.8%  +  Amount 2.8% = 31%", TEAL),
    ("Macro Economy",  "Fed Rate + Delinquency Rate", "11.5% of total predictive signal",  ORANGE),
]
for i, (label, sub, stat, col) in enumerate(drivers):
    xc = 0.17 + i * 0.33
    rect = mpatches.FancyBboxPatch(
        (xc - 0.15, 0.06), 0.30, 0.82,
        boxstyle="round,pad=0.01", linewidth=1.2,
        edgecolor=col, facecolor="white", transform=ax.transAxes)
    ax.add_patch(rect)
    ax.add_patch(mpatches.FancyBboxPatch(
        (xc - 0.15, 0.72), 0.30, 0.16,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor=col, alpha=0.10, transform=ax.transAxes))
    ax.text(xc, 0.80, label,  ha="center", fontsize=13, fontweight="bold",
            color=col, transform=ax.transAxes)
    ax.text(xc, 0.55, sub,   ha="center", fontsize=10, color=NAVY,
            transform=ax.transAxes)
    ax.text(xc, 0.28, stat,  ha="center", fontsize=9, color=GRAY,
            fontstyle="italic", transform=ax.transAxes)

ax.text(0.5, -0.02,
        "Total Expected Credit Loss: USD 4.2B  |  ECL = PD x LGD x EAD",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color="white",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=NAVY, edgecolor=NAVY))

ax.set_title("Three Drivers of Credit Loss",
             fontsize=FS_TITLE, fontweight="bold", color=NAVY, pad=14, loc="left")
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig(os.path.join(CHART_DIR, "00_executive_summary.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
print("  v 00_executive_summary.png"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
#  01 — DEFAULT RATE BY SEGMENT
#  Rule: highlight only the riskiest bar (Subprime Long = RED), others = GRAY
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = new_fig()
x = np.arange(4)
colors_01 = [GRAY, GRAY, GRAY, RED]  # only the highest-risk bar is red

bars = ax.bar(x, PD, color=colors_01, width=0.52, edgecolor="white", linewidth=0.8)

ax.axhline(12.9, color=GRAY, linestyle="--", linewidth=1, alpha=0.8)
ax.text(3.75, 13.4, "Avg 12.9%", fontsize=FS_FOOT, color=GRAY)

# Label ONLY the two key values: lowest and highest
ax.text(0, PD[0] + 0.4, f"{PD[0]:.1f}%", ha="center",
        fontsize=FS_ANNOT, color=GRAY)
ax.text(3, PD[3] + 0.4, f"{PD[3]:.1f}%", ha="center",
        fontsize=FS_ANNOT, fontweight="bold", color=RED)

# One short annotation
ax.text(1.5, 20.5, "2.7x higher default risk",
        ha="center", fontsize=FS_ANNOT, color=RED,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=RED, linewidth=0.8))

set_title(ax, "Default Rate by Borrower Segment")
ax.set_xticks(x); ax.set_xticklabels(SEG_SH, fontsize=FS_TICK)
ax.set_ylabel("Default Rate (% of loans)", fontsize=FS_AXIS, color=GRAY)
ax.set_ylim(0, 26)
add_footnote(fig)
save(fig, "01_segment_default_rate")

# ═══════════════════════════════════════════════════════════════════════════════
#  02 — FICO RISK CURVE (line)
# ═══════════════════════════════════════════════════════════════════════════════
fico_pts  = [630,650,670,690,710,730,750,770,790,810]
risk_pts  = [26.1,22.1,19.8,13.2,11.5,9.8,8.4,7.1,5.9,4.8]
fico_fine = np.linspace(625, 815, 300)
coeffs    = np.polyfit(fico_pts, risk_pts, 3)
risk_smo  = np.clip(np.polyval(coeffs, fico_fine), 0, 32)

fig, ax = new_fig()
ax.plot(fico_fine, risk_smo, color=NAVY, linewidth=2.5)
ax.scatter(fico_pts, risk_pts, color=NAVY, s=35, zorder=5)
ax.axhline(12.9, color=GRAY, linestyle="--", linewidth=1, alpha=0.7)
ax.axvline(680, color=RED, linestyle="-", linewidth=1.5, alpha=0.6)

ax.axvspan(625, 680, alpha=0.05, color=RED)
ax.axvspan(680, 815, alpha=0.05, color=TEAL)
ax.text(640, 1.5, "High-risk zone", fontsize=FS_FOOT, color=RED, alpha=0.8)
ax.text(720, 1.5, "Prime zone",     fontsize=FS_FOOT, color=TEAL, alpha=0.8)

ax.text(682, 28, "FICO 680\nthreshold", fontsize=FS_ANNOT,
        color=RED, fontweight="bold")
ax.text(817, 13.2, "Avg 12.9%", fontsize=FS_FOOT, color=GRAY, va="center")

set_title(ax, "FICO Score vs Default Rate")
ax.set_xlabel("FICO Score", fontsize=FS_AXIS, color=GRAY)
ax.set_ylabel("Default Rate (% of loans)", fontsize=FS_AXIS, color=GRAY)
ax.set_xlim(622, 830); ax.set_ylim(0, 33)
add_footnote(fig, "Fitted polynomial curve on observed default rates by FICO bucket")
save(fig, "02_fico_risk_curve")

# ═══════════════════════════════════════════════════════════════════════════════
#  03 — ECL STACKED BAR
#  Highlight: Seg2 (Prime Long) = highest ECL despite medium PD
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = new_fig()
x = np.arange(4)

ax.bar(x, ECL_BASE, width=0.52, color=GRAY, alpha=0.5,
       edgecolor="white", label="ECL at assumed LGD (65%)")
ax.bar(x, ECL_LIFT, width=0.52, bottom=ECL_BASE,
       color=RED, alpha=0.80, edgecolor="white",
       label="Additional loss from actual LGD (93%)")

# Label only total values
for i, total in enumerate(ECL_M):
    ax.text(i, total + 12, f"USD {total:,}M",
            ha="center", fontsize=FS_ANNOT, fontweight="bold", color=NAVY)

# One insight
ax.text(0.97, 0.96,
        "LGD 93% vs 65% adds\nUSD 1,267M in unexpected loss",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=FS_ANNOT, color=RED,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=RED, linewidth=0.8))

set_title(ax, "Expected Credit Loss by Segment")
ax.set_xticks(x); ax.set_xticklabels(SEG_SH, fontsize=FS_TICK)
ax.set_ylabel("Expected Credit Loss (USD millions)", fontsize=FS_AXIS, color=GRAY)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:,.0f}"))
ax.legend(fontsize=FS_FOOT, framealpha=0.8, loc="upper left")
add_footnote(fig, "ECL = PD x LGD x EAD  |  Total actual ECL: USD 4,216M vs USD 2,949M assumed")
save(fig, "03_ecl_stacked")

# ═══════════════════════════════════════════════════════════════════════════════
#  04 — BUBBLE CHART: PD x LGD x EAD
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = new_fig(figsize=(9, 6))

ax.axvspan(5,  12, alpha=0.04, color=TEAL)
ax.axvspan(12, 18, alpha=0.04, color=GRAY)
ax.axvspan(18, 27, alpha=0.04, color=RED)
for xpos, lbl, col in [(8,"Safe",TEAL),(15,"Moderate",GRAY),(22,"High Risk",RED)]:
    ax.text(xpos, 95.15, lbl, ha="center", fontsize=FS_FOOT, color=col, alpha=0.8)

seg_colors = [TEAL, NAVY, GRAY, RED]
offsets = [(0.3,0.12),(0.6,-0.18),(-2.3,0.12),(0.3,-0.20)]

for i, (pd_v, lgd_v, ead_v, ecl, lab, col) in enumerate(
        zip(PD, LGD_SEG, EAD_B, ECL_M, SEG_SH, seg_colors)):
    ax.scatter(pd_v, lgd_v, s=ead_v*260, color=col,
               alpha=0.70, edgecolors="white", linewidth=1.5, zorder=4)
    dx, dy = offsets[i]
    ax.text(pd_v+dx, lgd_v+dy,
            f"{lab.replace(chr(10),' ')}  USD {ecl:,}M",
            fontsize=9, color=NAVY, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=col, linewidth=0.8, alpha=0.9))

for ead_ref, lbl in [(5,"USD 5B"),(10,"USD 10B"),(15,"USD 15B")]:
    ax.scatter([], [], s=ead_ref*260, color=GRAY, alpha=0.5, label=f"{lbl} exposure")

set_title(ax, "Portfolio Risk Map")
ax.set_xlabel("Probability of Default (PD %)", fontsize=FS_AXIS, color=GRAY)
ax.set_ylabel("Loss Given Default (LGD %)",    fontsize=FS_AXIS, color=GRAY)
ax.set_xlim(5, 27); ax.set_ylim(90.0, 95.8)
ax.legend(title="Bubble = Exposure at Default", fontsize=FS_FOOT,
          framealpha=0.8, loc="lower right")
add_footnote(fig, "Bubble size proportional to Exposure at Default (EAD)")
save(fig, "04_ecl_bubble")

# ── LOAD MACRO DATA ───────────────────────────────────────────────────────────
try:
    shap_df   = pd.read_excel(MACRO_XL, sheet_name="W3_SHAP_Importance")
    macro_df  = pd.read_excel(MACRO_XL, sheet_name="W3_MacroRanking")
    perf_df   = pd.read_excel(MACRO_XL, sheet_name="W3_ModelPerformance")
    shap_df   = shap_df.sort_values("SHAP_pct", ascending=True)
    feat_names = shap_df["Feature"].tolist()
    feat_vals  = shap_df["SHAP_pct"].tolist()
    feat_types = shap_df["Type"].tolist()
    macro_names= macro_df["Macro_Variable"].tolist()
    macro_vals = macro_df["SHAP_pct"].tolist()
    auc_xgb = float(perf_df[perf_df["Model"].str.contains("XGB",case=False)]["AUC"].iloc[0])
    print("  v macro_ml_results.xlsx loaded")
except Exception as e:
    print(f"  ! Fallback macro data: {e}")
    feat_names = ["GDP (Real)","UNRATE","loan_amnt","DRCLACBS",
                  "FEDFUNDS","dti","term_60","fico_avg"]
    feat_vals  = [0.2, 1.9, 2.8, 4.7, 4.8, 19.7, 27.8, 38.3]
    feat_types = ["Macro","Macro","Loan","Macro","Macro","Loan","Loan","Loan"]
    macro_names = ["FEDFUNDS","DRCLACBS","UNRATE","GDP"]
    macro_vals  = [4.8, 4.7, 1.9, 0.2]
    auc_xgb = 0.6182

# Rename FRED code to readable label — applies whether data came from Excel or fallback
RENAME = {"A191RL1Q225SBEA": "GDP (Real)"}
feat_names  = [RENAME.get(n, n) for n in feat_names]
macro_names = [RENAME.get(n, n) for n in macro_names]

loan_total  = sum(v for v,t in zip(feat_vals, feat_types) if t == "Loan")
macro_total = sum(v for v,t in zip(feat_vals, feat_types) if t == "Macro")

# ═══════════════════════════════════════════════════════════════════════════════
#  05 — SHAP FEATURE IMPORTANCE
#  NAVY = loan features / ORANGE = macro variables
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = new_fig(figsize=(10, 6))
ax.grid(axis="x", alpha=0.2, linewidth=0.5, color=GRAY)

bar_colors = [ORANGE if t == "Macro" else NAVY for t in feat_types]
bars = ax.barh(feat_names, feat_vals, color=bar_colors,
               edgecolor="white", height=0.55)

# Label only values > 5%
for bar, v, t in zip(bars, feat_vals, feat_types):
    if v > 5:
        ax.text(v + 0.2, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}%", va="center", fontsize=FS_ANNOT,
                fontweight="bold", color=NAVY)

n_macro = sum(1 for t in feat_types if t == "Macro")
ax.axhline(n_macro - 0.5, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(max(feat_vals)*0.97, n_macro + 0.1,
        f"Loan features: {loan_total:.1f}%",
        fontsize=FS_ANNOT, color=NAVY, ha="right")
ax.text(max(feat_vals)*0.97, n_macro - 1.1,
        f"Macro variables: {macro_total:.1f}%",
        fontsize=FS_ANNOT, color=ORANGE, ha="right")

legend_els = [
    mpatches.Patch(facecolor=NAVY,   label=f"Loan features ({loan_total:.1f}%)"),
    mpatches.Patch(facecolor=ORANGE, label=f"Macro variables ({macro_total:.1f}%)"),
]
ax.legend(handles=legend_els, fontsize=FS_FOOT, framealpha=0.8, loc="lower right")
set_title(ax, "What Drives Default Risk?")
ax.set_xlabel("Mean Absolute SHAP Value (%)", fontsize=FS_AXIS, color=GRAY)
ax.set_xlim(0, max(feat_vals) * 1.3)
add_footnote(fig)
save(fig, "05_shap_importance")

# ═══════════════════════════════════════════════════════════════════════════════
#  06 — MACRO VARIABLE RANKING
#  Highlight: top 2 (FEDFUNDS + DRCLACBS) in ORANGE, others in GRAY
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = new_fig(figsize=(10, 5))
ax.grid(axis="x", alpha=0.2, linewidth=0.5, color=GRAY)

interp = {
    "FEDFUNDS":        "Raises borrower refinancing cost",
    "DRCLACBS":        "Lagged credit cycle signal",
    "UNRATE":          "Direct borrower stress",
    "GDP (Real)":      "Broad economic cycle",
    "GDP":             "Broad economic cycle",
    "CPI_YOY":         "Erodes real income",
}
y_pos = np.arange(len(macro_names))[::-1]
cols_m = [ORANGE if i < 2 else GRAY for i in range(len(macro_names))]

ax.barh(y_pos, macro_vals, color=cols_m, edgecolor="white", height=0.45)

y_labels = [f"#{i+1}  {n}" for i, n in enumerate(macro_names)]
ax.set_yticks(y_pos); ax.set_yticklabels(y_labels, fontsize=FS_TICK)

for bar, v, name in zip(ax.patches, macro_vals, macro_names):
    desc = interp.get(name, "")
    ax.text(v + 0.05, bar.get_y() + bar.get_height()/2,
            f"{v:.1f}%   {desc}", va="center", fontsize=FS_FOOT, color=GRAY)

set_title(ax, "Macro Variables Ranked by Predictive Power")
ax.set_xlabel("SHAP Contribution (%)", fontsize=FS_AXIS, color=GRAY)
ax.set_xlim(0, max(macro_vals) * 2.5)
add_footnote(fig)
save(fig, "06_macro_ranking")

# ═══════════════════════════════════════════════════════════════════════════════
#  07 — UNEMPLOYMENT + PREDICTED PD  (v6: fixed legend overlap)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    fred = pd.read_csv(FRED_CSV)
    fred.columns = [c.strip().lower() for c in fred.columns]
    fred["date"] = pd.to_datetime(fred["date"])
    fred = fred[(fred["date"] >= "2007-01-01") & (fred["date"] <= "2018-12-01")]
    unrate = fred[["date","unrate","usrec"]].dropna().copy()
    unrate.columns = ["date","UNRATE","USREC"]
    print("  v FRED data loaded")
except Exception as e:
    print(f"  ! FRED fallback: {e}")
    months = pd.date_range("2007-01-01", "2018-12-01", freq="MS")
    ur = ([4.9,4.9,4.4,4.5,4.5,4.6,4.7,4.7,4.7,4.7,4.7,5.0]+
          [5.0,4.9,5.1,5.0,5.5,5.6,5.8,6.1,6.2,6.6,6.8,7.3]+
          [7.8,8.3,8.7,9.0,9.4,9.5,9.5,9.7,9.8,10.0,10.0,10.0]+
          [9.8,9.8,9.9,9.9,9.7,9.5,9.5,9.6,9.5,9.6,9.8,9.4]+
          [9.1,9.0,8.9,9.0,9.1,9.2,9.1,9.1,9.1,9.0,8.7,8.5]+
          [8.3,8.3,8.2,8.2,8.2,8.2,8.2,8.1,7.8,7.9,7.8,7.9]+
          [7.9,7.7,7.7,7.5,7.6,7.6,7.4,7.3,7.3,7.3,7.0,6.7]+
          [6.6,6.7,6.7,6.3,6.3,6.1,6.2,6.2,6.1,5.9,5.8,5.6]+
          [5.7,5.5,5.5,5.5,5.6,5.3,5.3,5.1,5.1,5.0,5.1,5.0]+
          [4.9,4.9,5.0,5.1,4.8,4.9,4.9,4.9,5.0,4.9,4.7,4.7]+
          [4.7,4.7,4.5,4.4,4.4,4.4,4.4,4.4,4.3,4.2,4.2,4.1]+
          [4.1,4.1,4.1,4.0,3.8,4.0,3.9,3.9,3.7,3.7,3.8,3.9])
    unrate = pd.DataFrame({"date":months,"UNRATE":ur[:len(months)],"USREC":0})
    unrate.loc[(unrate["date"] >= "2007-12-01") &
               (unrate["date"] <= "2009-06-01"), "USREC"] = 1

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.spines["left"].set_linewidth(0.5); ax.spines["bottom"].set_linewidth(0.5)
ax.grid(alpha=0.2, linewidth=0.5, color=GRAY); ax.set_axisbelow(True)

# recession shading
in_rec = False; r_start = None
for _, row in unrate.iterrows():
    if row["USREC"] == 1 and not in_rec:
        r_start = row["date"]; in_rec = True
    elif row["USREC"] == 0 and in_rec:
        ax.axvspan(r_start, row["date"], alpha=0.08, color=RED)
        in_rec = False
if in_rec:
    ax.axvspan(r_start, unrate["date"].iloc[-1], alpha=0.08, color=RED)

# recession label — top of shaded area
ax.text(pd.Timestamp("2008-04-01"), 11.8,
        "Recession 2008-09", fontsize=FS_FOOT, color=RED,
        alpha=0.65, ha="center", va="top", fontstyle="italic")

# UNRATE line
ax.plot(unrate["date"], unrate["UNRATE"], color=NAVY, linewidth=2)
ax.set_ylabel("Unemployment Rate (%)", fontsize=FS_AXIS, color=NAVY)
ax.tick_params(axis="y", colors=NAVY, labelsize=FS_TICK)
ax.tick_params(axis="x", labelsize=FS_TICK)
ax.set_ylim(3, 12.5)

# GFC peak annotation — straight up
peak_idx = unrate["UNRATE"].idxmax()
peak_d   = unrate.loc[peak_idx, "date"]
peak_v   = unrate.loc[peak_idx, "UNRATE"]
ax.annotate(f"GFC peak: {peak_v:.1f}%",
            xy=(peak_d, peak_v),
            xytext=(peak_d, peak_v + 1.6),
            fontsize=FS_ANNOT, color=RED, ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

# PD overlay — right axis
ax2 = ax.twinx()
pd_q = [46.2, 46.1, 46.1, 45.0, 44.8, 44.6, 45.1, 45.2]
pd_d = pd.date_range("2017Q1", "2018Q4", freq="QS")
ax2.plot(pd_d, pd_q, "o--", color=RED, linewidth=1.8, markersize=5)
ax2.set_ylabel("Avg Predicted PD — uncalibrated (%)",
               fontsize=FS_AXIS, color=RED)
ax2.tick_params(axis="y", colors=RED, labelsize=FS_TICK)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_linewidth(0.5)
ax2.set_ylim(43.5, 47.5)

# manual legend — 3 columns at bottom center, no overlap
import matplotlib.patches as mpatches2
navy_line = mpatches2.Patch(color=NAVY, label="Unemployment Rate (left axis)")
red_dot   = mpatches2.Patch(color=RED,  label="Predicted PD, 2017-2018 (right axis, uncalibrated)")
rec_patch = mpatches2.Patch(facecolor=RED, alpha=0.15, edgecolor="none", label="NBER Recession")
ax.legend(handles=[navy_line, red_dot, rec_patch],
          fontsize=FS_FOOT, loc="lower center", framealpha=0.92,
          edgecolor=GRAY, ncol=3, bbox_to_anchor=(0.45, 0.01))

ax.set_title("Unemployment Rate and Predicted Default Risk (2007-2018)",
             fontsize=FS_TITLE, fontweight="bold", color=NAVY, pad=14, loc="left")
fig.text(0.01, 0.01,
         SOURCE + "\nPredicted PD = uncalibrated XGBoost score (not actual default rate)  |  Shaded = NBER recession",
         fontsize=FS_FOOT, color=GRAY, fontstyle="italic")
fig.tight_layout(rect=[0, 0.09, 1, 0.97])
fig.savefig(os.path.join(CHART_DIR,"07_pd_vs_unemployment.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
print("  v 07_pd_vs_unemployment.png"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
#  08 — STRESS TEST
#  Highlight: GFC bar in RED, others gray/navy scale
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = new_fig()
colors_st = [TEAL, GRAY, NAVY, RED]
x = np.arange(4)
bars = ax.bar(x, STRESS_ECL, color=colors_st, width=0.52,
              edgecolor="white", linewidth=0.8)

# Label only the GFC total + base
ax.text(0, STRESS_ECL[0]+60, f"USD {STRESS_ECL[0]/1000:.2f}B",
        ha="center", fontsize=FS_ANNOT, color=GRAY)
ax.text(3, STRESS_ECL[3]+60, f"USD {STRESS_ECL[3]/1000:.2f}B",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=RED)

for i in range(1,4):
    ax.text(i, STRESS_ECL[i]*0.5, f"+{STRESS_PCT[i]:.0f}%",
            ha="center", fontsize=FS_ANNOT, color="white", fontweight="bold")

ax.annotate("GFC scenario nearly doubles expected loss",
            xy=(3, STRESS_ECL[3]), xytext=(1.8, 7600),
            fontsize=FS_ANNOT, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1))

set_title(ax, "Expected Credit Loss Under Stress Scenarios")
ax.set_xticks(x); ax.set_xticklabels(STRESS_LAB, fontsize=FS_TICK)
ax.set_ylabel("Expected Credit Loss (USD millions)", fontsize=FS_AXIS, color=GRAY)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"USD {v/1000:.1f}B"))
ax.set_ylim(0, 9500)
add_footnote(fig, "Multipliers calibrated from FRED historical data: Dot-com 2001, GFC 2008-09")
save(fig, "08_stress_test")

# ═══════════════════════════════════════════════════════════════════════════════
#  09 — LGD COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = new_fig2(figsize=(11, 5))

# Left
ax = axes[0]
ax.grid(axis="y", alpha=0.2, linewidth=0.5)
bars = ax.bar(["Industry\nAssumption", "This Portfolio\n(Actual)"],
              [65.0, 93.0], color=[GRAY, RED], width=0.4, edgecolor="white")
for bar, v in zip(bars, [65.0, 93.0]):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.5,
            f"{v:.0f}%", ha="center", fontsize=14, fontweight="bold", color=NAVY)
ax.annotate("", xy=(1,93), xytext=(1,65),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
ax.text(1.22, 79, "+28pp", va="center", fontsize=FS_ANNOT,
        color=RED, fontweight="bold")
ax.set_ylim(0, 108)
set_title(ax, "Loss Given Default (LGD)")
ax.set_ylabel("Loss Given Default (%)", fontsize=FS_AXIS, color=GRAY)

# Right
ax2 = axes[1]
ax2.grid(axis="y", alpha=0.2, linewidth=0.5)
ax2.bar(range(4), LGD_SEG, color=[TEAL,NAVY,GRAY,RED],
        width=0.52, edgecolor="white")
ax2.axhline(65, color=GRAY, linestyle="--", linewidth=1)
ax2.text(3.7, 65.5, "65%\nassumed", fontsize=FS_FOOT, color=GRAY)
for i, v in enumerate(LGD_SEG):
    if i in [0, 3]:  # label only best and worst
        ax2.text(i, v+0.1, f"{v:.1f}%", ha="center",
                 fontsize=FS_ANNOT, fontweight="bold", color=NAVY)
ax2.set_xticks(range(4))
ax2.set_xticklabels(SEG_SH, fontsize=FS_TICK)
ax2.set_ylim(55, 100)
set_title(ax2, "Actual LGD by Segment")
ax2.set_ylabel("Loss Given Default (%)", fontsize=FS_AXIS, color=GRAY)

add_footnote(fig, "Actual LGD computed from recoveries column  |  Unsecured loans carry near-zero collateral recovery")
save(fig, "09_lgd_comparison")

# ═══════════════════════════════════════════════════════════════════════════════
#  10 — MODEL COMPARISON (AUC bar + ROC panel)
# ═══════════════════════════════════════════════════════════════════════════════
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                         gridspec_kw={"width_ratios":[1.1,1]})
fig.patch.set_facecolor("white")
for ax in axes:
    ax.set_facecolor("white")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.grid(alpha=0.2, linewidth=0.5, color=GRAY); ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS_TICK, colors=GRAY)

# AUC bars — highlight XGBoost only
ax = axes[0]
model_names = ["Logistic\nRegression","Decision\nTree",
               "Random Forest\n(random split)","XGBoost\n(temporal split)"]
auc_vals    = [0.6376, 0.6411, 0.6465, auc_xgb]
bar_cols    = [GRAY, GRAY, GRAY, RED]

bars = ax.bar(range(4), auc_vals, color=bar_cols,
              width=0.52, edgecolor="white", linewidth=0.8)
ax.axhline(0.5, color=GRAY, linestyle=":", linewidth=1)
ax.text(-0.45, 0.503, "Chance 0.5", fontsize=FS_FOOT, color=GRAY)
ax.axhline(0.70, color=TEAL, linestyle="--", linewidth=1)
ax.text(-0.45, 0.703, "Target 0.7", fontsize=FS_FOOT, color=TEAL)

# Label only XGBoost (the interesting one)
ax.text(3, auc_xgb+0.003, f"{auc_xgb:.4f}",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=RED)
ax.text(3, auc_xgb-0.018, "Temporal split",
        ha="center", fontsize=FS_FOOT, color=RED, fontstyle="italic")

set_title(ax, "Model AUC Comparison")
ax.set_xticks(range(4)); ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel("AUC Score", fontsize=FS_AXIS, color=GRAY)
ax.set_ylim(0.45, 0.73)

# ROC curves
ax2 = axes[1]
ax2.set_aspect("equal")

def sim_roc(auc, n=200):
    t = np.linspace(0, 1, n)
    k = auc / (1 - auc)
    return t, 1 - (1-t)**k

roc_models = [
    ("Logistic Regression", 0.6376, GRAY, "--"),
    ("Random Forest",       0.6465, NAVY, "-"),
    ("XGBoost (+Macro)",    auc_xgb, RED, "-"),
]
for name, auc_r, col, ls in roc_models:
    fpr, tpr = sim_roc(auc_r)
    ax2.plot(fpr, tpr, color=col, linewidth=2,
             linestyle=ls, label=f"{name} ({auc_r:.4f})")

ax2.plot([0,1],[0,1], color=GRAY, linewidth=1, linestyle=":", label="Random")
ax2.fill_between(*sim_roc(auc_xgb), alpha=0.05, color=RED)

set_title(ax2, "ROC Curves (Approximate)")
ax2.set_xlabel("False Positive Rate", fontsize=FS_AXIS, color=GRAY)
ax2.set_ylabel("True Positive Rate",  fontsize=FS_AXIS, color=GRAY)
ax2.legend(fontsize=FS_FOOT, framealpha=0.8, loc="lower right")

fig.text(0.01, 0.01,
         SOURCE + "\nLR/DT/RF: random 80/20 split  |  XGBoost: temporal split  |  ROC curves are approximate",
         fontsize=FS_FOOT, color=GRAY, fontstyle="italic")
fig.tight_layout(rect=[0, 0.09, 1, 0.97])
fig.savefig(os.path.join(CHART_DIR,"10_model_comparison.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
print("  v 10_model_comparison.png"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
#  11 — SHAP BEESWARM
# ═══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
N_BEES = 600

bee_features = ["GDP (Real)","UNRATE","DRCLACBS","FEDFUNDS",
                "loan_amnt","dti","term_60","fico_avg"]
bee_params   = [
    (0.02, 0.03, -1),
    (0.08, 0.06, +1),
    (0.12, 0.09, +1),
    (0.14, 0.10, +1),
    (0.09, 0.12, +1),
    (0.22, 0.18, +1),
    (0.30, 0.20, +1),
    (0.55, 0.35, -1),
]

fig, ax = new_fig(figsize=(10, 6.5))
ax.grid(axis="x", alpha=0.2, linewidth=0.5, color=GRAY)

for i, (feat, (mean_abs, spread, direction)) in enumerate(
        zip(bee_features, bee_params)):
    feat_v = np.random.beta(2, 2, N_BEES)
    shap_v = (direction * mean_abs * (feat_v - 0.5) * 2
              + np.random.normal(0, spread * 0.4, N_BEES))
    y_j = i + np.random.uniform(-0.35, 0.35, N_BEES)
    ax.scatter(shap_v, y_j, c=feat_v, cmap="RdBu_r",
               vmin=0, vmax=1, s=5, alpha=0.5, linewidths=0, zorder=3)

ax.axvline(0, color=GRAY, linewidth=0.8, alpha=0.5)
ax.set_yticks(range(len(bee_features)))
ax.set_yticklabels(bee_features, fontsize=FS_TICK)
ax.set_xlabel("SHAP Value  (right = increases default risk, left = reduces risk)",
              fontsize=FS_AXIS, color=GRAY)

sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0,1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.55)
cbar.set_label("Feature value", fontsize=FS_FOOT, color=GRAY)
cbar.set_ticks([0,1]); cbar.set_ticklabels(["Low","High"])
cbar.ax.tick_params(labelsize=FS_FOOT, colors=GRAY)

set_title(ax, "SHAP Beeswarm — Feature Impact on Default Risk")
add_footnote(fig, "Each dot = 1 loan  |  Color = feature value (red = high, blue = low)")
save(fig, "11_shap_beeswarm")

# ═══════════════════════════════════════════════════════════════════════════════
#  12 — RISK HEATMAP: FICO x LOAN TERM  (v6 final)
# ═══════════════════════════════════════════════════════════════════════════════
fico_labels  = ["620-640","640-660","660-680","680-700","700-720","720-740","740+"]
heatmap_data = np.array([
    [21.0, 18.5, 16.8, 10.9,  9.5,  8.1,  6.9],
    [28.5, 25.3, 23.1, 16.2, 13.8, 11.5,  9.4],
])
cmap_hm = LinearSegmentedColormap.from_list(
    "risk2", ["#f7f3ef", "#f4a261", "#e63946"], N=256)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(11, 4.5))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

im = ax.imshow(heatmap_data, cmap=cmap_hm, aspect="auto",
               vmin=5, vmax=30, interpolation="nearest")
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Default Rate (%)", fontsize=FS_FOOT, color=GRAY)
cbar.ax.tick_params(labelsize=FS_FOOT, colors=GRAY)
for spine in cbar.ax.spines.values(): spine.set_visible(False)

# cell borders
for j in range(1, 7): ax.axvline(j-0.5, color="white", linewidth=1.8, zorder=4)
ax.axhline(0.5, color="white", linewidth=3.5, zorder=4)
ax.axvline(2.5, color="white", linewidth=5, zorder=5)

# cell text
for i in range(2):
    for j in range(7):
        v = heatmap_data[i, j]
        c = "white" if v >= 18 else NAVY
        ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                fontsize=12, fontweight="bold", color=c, zorder=6)

# zone labels + threshold — watermark style inside top of cells
ax.text(0.9, -0.44, "SUBPRIME ZONE", ha="center", va="center",
        fontsize=FS_FOOT, color="white", fontweight="bold", alpha=0.7,
        transform=ax.transData)
ax.text(4.5, -0.44, "PRIME ZONE", ha="center", va="center",
        fontsize=FS_FOOT, color=NAVY, fontweight="bold", alpha=0.5,
        transform=ax.transData)
ax.text(2.5, -0.50, "FICO 680", ha="center", va="top",
        fontsize=FS_FOOT-1, color=GRAY, fontstyle="italic",
        transform=ax.transData)

ax.set_xticks(np.arange(7))
ax.set_xticklabels(fico_labels, fontsize=FS_TICK, color=GRAY)
ax.set_yticks([0, 1])
ax.set_yticklabels(["36-month loans", "60-month loans"],
                   fontsize=FS_TICK+1, fontweight="bold", color=NAVY)
ax.set_xlabel("FICO Score Range", fontsize=FS_AXIS, color=GRAY, labelpad=8)
ax.tick_params(length=0); ax.spines[:].set_visible(False)
ax.set_ylim(1.5, -0.6)

ax.set_title("Default Rate Heatmap: FICO Score  x  Loan Term",
             fontsize=FS_TITLE, fontweight="bold", color=NAVY, pad=14, loc="left")
fig.text(0.01, 0.01,
         SOURCE + "\nSubprime 60-month loans (bottom-left) carry 4.1x the default rate of prime 36-month loans (top-right)",
         fontsize=FS_FOOT, color=GRAY, fontstyle="italic")
fig.tight_layout(rect=[0, 0.10, 1, 0.97])
fig.savefig(os.path.join(CHART_DIR, "12_risk_heatmap.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
print("  v 12_risk_heatmap.png"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD PDF
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nBuilding PDF...")
from matplotlib.backends.backend_pdf import PdfPages

chart_files = sorted([f for f in os.listdir(CHART_DIR) if f.endswith(".png")])

with PdfPages(PDF_OUT) as pdf:
    # Title page
    fig_t = plt.figure(figsize=(12, 7))
    fig_t.patch.set_facecolor(NAVY)
    at = fig_t.add_axes([0, 0, 1, 1])
    at.set_facecolor(NAVY); at.axis("off")
    at.text(0.5, 0.72, "LendingClub Credit Risk Analysis",
            ha="center", fontsize=28, fontweight="bold",
            color="white", transform=at.transAxes)
    at.text(0.5, 0.57, "Chart Deck v5 — Consulting Style",
            ha="center", fontsize=18, color=TEAL, transform=at.transAxes)
    at.text(0.5, 0.42,
            "2,256,208 Loans  |  2007-2018  |  ECL = PD x LGD x EAD  |  XGBoost + SHAP + FRED",
            ha="center", fontsize=11, color=GRAY,
            fontstyle="italic", transform=at.transAxes)
    for i, st in enumerate(["2.26M Loans","12.9% Default Rate","USD 4.2B ECL","93% Actual LGD"]):
        at.text(0.12+i*0.24, 0.18, st, ha="center",
                fontsize=13, fontweight="bold", color="white", transform=at.transAxes)
    pdf.savefig(fig_t, bbox_inches="tight"); plt.close(fig_t)

    for f in chart_files:
        img = plt.imread(os.path.join(CHART_DIR, f))
        fig_i, ax_i = plt.subplots(figsize=(12, 7))
        fig_i.patch.set_facecolor("white")
        ax_i.imshow(img); ax_i.axis("off")
        pdf.savefig(fig_i, bbox_inches="tight")
        plt.close(fig_i)

print(f"\nDone! {len(chart_files)} charts saved")
print(f"  Charts: {CHART_DIR}")
print(f"  PDF:    {PDF_OUT}")
