"""
fred_stress_calibration.py
===========================
Week 5 — Pull FRED API data, calibrate stress test multipliers
from historical recession data, and update Chart 07.

Main scenario mapping:
  Base               = current portfolio
  Rate Hike          = mild stress (manual overlay, NOT full COVID multiplier)
  Recession          = Dot-com calibrated
  Severe Recession   = GFC calibrated
  COVID Shock        = appendix only

Outputs:
  fred_data_raw.csv
  fred_recession_analysis.csv
  fred_scenario_table.csv
  07_stress_test_calibrated.png
  fred_diagnostic.txt

Run:
  pip install fredapi pandas numpy matplotlib
  python fred_stress_calibration.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ── CONFIG ──────────────────────────────────────────────────────────────
FRED_API_KEY = "a4b2b8a0cc3cad35e7d9b6aa5a4146ce"

OUT_DIR       = r"C:\Users\Tuấn Minh\OneDrive\CREDIT"
OUT_CHART     = os.path.join(OUT_DIR, "charts", "07_stress_test_calibrated.png")
OUT_RAW_CSV   = os.path.join(OUT_DIR, "fred_data_raw.csv")
OUT_ANAL_CSV  = os.path.join(OUT_DIR, "fred_recession_analysis.csv")
OUT_SCEN_CSV  = os.path.join(OUT_DIR, "fred_scenario_table.csv")
OUT_DIAG      = os.path.join(OUT_DIR, "fred_diagnostic.txt")

# Baseline ECL from your project
ECL_BASE_M = 4_216  # $M

# Mild "Rate Hike" overlay — intentionally NOT using full COVID shock
RATE_HIKE_PD_MULT  = 1.18
RATE_HIKE_LGD_MULT = 1.03

# LGD sensitivity to GDP downturn
GAMMA_LGD = 0.30

# Palette
NAVY  = "#0D1B2A"
TEAL  = "#0F8B8D"
RED   = "#E63946"
AMBER = "#F4A261"
GOLD  = "#E9C46A"
GREY  = "#8D99AE"
CREAM = "#F8F6F1"
WHITE = "#FFFFFF"

# ── FRED SERIES ─────────────────────────────────────────────────────────
# Verified series:
#   DRCLACBS         = Delinquency Rate on Consumer Loans, All Commercial Banks
#   A191RL1Q225SBEA  = Real Gross Domestic Product, Percent Change from Preceding Period
FRED_SERIES = {
    "UNRATE":          "Unemployment Rate (%)",
    "GDPC1":           "Real GDP Level (Billions, chained 2017$)",
    "A191RL1Q225SBEA": "Real GDP Growth Rate (% change from preceding period)",
    "DRCLACBS":        "Delinquency Rate on Consumer Loans, All Commercial Banks (%)",
    "DRSFRMACBS":      "Delinquency Rate on Residential Mortgages (%)",
    "USREC":           "NBER Recession Indicator (1=recession)",
    "FEDFUNDS":        "Federal Funds Rate (%)",
    "CPIAUCSL":        "Consumer Price Index",
}

RECESSIONS = {
    "Dot-com (2001)": ("2001-03-01", "2001-11-01"),
    "GFC (2008-09)":  ("2007-12-01", "2009-06-01"),
    "COVID (2020)":   ("2020-02-01", "2020-04-01"),
}


# ── STEP 1: PULL DATA ───────────────────────────────────────────────────
def pull_fred_data(api_key: str) -> pd.DataFrame:
    try:
        from fredapi import Fred
    except ImportError:
        print("❌ fredapi chưa cài. Chạy: pip install fredapi")
        sys.exit(1)

    if not api_key or api_key == "REPLACE_WITH_NEW_KEY":
        print("❌ Chưa điền FRED API key mới.")
        sys.exit(1)

    fred = Fred(api_key=api_key)
    frames = {}

    print("\n📡 Pulling FRED data...")
    for series_id, desc in FRED_SERIES.items():
        try:
            s = fred.get_series(
                series_id,
                observation_start="1995-01-01",
                observation_end="2024-12-31",
            )
            s = pd.Series(s, name=series_id).dropna()
            frames[series_id] = s
            print(f"  ✓ {series_id:20s} — {len(s):,} observations  ({desc})")
        except Exception as e:
            print(f"  ✗ {series_id}: {e}")

    if not frames:
        raise RuntimeError("Không pull được series nào từ FRED.")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Align all series to monthly-start frequency
    df = df.resample("MS").last().ffill()

    print(f"\n  Combined DataFrame: {df.shape[0]} months × {df.shape[1]} series")
    print(f"  Date range: {df.index.min().date()} → {df.index.max().date()}")
    return df


# ── STEP 2: PD MULTIPLIER ───────────────────────────────────────────────
def compute_pd_multiplier(df: pd.DataFrame, recession: tuple) -> dict:
    """
    PD multiplier methods:
      1. Unemployment-based: UNRATE_peak / UNRATE_pre
      2. Delinquency-based:  DRCLACBS_peak / DRCLACBS_pre

    For traditional recessions:
      final = average(unemployment, delinquency)

    For COVID:
      final = delinquency-only override
      because unemployment spike was distorted by lockdowns and fiscal support.
    """
    start, end = pd.Timestamp(recession[0]), pd.Timestamp(recession[1])

    pre_start = start - pd.DateOffset(months=12)
    pre_data  = df.loc[pre_start:start]
    dur_data  = df.loc[start:end]
    post_end  = end + pd.DateOffset(months=6)
    full_data = df.loc[start:post_end]

    results = {}

    # Method 1 — unemployment
    unrate_pre = pre_data["UNRATE"].mean()
    unrate_peak = full_data["UNRATE"].max()
    pd_mult_unemp = unrate_peak / unrate_pre if pd.notna(unrate_pre) and unrate_pre > 0 else 1.0

    results["unrate_pre"] = round(float(unrate_pre), 2)
    results["unrate_peak"] = round(float(unrate_peak), 2)
    results["pd_mult_unemp"] = round(float(pd_mult_unemp), 4)

    pd_components = [pd_mult_unemp]

    # Method 2 — consumer delinquency
    if "DRCLACBS" in df.columns and df["DRCLACBS"].notna().sum() > 10:
        delq_pre = pre_data["DRCLACBS"].mean()
        delq_peak = full_data["DRCLACBS"].max()
        pd_mult_delq = delq_peak / delq_pre if pd.notna(delq_pre) and delq_pre > 0 else 1.0

        results["delq_pre"] = round(float(delq_pre), 2)
        results["delq_peak"] = round(float(delq_peak), 2)
        results["pd_mult_delq"] = round(float(pd_mult_delq), 4)

        if start >= pd.Timestamp("2020-01-01"):
            results["pd_mult_final"] = round(float(pd_mult_delq), 4)
            results["pd_method_note"] = "COVID override: delinquency-only"
        else:
            results["pd_mult_final"] = round(float(np.mean([pd_mult_unemp, pd_mult_delq])), 4)
            results["pd_method_note"] = "Average of unemployment and delinquency"
    else:
        results["pd_mult_final"] = round(float(pd_mult_unemp), 4)
        results["pd_method_note"] = "Unemployment-only fallback"

    # Optional context
    if len(dur_data) > 0:
        results["unrate_end"] = round(float(dur_data["UNRATE"].iloc[-1]), 2)

    return results


# ── STEP 3: LGD MULTIPLIER ──────────────────────────────────────────────
def compute_lgd_multiplier(df: pd.DataFrame, recession: tuple) -> dict:
    """
    LGD multiplier from GDP trough:

      LGD_mult = 1 + GAMMA_LGD × abs(GDP_growth_trough / 100)

    Preferred source:
      A191RL1Q225SBEA (GDP growth rate)

    Fallback:
      derive decline from GDPC1 level
    """
    start, end = pd.Timestamp(recession[0]), pd.Timestamp(recession[1])

    pre_start = start - pd.DateOffset(months=12)
    post_end = end + pd.DateOffset(months=6)

    pre_data = df.loc[pre_start:start]
    full_data = df.loc[start:post_end]

    results = {}

    if "A191RL1Q225SBEA" in df.columns and df["A191RL1Q225SBEA"].notna().sum() > 0:
        gdp_growth_trough = full_data["A191RL1Q225SBEA"].min()
        lgd_mult = 1 + GAMMA_LGD * abs(gdp_growth_trough / 100)
        results["lgd_method_note"] = "GDP growth rate series"
    else:
        gdp_pre = pre_data["GDPC1"].mean()
        gdp_trough = full_data["GDPC1"].min()
        gdp_decline_pct = (gdp_trough - gdp_pre) / gdp_pre * 100
        gdp_growth_trough = gdp_decline_pct
        lgd_mult = 1 + GAMMA_LGD * abs(gdp_decline_pct / 100)
        results["lgd_method_note"] = "Fallback from GDP level decline"

    results["gdp_trough_growth"] = round(float(gdp_growth_trough), 2)
    results["lgd_mult"] = round(float(lgd_mult), 4)
    return results


# ── STEP 4: BUILD SCENARIOS ──────────────────────────────────────────────
def build_scenarios(recession_results: dict) -> pd.DataFrame:
    """
    Main deck scenarios:
      Base             = current
      Rate Hike        = mild manual stress
      Recession        = Dot-com calibrated
      Severe Recession = GFC calibrated

    Appendix:
      COVID Shock      = optional only
    """
    scenarios = [
        {
            "scenario": "Base",
            "label": "Base\n(Current)",
            "pd_mult": 1.00,
            "lgd_mult": 1.00,
            "source": "Current portfolio",
            "color": TEAL,
            "scenario_group": "main",
        },
        {
            "scenario": "Rate Hike",
            "label": "Rate Hike\n(Mild stress)",
            "pd_mult": RATE_HIKE_PD_MULT,
            "lgd_mult": RATE_HIKE_LGD_MULT,
            "source": "Manual mild-stress overlay",
            "color": AMBER,
            "scenario_group": "main",
        },
    ]

    if "Dot-com (2001)" in recession_results:
        r = recession_results["Dot-com (2001)"]
        scenarios.append({
            "scenario": "Recession",
            "label": "Recession\n(Dot-com)",
            "pd_mult": r["pd_mult_final"],
            "lgd_mult": r["lgd_mult"],
            "source": "Dot-com (2001)",
            "color": GOLD,
            "scenario_group": "main",
        })

    if "GFC (2008-09)" in recession_results:
        r = recession_results["GFC (2008-09)"]
        scenarios.append({
            "scenario": "Severe Recession",
            "label": "Severe\n(GFC)",
            "pd_mult": r["pd_mult_final"],
            "lgd_mult": r["lgd_mult"],
            "source": "GFC (2008-09)",
            "color": RED,
            "scenario_group": "main",
        })

    if "COVID (2020)" in recession_results:
        r = recession_results["COVID (2020)"]
        scenarios.append({
            "scenario": "COVID Shock",
            "label": "COVID Shock\n(Appendix)",
            "pd_mult": r["pd_mult_final"],
            "lgd_mult": r["lgd_mult"],
            "source": "COVID (2020)",
            "color": GREY,
            "scenario_group": "appendix",
        })

    df = pd.DataFrame(scenarios)
    df["ecl_stressed_M"] = (ECL_BASE_M * df["pd_mult"] * df["lgd_mult"]).round(0).astype(int)
    df["ecl_pct_change"] = ((df["ecl_stressed_M"] / ECL_BASE_M - 1) * 100).round(1)
    return df


# ── STEP 5: PLOT CHART ───────────────────────────────────────────────────
def plot_stress_test(scenarios_df: pd.DataFrame, recession_results: dict, fred_df: pd.DataFrame = None):
    main_df = scenarios_df[scenarios_df["scenario_group"] == "main"].copy()

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(CREAM)

    # Panel 1: main scenarios
    ax1 = fig.add_axes([0.05, 0.42, 0.60, 0.52])
    ax1.set_facecolor(CREAM)

    x = np.arange(len(main_df))
    bars = ax1.bar(
        x,
        main_df["ecl_stressed_M"],
        color=main_df["color"].tolist(),
        edgecolor=WHITE,
        linewidth=0.8,
        zorder=3,
        width=0.62,
    )

    for bar, row in zip(bars, main_df.itertuples()):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            row.ecl_stressed_M + 60,
            f"${row.ecl_stressed_M:,}M",
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
            color=NAVY,
        )

        pct_label = "Baseline" if row.ecl_pct_change == 0 else f"+{row.ecl_pct_change:.0f}%"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            row.ecl_stressed_M / 2,
            pct_label,
            ha="center",
            va="center",
            fontsize=12,
            color=WHITE,
            fontweight="bold",
        )

        if row.ecl_pct_change != 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                -350,
                f"PDx{row.pd_mult:.2f}\nLGDx{row.lgd_mult:.2f}",
                ha="center",
                va="top",
                fontsize=8.5,
                color=GREY,
                fontstyle="italic",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(main_df["label"], fontsize=10, color=NAVY)
    ax1.set_ylabel("Total ECL ($M)", fontsize=11, color=NAVY)
    ax1.set_title(
        "Stress Test — Evidence-Based ECL Scenarios\n"
        "(Main deck excludes full COVID shock from mild-stress mapping)",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1000:.1f}B"))
    ax1.set_ylim(-500, max(main_df["ecl_stressed_M"]) * 1.18)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color(GREY)
    ax1.spines["bottom"].set_color(GREY)
    ax1.tick_params(colors=NAVY, labelsize=10)
    ax1.grid(axis="y", color=GREY, alpha=0.2, linewidth=0.6, linestyle="--")
    ax1.set_axisbelow(True)

    # Panel 2: UNRATE history
    ax2 = fig.add_axes([0.05, 0.06, 0.58, 0.30])
    ax2.set_facecolor(CREAM)

    if fred_df is not None and "UNRATE" in fred_df.columns:
        unrate = fred_df["UNRATE"].dropna()
        ax2.plot(unrate.index, unrate.values, color=NAVY, linewidth=1.8, zorder=4)

        rec_colors = {
            "Dot-com (2001)": GOLD,
            "GFC (2008-09)": RED,
            "COVID (2020)": AMBER,
        }

        for rec_name, (rstart, rend) in RECESSIONS.items():
            c = rec_colors.get(rec_name, GREY)
            ax2.axvspan(pd.Timestamp(rstart), pd.Timestamp(rend), color=c, alpha=0.25, zorder=2)

            rec_df = fred_df.loc[rstart:rend, "UNRATE"].dropna()
            if len(rec_df) > 0:
                peak_date = rec_df.idxmax()
                peak_val = rec_df.max()
                ax2.annotate(
                    f"{peak_val:.1f}%",
                    xy=(peak_date, peak_val),
                    xytext=(peak_date, peak_val + 0.7),
                    fontsize=9,
                    color=c,
                    fontweight="bold",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.2),
                )

        ax2.axhline(4.0, color=GREY, linestyle="--", linewidth=1, alpha=0.5)

    else:
        ax2.text(
            0.5,
            0.5,
            "UNRATE chart will appear after FRED data is pulled",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=GREY,
            fontstyle="italic",
        )

    ax2.set_title("Unemployment Rate (UNRATE) — Historical Context", fontsize=10.5, fontweight="bold", color=NAVY, pad=6)
    ax2.set_ylabel("UNRATE (%)", fontsize=9.5, color=NAVY)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(colors=NAVY, labelsize=9)
    ax2.grid(axis="y", color=GREY, alpha=0.2, linewidth=0.5, linestyle="--")
    ax2.set_axisbelow(True)

    # Panel 3: source table
    ax3 = fig.add_axes([0.68, 0.06, 0.30, 0.88])
    ax3.set_facecolor(NAVY)
    ax3.axis("off")

    ax3.text(
        0.5, 0.975, "CALIBRATION SOURCE",
        transform=ax3.transAxes,
        ha="center", va="top",
        fontsize=11, fontweight="bold", color=GOLD,
        fontfamily="monospace"
    )

    table_rows = [
        ("", "PDx", "LGDx", "ECL"),
        ("Base", "1.00", "1.00", f"${ECL_BASE_M/1000:.2f}B"),
        ("Rate", f"{RATE_HIKE_PD_MULT:.2f}", f"{RATE_HIKE_LGD_MULT:.2f}", f"${(ECL_BASE_M*RATE_HIKE_PD_MULT*RATE_HIKE_LGD_MULT)/1000:.2f}B"),
    ]

    if "Dot-com (2001)" in recession_results:
        r = recession_results["Dot-com (2001)"]
        table_rows.append(("2001", f"{r['pd_mult_final']:.2f}", f"{r['lgd_mult']:.2f}", f"${(ECL_BASE_M*r['pd_mult_final']*r['lgd_mult'])/1000:.2f}B"))

    if "GFC (2008-09)" in recession_results:
        r = recession_results["GFC (2008-09)"]
        table_rows.append(("GFC", f"{r['pd_mult_final']:.2f}", f"{r['lgd_mult']:.2f}", f"${(ECL_BASE_M*r['pd_mult_final']*r['lgd_mult'])/1000:.2f}B"))

    if "COVID (2020)" in recession_results:
        r = recession_results["COVID (2020)"]
        table_rows.append(("COVID*", f"{r['pd_mult_final']:.2f}", f"{r['lgd_mult']:.2f}", f"${(ECL_BASE_M*r['pd_mult_final']*r['lgd_mult'])/1000:.2f}B"))

    row_colors = [GREY, TEAL, AMBER, GOLD, RED, GREY]
    for i, (row, rc) in enumerate(zip(table_rows, row_colors[:len(table_rows)])):
        y = 0.88 - i * 0.09
        ax3.add_patch(
            plt.Rectangle(
                (0.02, y - 0.04), 0.96, 0.07,
                transform=ax3.transAxes,
                facecolor=rc if i == 0 else "#1B3A5C",
                alpha=0.4,
                zorder=1,
            )
        )
        for val, xpos in zip(row, [0.08, 0.35, 0.60, 0.82]):
            ax3.text(
                xpos, y, val,
                transform=ax3.transAxes,
                fontsize=10,
                color=WHITE if i > 0 else NAVY,
                fontweight="bold" if i == 0 else "normal",
                va="center",
            )

    y_formula = 0.40
    ax3.text(0.5, y_formula, "FORMULA", transform=ax3.transAxes, ha="center", fontsize=9, fontweight="bold", color=GOLD)

    formula_lines = [
        "ECL_stressed =",
        "  ECL_base",
        "  x PD_mult",
        "  x LGD_mult",
        "",
        "PD_mult source:",
        "  UNRATE + DRCLACBS",
        "  (COVID override:",
        "   delinquency only)",
        "",
        "LGD_mult source:",
        "  A191RL1Q225SBEA",
        f"  gamma = {GAMMA_LGD}",
        "",
        "* COVID kept in appendix",
    ]

    for k, line in enumerate(formula_lines):
        ax3.text(
            0.06, y_formula - 0.05 - k * 0.042,
            line,
            transform=ax3.transAxes,
            fontsize=8.5,
            color="#CADCFC",
            fontfamily="monospace",
        )

    ax3.text(
        0.5, 0.02,
        "Source: FRED via API",
        transform=ax3.transAxes,
        ha="center", va="bottom",
        fontsize=8, color=GREY, fontstyle="italic"
    )

    os.makedirs(os.path.dirname(OUT_CHART), exist_ok=True)
    fig.savefig(OUT_CHART, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n✅ Chart saved: {OUT_CHART}")
    plt.close()


# ── STEP 6: SAVE OUTPUTS ────────────────────────────────────────────────
def save_outputs(fred_df, scenarios_df, recession_results):
    fred_df.to_csv(OUT_RAW_CSV)
    print(f"✅ Raw FRED data: {OUT_RAW_CSV}")

    rows = []
    for rec_name, r in recession_results.items():
        rows.append({"recession": rec_name, **r})
    pd.DataFrame(rows).to_csv(OUT_ANAL_CSV, index=False)
    print(f"✅ Recession analysis: {OUT_ANAL_CSV}")

    scenarios_df.to_csv(OUT_SCEN_CSV, index=False)
    print(f"✅ Scenario table: {OUT_SCEN_CSV}")

    diag = f"""
=======================================================================
FRED STRESS CALIBRATION — DIAGNOSTIC REPORT
=======================================================================
ECL Base: ${ECL_BASE_M:,}M

MAIN SCENARIO DESIGN
============================================================
Base             = current portfolio
Rate Hike        = mild manual stress, not full COVID multiplier
Recession        = Dot-com calibrated
Severe Recession = GFC calibrated
COVID Shock      = appendix only

RECESSION ANALYSIS
============================================================
"""
    for rec_name, r in recession_results.items():
        diag += f"\n{rec_name}\n"
        for k, v in r.items():
            diag += f"  {k:25s}: {v}\n"

    diag += f"\nSCENARIO TABLE\n{'='*60}\n"
    diag += scenarios_df[["scenario", "pd_mult", "lgd_mult", "ecl_stressed_M", "ecl_pct_change", "source", "scenario_group"]].to_string(index=False)

    diag += f"""

METHODOLOGY NOTES
============================================================
PD Multiplier
  Traditional recessions:
    average(UNRATE_peak / UNRATE_pre, DRCLACBS_peak / DRCLACBS_pre)

  COVID:
    delinquency-only override

LGD Multiplier
  1 + {GAMMA_LGD} × |GDP_growth_trough / 100|

RATE HIKE SCENARIO
  PD_mult  = {RATE_HIKE_PD_MULT}
  LGD_mult = {RATE_HIKE_LGD_MULT}

SERIES PULLED
============================================================
"""
    for s, desc in FRED_SERIES.items():
        diag += f"  {s:20s}: {desc}\n"

    with open(OUT_DIAG, "w", encoding="utf-8") as f:
        f.write(diag)
    print(f"✅ Diagnostic: {OUT_DIAG}")


# ── MAIN ────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("FRED STRESS CALIBRATION — Week 5")
    print("=" * 70)

    fred_df = pull_fred_data(FRED_API_KEY)

    print("\n📐 Computing recession multipliers...")
    recession_results = {}

    for rec_name, period in RECESSIONS.items():
        print(f"\n  {rec_name} ({period[0]} → {period[1]})")
        pd_result = compute_pd_multiplier(fred_df, period)
        lgd_result = compute_lgd_multiplier(fred_df, period)

        combined = {**pd_result, **lgd_result}
        recession_results[rec_name] = combined

        print(f"    UNRATE: {combined.get('unrate_pre','N/A')}% → {combined.get('unrate_peak','N/A')}%")
        if "delq_pre" in combined:
            print(f"    Delinquency: {combined.get('delq_pre','N/A')}% → {combined.get('delq_peak','N/A')}%")
        print(f"    GDP trough growth: {combined.get('gdp_trough_growth','N/A')}%")
        print("    ─────────────────────────────────")
        print(f"    PD  multiplier: × {combined['pd_mult_final']:.4f}")
        print(f"    LGD multiplier: × {combined['lgd_mult']:.4f}")
        print(f"    PD note:        {combined.get('pd_method_note', 'N/A')}")
        print(f"    LGD note:       {combined.get('lgd_method_note', 'N/A')}")
        ecl_s = ECL_BASE_M * combined["pd_mult_final"] * combined["lgd_mult"]
        print(f"    ECL stressed:   ${ecl_s:,.0f}M  ({(ecl_s/ECL_BASE_M-1)*100:+.1f}%)")

    print("\n📊 Building scenario table...")
    scenarios_df = build_scenarios(recession_results)
    print(scenarios_df[["scenario", "pd_mult", "lgd_mult", "ecl_stressed_M", "ecl_pct_change", "scenario_group"]].to_string(index=False))

    print("\n🎨 Plotting updated Chart 07...")
    plot_stress_test(scenarios_df, recession_results, fred_df)

    print("\n📋 COMPARISON: Main scenarios vs old assumptions")
    print(f"{'Scenario':<20} {'Old ECL':>10} {'New ECL':>10} {'Difference':>12}")
    print("-" * 58)

    old = {"Base": 4216, "Rate Hike": 4963, "Recession": 5700, "Severe Recession": 6898}
    for _, row in scenarios_df[scenarios_df["scenario_group"] == "main"].iterrows():
        old_match = old.get(row["scenario"])
        if old_match is not None:
            diff = row["ecl_stressed_M"] - old_match
            print(f"{row['scenario']:<20} ${old_match:>8,}M ${row['ecl_stressed_M']:>8,}M  {diff:>+8,}M")

    print("\n📎 Appendix scenario")
    appendix_df = scenarios_df[scenarios_df["scenario_group"] == "appendix"]
    if not appendix_df.empty:
        print(appendix_df[["scenario", "pd_mult", "lgd_mult", "ecl_stressed_M", "ecl_pct_change"]].to_string(index=False))

    save_outputs(fred_df, scenarios_df, recession_results)

    print("\n✅ DONE")
    print(f"   Chart : {OUT_CHART}")
    print(f"   Data  : {OUT_RAW_CSV}")


if __name__ == "__main__":
    main()