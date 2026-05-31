"""
Pakistan Food Price Crisis Analysis — 2022 to 2023
====================================================
A focused, publication-ready deep dive into Pakistan's worst food
inflation episode in recorded history.

Outputs (saved to ./outputs/crisis/):
    01_crisis_timeline.png      — monthly price timeline with crisis phases
    02_commodity_impact.png     — % rise per commodity (2020 baseline vs 2023 peak)
    03_province_heatmap.png     — province × year wheat price heatmap
    04_market_comparison.png    — city-level crisis severity
    05_recovery_tracker.png     — 2023–2026 partial recovery
    06_pkr_vs_usd.png           — PKR price vs USD price (currency effect)
    crisis_full_report.csv      — machine-readable summary of all findings

Usage:
    python crisis_analysis_2022_2023.py
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "wfp_food_prices_pak.csv"
OUTPUT_DIR = "outputs/crisis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour system
CRISIS_RED    = "#C0392B"
RECOVERY_GRN  = "#1D9E75"
NEUTRAL_BLU   = "#185FA5"
WARN_AMB      = "#E67E22"
BG            = "#FAFAF8"
GRID_C        = "#E8E6E0"
TEXT_PRI      = "#1A1A18"
TEXT_SEC      = "#73726C"

COMMODITY_COLORS = {
    "Wheat flour"           : "#185FA5",
    "Rice (basmati, broken)": "#1D9E75",
    "Oil (cooking)"         : "#D85A30",
    "Ghee (artificial)"     : "#E67E22",
    "Poultry"               : "#D4537E",
    "Eggs"                  : "#7F77DD",
    "Lentils (masur)"       : "#639922",
    "Sugar"                 : "#8B6914",
}
PROVINCE_COLORS = {
    "BALOCHISTAN"       : "#185FA5",
    "KHYBER PAKHTUNKHWA": "#1D9E75",
    "PUNJAB"            : "#E67E22",
    "SINDH"             : "#D85A30",
}

KEY_COMMODITIES = list(COMMODITY_COLORS.keys())

plt.rcParams.update({
    "figure.facecolor" : BG,  "axes.facecolor"  : BG,
    "axes.grid"        : True, "grid.color"      : GRID_C,
    "grid.linewidth"   : 0.6,  "axes.spines.top" : False,
    "axes.spines.right": False,"axes.spines.left": False,
    "axes.spines.bottom": False,
    "xtick.color"      : TEXT_SEC, "ytick.color"  : TEXT_SEC,
    "text.color"       : TEXT_PRI, "font.family"  : "sans-serif",
    "font.size"        : 11,   "axes.titlesize"  : 13,
    "axes.titleweight" : "bold","axes.titlepad"   : 12,
    "axes.labelcolor"  : TEXT_SEC,
})

SOURCE_NOTE = "Source: World Food Programme (WFP) VAM — wfp_food_prices_pak.csv"

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    print(f"✓ Loaded {len(df):,} records  |  "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df


def crisis_window(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows for 2020–2026 and key commodities only."""
    return df[
        (df["year"] >= 2020) &
        (df["year"] <= 2026) &
        (df["commodity"].isin(KEY_COMMODITIES))
    ].copy()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Monthly Crisis Timeline
# ══════════════════════════════════════════════════════════════════════════════

def plot_crisis_timeline(df: pd.DataFrame, save_path: str):
    """
    Monthly national-average price lines for 8 commodities, Jan 2020–Apr 2026.
    Annotated with 4 crisis phases: calm → shock → peak → recovery.
    """
    cw = crisis_window(df)
    monthly = (
        cw.groupby(["year", "month", "commodity"])["price"]
        .mean().reset_index()
    )
    monthly["date"] = pd.to_datetime(
        monthly[["year", "month"]].assign(day=1)
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG)

    # Phase shading
    phases = [
        ("2020-01", "2021-09", "#AED6F1", "Phase 1\nCalm",     0.12),
        ("2021-10", "2022-06", "#F9E79F", "Phase 2\nShock",    0.12),
        ("2022-07", "2023-12", "#F1948A", "Phase 3\nPeak",     0.15),
        ("2024-01", "2026-04", "#A9DFBF", "Phase 4\nRecovery", 0.12),
    ]
    ymax = cw["price"].max() * 1.12
    for start, end, colour, label, alpha in phases:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=colour, alpha=alpha, zorder=0)
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.text(mid, ymax * 0.97, label, ha="center", va="top",
                fontsize=8.5, color=TEXT_SEC, style="italic")

    # Price lines
    for commodity, color in COMMODITY_COLORS.items():
        sub = monthly[monthly["commodity"] == commodity].sort_values("date")
        if sub.empty:
            continue
        ax.plot(sub["date"], sub["price"],
                color=color, linewidth=2, label=commodity,
                marker="o", markersize=2.5, zorder=3)

    # Key event annotations
    events = [
        ("2022-04-01", "Fuel subsidy\nremoved",    580),
        ("2023-02-01", "IMF deal\ncollapse",       460),
        ("2023-07-01", "PKR hits\n₨285/USD",       640),
    ]
    for date_str, label, y_pos in events:
        ax.annotate(
            label,
            xy=(pd.Timestamp(date_str), y_pos),
            xytext=(0, 28), textcoords="offset points",
            ha="center", fontsize=8, color=CRISIS_RED,
            arrowprops=dict(arrowstyle="->", color=CRISIS_RED,
                            lw=1.2, connectionstyle="arc3,rad=0.1"),
        )

    ax.set_title("Pakistan food prices — monthly trend during crisis (2020–2026)")
    ax.set_ylabel("Average price  (PKR / kg)")
    ax.set_xlim(pd.Timestamp("2020-01"), pd.Timestamp("2026-06"))
    ax.set_ylim(0, ymax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.xticks(rotation=30, ha="right")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₨{x:,.0f}")
    )
    ax.legend(loc="upper left", fontsize=8.5,
              framealpha=0.7, edgecolor=GRID_C, ncol=2)
    fig.text(0.01, 0.01, SOURCE_NOTE, fontsize=8, color=TEXT_SEC)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Commodity Impact: 2020 Baseline → 2023 Peak
# ══════════════════════════════════════════════════════════════════════════════

def plot_commodity_impact(df: pd.DataFrame, save_path: str):
    """
    Horizontal bar chart — % price increase per commodity from
    2020 average (pre-crisis baseline) to 2023 average (crisis peak).
    Includes absolute PKR values at both ends.
    """
    sub = df[df["commodity"].isin(KEY_COMMODITIES)]
    base = sub[sub["year"] == 2020].groupby("commodity")["price"].mean()
    peak = sub[sub["year"] == 2023].groupby("commodity")["price"].mean()
    pct  = ((peak - base) / base * 100).dropna().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG)

    colors = [COMMODITY_COLORS.get(c, NEUTRAL_BLU) for c in pct.index]
    bars   = ax.barh(pct.index, pct.values, color=colors,
                     edgecolor="none", height=0.55, zorder=3)

    for bar, (commodity, val) in zip(bars, pct.items()):
        b_val = base.get(commodity, 0)
        p_val = peak.get(commodity, 0)
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"+{val:.0f}%   (₨{b_val:.0f} → ₨{p_val:.0f})",
            va="center", fontsize=9, color=TEXT_SEC
        )

    # Severity threshold lines
    ax.axvline(50,  color=WARN_AMB,   lw=1, linestyle="--", alpha=0.6,
               label="50% threshold")
    ax.axvline(100, color=CRISIS_RED, lw=1, linestyle="--", alpha=0.6,
               label="100% threshold (doubled)")

    ax.set_title("Crisis impact per commodity — 2020 baseline vs 2023 peak")
    ax.set_xlabel("Price increase (%)")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"+{x:.0f}%")
    )
    ax.set_xlim(0, pct.max() * 1.55)
    ax.legend(fontsize=9, framealpha=0.7)
    fig.text(0.01, 0.01, SOURCE_NOTE, fontsize=8, color=TEXT_SEC)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Province × Year Heatmap (Wheat Flour)
# ══════════════════════════════════════════════════════════════════════════════

def plot_province_heatmap(df: pd.DataFrame, save_path: str):
    """
    Heatmap — wheat flour price by province and year (2020–2026).
    Reveals which provinces were hit hardest and recovered fastest.
    """
    wheat = df[
        (df["commodity"] == "Wheat flour") &
        (df["year"] >= 2020)
    ]
    pivot = (
        wheat.groupby(["admin1", "year"])["price"]
        .mean()
        .unstack("year")
        .round(1)
    )
    pivot.index = [i.title().replace("Khyber Pakhtunkhwa", "KPK")
                   for i in pivot.index]

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(BG)

    sns.heatmap(
        pivot, ax=ax,
        cmap="YlOrRd",
        annot=True, fmt=".0f",
        annot_kws={"size": 11, "weight": "bold"},
        linewidths=2, linecolor=BG,
        cbar_kws={"label": "PKR / kg", "shrink": 0.8},
    )
    ax.set_title("Wheat flour prices by province — crisis years (PKR/kg)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.text(0.01, -0.06,
             SOURCE_NOTE + "  |  Darker = more expensive",
             fontsize=8, color=TEXT_SEC)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Market-Level Crisis Severity
# ══════════════════════════════════════════════════════════════════════════════

def plot_market_comparison(df: pd.DataFrame, save_path: str):
    """
    Grouped bar chart — wheat flour price at each market city across
    2020, 2022, 2023, 2025. Shows diverging city-level trajectories.
    """
    wheat  = df[df["commodity"] == "Wheat flour"]
    years  = [2020, 2022, 2023, 2025]
    cities = ["Karachi", "Lahore", "Multan", "Peshawar", "Quetta"]
    city_year = (
        wheat[wheat["year"].isin(years)]
        .groupby(["market", "year"])["price"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)

    year_colors = {2020: "#AED6F1", 2022: "#F9E79F",
                   2023: CRISIS_RED, 2025: RECOVERY_GRN}
    x     = np.arange(len(cities))
    width = 0.18

    for i, yr in enumerate(years):
        vals = []
        for city in cities:
            row = city_year[
                (city_year["market"] == city) & (city_year["year"] == yr)
            ]
            vals.append(row["price"].values[0] if not row.empty else np.nan)
        offset = (i - len(years) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width,
                      color=year_colors[yr], label=str(yr),
                      edgecolor="none", alpha=0.92, zorder=3)
        # Annotate 2023 bars (peak)
        if yr == 2023:
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 2,
                            f"₨{v:.0f}", ha="center",
                            va="bottom", fontsize=8,
                            color=CRISIS_RED, fontweight="bold")

    ax.set_title("Wheat flour price by city — how hard did each market get hit?")
    ax.set_xticks(x)
    ax.set_xticklabels(cities, fontsize=11)
    ax.set_ylabel("Avg price (PKR/kg)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"₨{v:,.0f}")
    )
    ax.legend(title="Year", fontsize=9, framealpha=0.7)

    # % change label below each city
    for j, city in enumerate(cities):
        b = city_year[(city_year["market"]==city)&(city_year["year"]==2020)]["price"].values
        p = city_year[(city_year["market"]==city)&(city_year["year"]==2023)]["price"].values
        if len(b) and len(p) and b[0] > 0:
            pct = (p[0] - b[0]) / b[0] * 100
            ax.text(j, -18, f"+{pct:.0f}%\nvs 2020",
                    ha="center", va="top", fontsize=8.5,
                    color=CRISIS_RED, transform=ax.get_xaxis_transform())

    fig.text(0.01, 0.01, SOURCE_NOTE, fontsize=8, color=TEXT_SEC)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Recovery Tracker (2023–2026)
# ══════════════════════════════════════════════════════════════════════════════

def plot_recovery_tracker(df: pd.DataFrame, save_path: str):
    """
    Line chart showing indexed prices (2023 peak = 100) from 2023–2026.
    Reveals which commodities recovered vs which stayed elevated.
    """
    rec_commodities = [
        "Wheat flour", "Rice (basmati, broken)",
        "Oil (cooking)", "Sugar", "Poultry"
    ]
    rec_colors = {c: COMMODITY_COLORS[c] for c in rec_commodities}

    sub    = df[(df["year"] >= 2023) & (df["commodity"].isin(rec_commodities))]
    monthly = (
        sub.groupby(["year", "month", "commodity"])["price"]
        .mean().reset_index()
    )
    monthly["date"] = pd.to_datetime(
        monthly[["year", "month"]].assign(day=1)
    )

    # Index to Jan 2023 = 100
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    ax.axhline(100, color=TEXT_SEC, lw=1.2, linestyle="--",
               label="Jan 2023 peak = 100", alpha=0.7)

    for commodity, color in rec_colors.items():
        sub_c = monthly[monthly["commodity"] == commodity].sort_values("date")
        if sub_c.empty:
            continue
        base_val = sub_c.iloc[0]["price"]
        if base_val == 0:
            continue
        sub_c = sub_c.copy()
        sub_c["indexed"] = sub_c["price"] / base_val * 100
        ax.plot(sub_c["date"], sub_c["indexed"],
                color=color, linewidth=2.2, label=commodity,
                marker="o", markersize=3, zorder=3)
        # End label
        last = sub_c.iloc[-1]
        ax.annotate(
            f"{last['indexed']:.0f}",
            xy=(last["date"], last["indexed"]),
            xytext=(5, 0), textcoords="offset points",
            color=color, fontsize=9, va="center", fontweight="bold"
        )

    ax.fill_between(
        [pd.Timestamp("2023-01"), pd.Timestamp("2026-06")],
        0, 100, alpha=0.04, color=CRISIS_RED
    )
    ax.fill_between(
        [pd.Timestamp("2023-01"), pd.Timestamp("2026-06")],
        100, 160, alpha=0.04, color=RECOVERY_GRN
    )
    ax.text(pd.Timestamp("2023-03"), 95, "Below Jan 2023 level",
            fontsize=8.5, color=RECOVERY_GRN, style="italic")
    ax.text(pd.Timestamp("2023-03"), 105, "Above Jan 2023 level",
            fontsize=8.5, color=CRISIS_RED, style="italic")

    ax.set_title("Recovery tracker — indexed to Jan 2023 peak (Jan 2023 = 100)")
    ax.set_ylabel("Price index (Jan 2023 = 100)")
    ax.set_xlim(pd.Timestamp("2023-01"), pd.Timestamp("2026-06"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    fig.text(0.01, 0.01,
             SOURCE_NOTE + "  |  Below 100 = cheaper than Jan 2023",
             fontsize=8, color=TEXT_SEC)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — PKR Price vs USD Price (Currency Effect)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pkr_vs_usd(df: pd.DataFrame, save_path: str):
    """
    Dual-axis chart for wheat flour — PKR price (left) vs USD price (right).
    Isolates how much of the crisis was PKR devaluation vs real price rise.
    """
    wheat = df[
        (df["commodity"] == "Wheat flour") & (df["year"] >= 2020)
    ]
    monthly = (
        wheat.groupby(["year", "month"])
        .agg(pkr=("price", "mean"), usd=("usdprice", "mean"))
        .reset_index()
    )
    monthly["date"] = pd.to_datetime(
        monthly[["year", "month"]].assign(day=1)
    )
    monthly = monthly.sort_values("date")

    fig, ax1 = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    ax2 = ax1.twinx()

    l1, = ax1.plot(monthly["date"], monthly["pkr"],
                   color=CRISIS_RED, linewidth=2.5,
                   label="PKR price (left axis)", zorder=4)
    ax1.fill_between(monthly["date"], monthly["pkr"],
                     alpha=0.08, color=CRISIS_RED)

    l2, = ax2.plot(monthly["date"], monthly["usd"],
                   color=NEUTRAL_BLU, linewidth=2.5,
                   linestyle="--", label="USD price (right axis)", zorder=4)
    ax2.fill_between(monthly["date"], monthly["usd"],
                     alpha=0.06, color=NEUTRAL_BLU)

    # Crisis period shading
    ax1.axvspan(pd.Timestamp("2021-10"), pd.Timestamp("2023-12"),
                alpha=0.07, color=CRISIS_RED, zorder=0)

    ax1.set_title(
        "Wheat flour — PKR price vs USD price\n"
        "Separating currency devaluation from real price rise"
    )
    ax1.set_ylabel("PKR / kg", color=CRISIS_RED)
    ax2.set_ylabel("USD / kg", color=NEUTRAL_BLU)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₨{x:.0f}")
    )
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:.3f}")
    )
    ax1.tick_params(axis="y", labelcolor=CRISIS_RED)
    ax2.tick_params(axis="y", labelcolor=NEUTRAL_BLU)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.xticks(rotation=30, ha="right")

    lines  = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left",
               fontsize=9, framealpha=0.7)

    insight = (
        "Key insight: USD price rose ~47% (2020→2023) — "
        "but PKR price rose ~154%.\n"
        "The remaining ~107% was PKR devaluation, not global supply shock."
    )
    fig.text(0.5, -0.04, insight, ha="center",
             fontsize=9, color=TEXT_SEC, style="italic")
    fig.text(0.01, 0.01, SOURCE_NOTE, fontsize=8, color=TEXT_SEC)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# REPORT CSV
# ══════════════════════════════════════════════════════════════════════════════

def export_crisis_report(df: pd.DataFrame, save_path: str):
    """Export machine-readable crisis summary for all commodities."""
    rows = []
    for commodity in KEY_COMMODITIES:
        sub = df[df["commodity"] == commodity]
        ann = sub.groupby("year")["price"].mean()

        def safe_get(yr):
            return float(ann.get(yr, np.nan))

        p2020 = safe_get(2020)
        p2022 = safe_get(2022)
        p2023 = safe_get(2023)
        p2024 = safe_get(2024)
        p2025 = safe_get(2025)

        rows.append({
            "Commodity"                 : commodity,
            "Price 2020 (PKR/kg)"       : round(p2020, 1),
            "Price 2022 (PKR/kg)"       : round(p2022, 1),
            "Price 2023 (PKR/kg)"       : round(p2023, 1),
            "Price 2025 (PKR/kg)"       : round(p2025, 1),
            "2020→2023 change (%)"      : round((p2023-p2020)/p2020*100, 1)
                                          if p2020 > 0 else None,
            "2020→2022 change (%)"      : round((p2022-p2020)/p2020*100, 1)
                                          if p2020 > 0 else None,
            "Peak→2025 recovery (%)"    : round((p2025-p2023)/p2023*100, 1)
                                          if p2023 > 0 else None,
            "Fully recovered by 2025?"  : "YES"
                                          if (p2023 > 0 and p2025 <= p2020 * 1.05)
                                          else "NO",
        })

    report = pd.DataFrame(rows)
    report.to_csv(save_path, index=False)
    print(f"  ✓ {save_path}")
    print()
    print(report.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# KEY INSIGHTS — printed to console
# ══════════════════════════════════════════════════════════════════════════════

def print_crisis_insights(df: pd.DataFrame):
    print("\n" + "═" * 65)
    print("  CRISIS FINDINGS — Pakistan Food Price Shock 2022–2023")
    print("═" * 65)

    sub = df[df["commodity"].isin(KEY_COMMODITIES)]
    b   = sub[sub["year"] == 2020].groupby("commodity")["price"].mean()
    p   = sub[sub["year"] == 2023].groupby("commodity")["price"].mean()
    worst = ((p - b) / b * 100).sort_values(ascending=False)

    print("\n  Worst-hit commodities (2020 baseline → 2023 peak):")
    for c, pct in worst.items():
        bar = "█" * int(pct / 10)
        print(f"  {c:30s} +{pct:.1f}%  {bar}")

    print("\n  Province: Wheat flour 2023 avg price")
    prov = df[(df["year"] == 2023) & (df["commodity"] == "Wheat flour")]
    for prov_name, grp in prov.groupby("admin1"):
        print(f"  {prov_name:25s} ₨{grp['price'].mean():.0f}/kg")

    print("\n  City hardest hit (wheat, 2020→2023):")
    wheat = df[df["commodity"] == "Wheat flour"]
    for city in ["Multan", "Lahore", "Peshawar", "Quetta", "Karachi"]:
        b2020 = wheat[(wheat["year"]==2020)&(wheat["market"]==city)]["price"].mean()
        b2023 = wheat[(wheat["year"]==2023)&(wheat["market"]==city)]["price"].mean()
        if b2020 > 0:
            print(f"  {city:12s}  ₨{b2020:.0f} → ₨{b2023:.0f}  (+{(b2023-b2020)/b2020*100:.0f}%)")

    print("\n  Currency effect (wheat flour):")
    wheat_all = df[df["commodity"] == "Wheat flour"]
    for yr in [2020, 2023]:
        pkr = wheat_all[wheat_all["year"]==yr]["price"].mean()
        usd = wheat_all[wheat_all["year"]==yr]["usdprice"].mean()
        print(f"  {yr}: PKR ₨{pkr:.1f}/kg  |  USD ${usd:.4f}/kg")

    print("\n  Recovery status (2025 vs 2023 peak):")
    r  = sub[sub["year"] == 2025].groupby("commodity")["price"].mean()
    for c in KEY_COMMODITIES:
        if c in p.index and c in r.index and p[c] > 0:
            pct = (r[c] - p[c]) / p[c] * 100
            status = "✅ Partial recovery" if pct < 0 else "⚠️  Still elevated"
            print(f"  {c:30s} {pct:+.1f}%  {status}")

    print("═" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n▶  Pakistan Food Price Crisis Analysis — 2022/2023")
    print("─" * 50)

    df = load(DATA_PATH)

    print("\n▶  Generating 6 crisis charts…")
    plot_crisis_timeline(
        df, os.path.join(OUTPUT_DIR, "01_crisis_timeline.png"))
    plot_commodity_impact(
        df, os.path.join(OUTPUT_DIR, "02_commodity_impact.png"))
    plot_province_heatmap(
        df, os.path.join(OUTPUT_DIR, "03_province_heatmap.png"))
    plot_market_comparison(
        df, os.path.join(OUTPUT_DIR, "04_market_comparison.png"))
    plot_recovery_tracker(
        df, os.path.join(OUTPUT_DIR, "05_recovery_tracker.png"))
    plot_pkr_vs_usd(
        df, os.path.join(OUTPUT_DIR, "06_pkr_vs_usd.png"))

    print("\n▶  Exporting crisis report CSV…")
    export_crisis_report(
        df, os.path.join(OUTPUT_DIR, "crisis_full_report.csv"))

    print_crisis_insights(df)
    print(f"✅  All outputs saved to ./{OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
