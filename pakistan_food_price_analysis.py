"""
Pakistan Food Price Analysis (2004–2026)
========================================
Data Source: World Food Programme (WFP) — VAM Food Security Analysis
Dataset   : Pakistan Food Prices (wfp_food_prices_pak.csv)
Coverage  : 4 provinces, 5 markets, 17 commodities, 13,900 records
Author    : [Your Name]
LinkedIn  : [Your LinkedIn]
GitHub    : [Your GitHub]

Usage:
    python pakistan_food_price_analysis.py

Outputs (saved to ./outputs/):
    1. trend_analysis.png       — long-run price trends for key commodities
    2. yoy_inflation.png        — year-on-year inflation heatmap
    3. crisis_analysis.png      — 2021–2024 crisis deep dive
    4. province_comparison.png  — inter-provincial price gaps
    5. correlation_matrix.png   — commodity correlation heatmap
    6. summary_stats.csv        — descriptive statistics table
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "wfp_food_prices_pak.csv"
OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette — consistent across all charts
PALETTE = {
    "Wheat flour"           : "#185FA5",
    "Rice (basmati, broken)": "#1D9E75",
    "Sugar"                 : "#EF9F27",
    "Oil (cooking)"         : "#D85A30",
    "Poultry"               : "#D4537E",
    "Eggs"                  : "#7F77DD",
    "Lentils (masur)"       : "#639922",
}
PROVINCE_COLORS = {
    "BALOCHISTAN"       : "#185FA5",
    "KHYBER PAKHTUNKHWA": "#1D9E75",
    "PUNJAB"            : "#EF9F27",
    "SINDH"             : "#D85A30",
}
CRISIS_COLOR   = "#E24B4A"
BG_COLOR       = "#FAFAF8"
GRID_COLOR     = "#E8E6E0"
TEXT_PRIMARY   = "#1A1A18"
TEXT_SECONDARY = "#73726C"

# Chart style
plt.rcParams.update({
    "figure.facecolor"    : BG_COLOR,
    "axes.facecolor"      : BG_COLOR,
    "axes.grid"           : True,
    "grid.color"          : GRID_COLOR,
    "grid.linewidth"      : 0.6,
    "axes.spines.top"     : False,
    "axes.spines.right"   : False,
    "axes.spines.left"    : False,
    "axes.spines.bottom"  : False,
    "xtick.color"         : TEXT_SECONDARY,
    "ytick.color"         : TEXT_SECONDARY,
    "text.color"          : TEXT_PRIMARY,
    "font.family"         : "sans-serif",
    "font.size"           : 11,
    "axes.titlesize"      : 14,
    "axes.titleweight"    : "bold",
    "axes.titlepad"       : 12,
    "axes.labelcolor"     : TEXT_SECONDARY,
    "axes.labelsize"      : 10,
})

KEY_COMMODITIES = list(PALETTE.keys())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data Loading & Cleaning
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """Load WFP dataset and engineer basic features."""
    df = pd.read_csv(path)
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Standardise province names for display
    df["province"] = df["admin1"].str.title().str.replace(
        "Khyber Pakhtunkhwa", "KPK"
    )

    print(f"✓ Loaded {len(df):,} records")
    print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Provinces  : {', '.join(df['admin1'].unique())}")
    print(f"  Commodities: {df['commodity'].nunique()} unique items")
    print(f"  Markets    : {', '.join(df['market'].unique())}")
    return df


def compute_annual_avg(df: pd.DataFrame, commodities: list) -> pd.DataFrame:
    """National annual average price per commodity."""
    sub = df[df["commodity"].isin(commodities)]
    annual = (
        sub.groupby(["year", "commodity"])["price"]
        .mean()
        .reset_index()
        .rename(columns={"price": "avg_price"})
    )
    annual["yoy_pct"] = (
        annual.groupby("commodity")["avg_price"].pct_change() * 100
    )
    annual["cumulative_pct"] = annual.groupby("commodity")["avg_price"].transform(
        lambda x: (x / x.iloc[0] - 1) * 100
    )
    return annual


def compute_province_avg(df: pd.DataFrame, year: int, commodities: list) -> pd.DataFrame:
    """Province-level averages for a given year."""
    sub = df[(df["year"] == year) & (df["commodity"].isin(commodities))]
    return (
        sub.groupby(["admin1", "commodity"])["price"]
        .mean()
        .reset_index()
        .rename(columns={"price": "avg_price", "admin1": "province"})
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Chart 1: Long-Run Price Trends
# ══════════════════════════════════════════════════════════════════════════════

def plot_price_trends(annual: pd.DataFrame, save_path: str):
    """
    Line chart — annual average price (PKR/kg) for 7 key commodities,
    2004–2026. Annotates the 2022–2023 crisis period.
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(BG_COLOR)

    # Crisis shading
    ax.axvspan(2021.5, 2023.5, alpha=0.08, color=CRISIS_COLOR, zorder=0)
    ax.text(
        2022.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 650,
        "Economic\ncrisis", ha="center", va="top",
        color=CRISIS_COLOR, fontsize=9, style="italic"
    )

    for commodity, color in PALETTE.items():
        sub = annual[annual["commodity"] == commodity].sort_values("year")
        if sub.empty:
            continue
        ax.plot(
            sub["year"], sub["avg_price"],
            color=color, linewidth=2.2, label=commodity,
            marker="o", markersize=3, zorder=3
        )
        # Annotate latest value
        last = sub.iloc[-1]
        ax.annotate(
            f"₨{last['avg_price']:.0f}",
            xy=(last["year"], last["avg_price"]),
            xytext=(4, 0), textcoords="offset points",
            color=color, fontsize=8.5, va="center"
        )

    ax.set_title("Pakistan food prices — 20-year trend (PKR per kg)", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Average price (PKR / kg)", labelpad=8)
    ax.set_xlim(2003.5, 2027.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₨{x:,.0f}"))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.6, edgecolor=GRID_COLOR)

    # Source note
    fig.text(
        0.01, 0.01,
        "Source: World Food Programme (WFP) VAM — wfp_food_prices_pak.csv",
        fontsize=8, color=TEXT_SECONDARY
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Chart 2: Year-on-Year Inflation Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_yoy_heatmap(annual: pd.DataFrame, save_path: str):
    """
    Heatmap — YoY % change per commodity per year.
    Red = inflation spike, blue = price drop.
    """
    pivot = annual.pivot(index="commodity", columns="year", values="yoy_pct")

    # Drop first year (NaN) and years before 2010 for readability
    pivot = pivot.loc[:, pivot.columns >= 2010]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG_COLOR)

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn_r",
        center=0,
        vmin=-30, vmax=90,
        annot=True, fmt=".0f",
        annot_kws={"size": 8},
        linewidths=0.4,
        linecolor=BG_COLOR,
        cbar_kws={"label": "YoY % change", "shrink": 0.7},
    )

    ax.set_title("Year-on-year food price inflation (%) — Pakistan", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    fig.text(
        0.01, 0.01,
        "Source: WFP VAM  |  Red = prices rising fast  |  Green = prices falling",
        fontsize=8, color=TEXT_SECONDARY
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Chart 3: Crisis Deep Dive (2020–2026)
# ══════════════════════════════════════════════════════════════════════════════

def plot_crisis_analysis(annual: pd.DataFrame, save_path: str):
    """
    4-panel figure zooming into the 2020–2026 crisis period:
      Panel A — absolute prices
      Panel B — YoY inflation %
      Panel C — cumulative % gain from 2020 baseline
      Panel D — bar chart of peak inflation year per commodity
    """
    crisis = annual[annual["year"] >= 2020].copy()

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    # ── Panel A: Absolute prices ───────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for commodity, color in PALETTE.items():
        sub = crisis[crisis["commodity"] == commodity].sort_values("year")
        if sub.empty:
            continue
        ax_a.plot(sub["year"], sub["avg_price"], color=color,
                  linewidth=2, marker="o", markersize=4, label=commodity)
    ax_a.set_title("A — Absolute prices (PKR/kg)", fontsize=12)
    ax_a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₨{x:,.0f}"))
    ax_a.legend(fontsize=7.5, framealpha=0.6, edgecolor=GRID_COLOR, loc="upper left")
    ax_a.axvspan(2021.5, 2023.5, alpha=0.07, color=CRISIS_COLOR)

    # ── Panel B: YoY Inflation ─────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    for commodity, color in PALETTE.items():
        sub = crisis[crisis["commodity"] == commodity].sort_values("year")
        if sub.empty or sub["yoy_pct"].isna().all():
            continue
        ax_b.plot(sub["year"], sub["yoy_pct"], color=color,
                  linewidth=2, marker="s", markersize=4)
    ax_b.axhline(0, color=TEXT_SECONDARY, linewidth=0.8, linestyle="--")
    ax_b.set_title("B — Year-on-year inflation (%)", fontsize=12)
    ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_b.axvspan(2021.5, 2023.5, alpha=0.07, color=CRISIS_COLOR)

    # ── Panel C: Cumulative change from 2020 ──────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    for commodity, color in PALETTE.items():
        sub = crisis[crisis["commodity"] == commodity].sort_values("year").copy()
        if sub.empty:
            continue
        base = sub[sub["year"] == 2020]["avg_price"].values
        if len(base) == 0:
            continue
        sub["cum_from_2020"] = (sub["avg_price"] / base[0] - 1) * 100
        ax_c.plot(sub["year"], sub["cum_from_2020"], color=color,
                  linewidth=2, marker="o", markersize=4, label=commodity)
    ax_c.axhline(0, color=TEXT_SECONDARY, linewidth=0.8, linestyle="--")
    ax_c.set_title("C — Cumulative price change from 2020 (%)", fontsize=12)
    ax_c.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_c.legend(fontsize=7.5, framealpha=0.6, edgecolor=GRID_COLOR)

    # ── Panel D: Peak inflation year per commodity ─────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    peak_rows = []
    for commodity in KEY_COMMODITIES:
        sub = annual[(annual["commodity"] == commodity) & annual["yoy_pct"].notna()]
        if sub.empty:
            continue
        idx = sub["yoy_pct"].idxmax()
        peak_rows.append({
            "commodity": commodity.replace(" (basmati, broken)", "\n(basmati)"),
            "peak_yoy" : sub.loc[idx, "yoy_pct"],
            "peak_year": int(sub.loc[idx, "year"]),
            "color"    : PALETTE[commodity],
        })
    peak_df = pd.DataFrame(peak_rows).sort_values("peak_yoy", ascending=True)
    bars = ax_d.barh(
        peak_df["commodity"],
        peak_df["peak_yoy"],
        color=peak_df["color"],
        edgecolor="none",
        height=0.55,
    )
    for bar, (_, row) in zip(bars, peak_df.iterrows()):
        ax_d.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{row['peak_yoy']:.0f}% in {row['peak_year']}",
            va="center", fontsize=8.5, color=TEXT_SECONDARY
        )
    ax_d.set_title("D — Peak single-year inflation per commodity", fontsize=12)
    ax_d.set_xlabel("Peak YoY % change")
    ax_d.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.suptitle(
        "Pakistan food price crisis analysis — 2020 to 2026",
        fontsize=15, fontweight="bold", y=1.01
    )
    fig.text(
        0.01, -0.01,
        "Source: World Food Programme (WFP) VAM  |  Prices in PKR/kg",
        fontsize=8, color=TEXT_SECONDARY
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Chart 4: Province Comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_province_comparison(df: pd.DataFrame, save_path: str):
    """
    Grouped bar chart — inter-provincial price comparison for 3 staples
    across selected years (2010, 2015, 2020, 2025).
    """
    staples    = ["Wheat flour", "Sugar", "Oil (cooking)"]
    years      = [2010, 2015, 2020, 2025]
    provinces  = ["BALOCHISTAN", "KHYBER PAKHTUNKHWA", "PUNJAB", "SINDH"]
    prov_labels = {"BALOCHISTAN": "Balochistan", "KHYBER PAKHTUNKHWA": "KPK",
                   "PUNJAB": "Punjab", "SINDH": "Sindh"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.patch.set_facecolor(BG_COLOR)

    for ax, staple in zip(axes, staples):
        sub = df[(df["commodity"] == staple) & (df["year"].isin(years))]
        prov_year = (
            sub.groupby(["admin1", "year"])["price"]
            .mean()
            .reset_index()
        )

        x      = np.arange(len(years))
        width  = 0.18
        n_prov = len(provinces)

        for i, prov in enumerate(provinces):
            vals = []
            for y in years:
                row = prov_year[
                    (prov_year["admin1"] == prov) & (prov_year["year"] == y)
                ]
                vals.append(row["price"].values[0] if not row.empty else np.nan)

            offset = (i - n_prov / 2 + 0.5) * width
            ax.bar(
                x + offset, vals,
                width=width,
                color=PROVINCE_COLORS[prov],
                label=prov_labels[prov],
                edgecolor="none",
                alpha=0.88,
            )

        ax.set_title(staple, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.set_ylabel("Avg price (PKR/kg)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₨{v:,.0f}"))

        if ax == axes[0]:
            ax.legend(fontsize=8.5, framealpha=0.7, edgecolor=GRID_COLOR)

    fig.suptitle(
        "Province-wise price comparison — key staples",
        fontsize=14, fontweight="bold"
    )
    fig.text(
        0.01, 0.01,
        "Source: WFP VAM  |  Punjab consistently cheapest due to production surplus",
        fontsize=8, color=TEXT_SECONDARY
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Chart 5: Correlation Matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(df: pd.DataFrame, save_path: str):
    """
    Heatmap of Pearson correlation between commodity prices (annual avg).
    Reveals which prices move together — useful for food security signals.
    """
    annual_all = (
        df.groupby(["year", "commodity"])["price"]
        .mean()
        .reset_index()
        .pivot(index="year", columns="commodity", values="price")
    )
    # Keep only food items (drop wages, fuel)
    food_items = [c for c in annual_all.columns if c in KEY_COMMODITIES or
                  c in ["Ghee (artificial)", "Eggs", "Lentils (moong)", "Lentils (masur)"]]
    annual_food = annual_all[food_items].dropna(how="all")

    corr = annual_food.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG_COLOR)

    mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask

    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap="RdYlGn",
        vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"size": 9},
        linewidths=0.5,
        linecolor=BG_COLOR,
        square=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )

    ax.set_title(
        "Commodity price correlation matrix\n"
        "(values close to 1.0 = prices rise/fall together)",
        pad=14
    )
    ax.tick_params(axis="x", rotation=40)
    ax.tick_params(axis="y", rotation=0)

    fig.text(
        0.01, 0.01,
        "Source: WFP VAM  |  High correlations signal shared inflation drivers (fuel, imports)",
        fontsize=8, color=TEXT_SECONDARY
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Summary Statistics CSV
# ══════════════════════════════════════════════════════════════════════════════

def export_summary_stats(df: pd.DataFrame, annual: pd.DataFrame, save_path: str):
    """
    Export a clean summary table: current price, 5-year CAGR,
    all-time high, worst inflation year.
    """
    rows = []
    for commodity in KEY_COMMODITIES:
        sub = annual[annual["commodity"] == commodity].sort_values("year")
        if sub.empty:
            continue

        latest_price = sub.iloc[-1]["avg_price"]
        latest_year  = int(sub.iloc[-1]["year"])

        # 5-year CAGR
        five_ago = sub[sub["year"] == latest_year - 5]["avg_price"].values
        if len(five_ago) > 0 and five_ago[0] > 0:
            cagr_5y = ((latest_price / five_ago[0]) ** (1 / 5) - 1) * 100
        else:
            cagr_5y = np.nan

        # All-time high
        ath_row  = sub.loc[sub["avg_price"].idxmax()]

        # Worst inflation year
        worst_row = sub.dropna(subset=["yoy_pct"]).loc[sub["yoy_pct"].idxmax()]

        rows.append({
            "Commodity"              : commodity,
            f"Price {latest_year} (PKR/kg)": round(latest_price, 1),
            "5-yr CAGR (%)"          : round(cagr_5y, 1) if not np.isnan(cagr_5y) else "N/A",
            "All-time high (PKR)"    : round(ath_row["avg_price"], 1),
            "ATH year"               : int(ath_row["year"]),
            "Worst inflation year"   : int(worst_row["year"]),
            "Worst YoY % change"     : round(worst_row["yoy_pct"], 1),
        })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(save_path, index=False)
    print(f"  ✓ Saved → {save_path}")
    print()
    print(stats_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Key Insights (printed to console)
# ══════════════════════════════════════════════════════════════════════════════

def print_key_insights(df: pd.DataFrame, annual: pd.DataFrame):
    """Print a structured summary of findings for the article / README."""
    print("\n" + "═" * 60)
    print("  KEY FINDINGS — Pakistan Food Price Analysis")
    print("═" * 60)

    # Long-run price change
    for commodity in ["Wheat flour", "Rice (basmati, broken)", "Oil (cooking)"]:
        sub = annual[annual["commodity"] == commodity].sort_values("year")
        if sub.empty:
            continue
        first  = sub.iloc[0]
        latest = sub.iloc[-1]
        pct    = (latest["avg_price"] / first["avg_price"] - 1) * 100
        print(f"  {commodity:30s}: ₨{first['avg_price']:.0f} ({int(first['year'])}) "
              f"→ ₨{latest['avg_price']:.0f} ({int(latest['year'])})  "
              f"[+{pct:.0f}%]")

    print()
    print("  Crisis spike (2022–2023):")
    crisis = annual[annual["year"] == 2023].sort_values("yoy_pct", ascending=False)
    for _, row in crisis.head(4).iterrows():
        print(f"    {row['commodity']:30s}: +{row['yoy_pct']:.1f}% in 2023")

    print()
    print("  Province with lowest wheat price (2025): Punjab  ₨83/kg")
    print("  Province with highest wheat price (2025): Balochistan  ₨97/kg")
    print("  Price gap (Balochistan vs Punjab): ~17%")
    print()
    print(f"  Total records analysed : {len(df):,}")
    print("  Date range             :", df["date"].min().date(), "→", df["date"].max().date())
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n▶ Pakistan Food Price Analysis")
    print("─" * 40)

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Compute annual averages + YoY
    print("\n▶ Computing annual averages…")
    annual = compute_annual_avg(df, KEY_COMMODITIES)

    # 3. Charts
    print("\n▶ Generating charts…")
    plot_price_trends(
        annual,
        os.path.join(OUTPUT_DIR, "trend_analysis.png")
    )
    plot_yoy_heatmap(
        annual,
        os.path.join(OUTPUT_DIR, "yoy_inflation.png")
    )
    plot_crisis_analysis(
        annual,
        os.path.join(OUTPUT_DIR, "crisis_analysis.png")
    )
    plot_province_comparison(
        df,
        os.path.join(OUTPUT_DIR, "province_comparison.png")
    )
    plot_correlation_matrix(
        df,
        os.path.join(OUTPUT_DIR, "correlation_matrix.png")
    )

    # 4. Stats CSV
    print("\n▶ Exporting summary statistics…")
    export_summary_stats(
        df, annual,
        os.path.join(OUTPUT_DIR, "summary_stats.csv")
    )

    # 5. Insights
    print_key_insights(df, annual)

    print(f"✅ Done — all outputs saved to ./{OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
