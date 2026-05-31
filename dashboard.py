"""
Pakistan Food Price Dashboard (2004–2026)
==========================================
Interactive dashboard built with Streamlit + Plotly.
Data: World Food Programme (WFP) VAM Food Security Analysis

Run locally:
    pip install streamlit plotly pandas statsmodels
    streamlit run dashboard.py

Deploy free:
    https://share.streamlit.io
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan Food Price Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FAFAF8; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #E8E6E0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .metric-label { font-size: 12px; color: #73726C; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 600; color: #1A1A18; }
    .metric-delta-up   { font-size: 12px; color: #C0392B; }
    .metric-delta-down { font-size: 12px; color: #1D9E75; }
    .section-title {
        font-size: 16px; font-weight: 600;
        color: #1A1A18; margin: 1rem 0 0.5rem 0;
    }
    .insight-box {
        background: #EBF5FB;
        border-left: 4px solid #185FA5;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 13px;
        color: #1A1A18;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #FDEDEC;
        border-left: 4px solid #C0392B;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 13px;
        color: #1A1A18;
        margin: 0.5rem 0;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    "Wheat flour"           : "#185FA5",
    "Rice (basmati, broken)": "#1D9E75",
    "Oil (cooking)"         : "#D85A30",
    "Ghee (artificial)"     : "#E67E22",
    "Poultry"               : "#D4537E",
    "Eggs"                  : "#7F77DD",
    "Lentils (masur)"       : "#639922",
    "Sugar"                 : "#8B6914",
    "Rice (coarse)"         : "#2ECC71",
    "Wheat"                 : "#3498DB",
    "Milk"                  : "#9B59B6",
    "Salt"                  : "#95A5A6",
    "Beans(mash)"           : "#E74C3C",
    "Lentils (moong)"       : "#F39C12",
}
PROVINCE_COLORS = {
    "BALOCHISTAN"       : "#185FA5",
    "KHYBER PAKHTUNKHWA": "#1D9E75",
    "PUNJAB"            : "#E67E22",
    "SINDH"             : "#D85A30",
}
CITY_COLORS = {
    "Quetta"  : "#185FA5",
    "Peshawar": "#1D9E75",
    "Lahore"  : "#E67E22",
    "Multan"  : "#D85A30",
    "Karachi" : "#D4537E",
}

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("wfp_food_prices_pak.csv")
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["province_label"] = df["admin1"].str.title().str.replace(
        "Khyber Pakhtunkhwa", "KPK"
    )
    return df

@st.cache_data
def run_arima_forecast(series_values, steps=3):
    """Fit ARIMA(1,1,1) and return forecast + confidence intervals."""
    try:
        model  = ARIMA(series_values, order=(1, 1, 1))
        result = model.fit()
        fc     = result.get_forecast(steps=steps)
        mean   = np.array(fc.predicted_mean)
        ci     = np.array(fc.conf_int())
        return mean, ci
    except Exception:
        return None, None

df = load_data()

FOOD_COMMODITIES = [c for c in df["commodity"].unique()
                    if c not in ["Wage (non-qualified labour, non-agricultural)",
                                 "Fuel (diesel)", "Fuel (petrol-gasoline)"]]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_of_Pakistan.svg/240px-Flag_of_Pakistan.svg.png",
             width=60)
    st.markdown("## 🌾 Pakistan Food Prices")
    st.markdown("*WFP VAM Data · 2004–2026*")
    st.divider()

    st.markdown("### Filters")

    selected_commodities = st.multiselect(
        "Commodities",
        options=sorted(FOOD_COMMODITIES),
        default=["Wheat flour", "Rice (basmati, broken)", "Oil (cooking)", "Sugar"],
    )

    selected_provinces = st.multiselect(
        "Provinces",
        options=sorted(df["admin1"].unique().tolist()),
        default=df["admin1"].unique().tolist(),
    )

    year_range = st.slider(
        "Year range",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=(2015, int(df["year"].max())),
    )

    show_crisis = st.toggle("Highlight crisis period (2022–23)", value=True)
    show_usd    = st.toggle("Show USD prices", value=False)

    st.divider()
    st.markdown("### Forecast settings")
    forecast_commodity = st.selectbox(
        "Commodity to forecast",
        options=sorted(FOOD_COMMODITIES),
        index=sorted(FOOD_COMMODITIES).index("Wheat flour"),
    )
    forecast_years = st.slider("Forecast horizon (years)", 1, 5, 3)

    st.divider()
    st.caption("Data: [WFP VAM](https://data.humdata.org/dataset/wfp-food-prices-for-pakistan)")
    st.caption("Built with Python · Streamlit · Plotly · ARIMA")

# ── Filtered data ──────────────────────────────────────────────────────────────
mask = (
    df["commodity"].isin(selected_commodities) &
    df["admin1"].isin(selected_provinces) &
    df["year"].between(*year_range)
)
dff = df[mask].copy()

price_col = "usdprice" if show_usd else "price"
currency  = "USD" if show_usd else "PKR"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🌾 Pakistan Food Price Dashboard")
st.markdown(
    f"**World Food Programme (WFP) VAM Data · 2004–2026 · "
    f"{len(df):,} records · 4 provinces · 5 cities**"
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price Trends",
    "🔥 Crisis Analysis",
    "🗺️ Province & City",
    "🔗 Correlation",
    "🔮 Forecast (ARIMA)",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICE TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    if selected_commodities:
        kpi_cols = st.columns(min(len(selected_commodities), 4))
        for i, commodity in enumerate(selected_commodities[:4]):
            sub = df[df["commodity"] == commodity]
            latest_yr  = sub["year"].max()
            prev_yr    = latest_yr - 1
            latest_val = sub[sub["year"] == latest_yr][price_col].mean()
            prev_val   = sub[sub["year"] == prev_yr][price_col].mean()
            base_val   = sub[sub["year"] == 2020][price_col].mean()
            delta_yoy  = (latest_val - prev_val) / prev_val * 100 if prev_val > 0 else 0
            delta_2020 = (latest_val - base_val) / base_val * 100 if base_val > 0 else 0
            arrow      = "↑" if delta_yoy >= 0 else "↓"
            clr        = "metric-delta-up" if delta_yoy >= 0 else "metric-delta-down"
            sym        = "$" if show_usd else "₨"

            with kpi_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{commodity}</div>
                  <div class="metric-value">{sym}{latest_val:,.0f}</div>
                  <div class="{clr}">{arrow} {abs(delta_yoy):.1f}% vs {prev_yr}
                  &nbsp;·&nbsp; {'+' if delta_2020>=0 else ''}{delta_2020:.0f}% vs 2020</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("")

    # ── Line chart ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Annual average price trend</div>',
                unsafe_allow_html=True)

    annual = (
        dff.groupby(["year", "commodity"])[price_col]
        .mean().reset_index()
    )

    fig_trend = go.Figure()

    if show_crisis:
        fig_trend.add_vrect(
            x0=2021.5, x1=2023.5,
            fillcolor="#E74C3C", opacity=0.07,
            layer="below", line_width=0,
            annotation_text="Crisis", annotation_position="top left",
            annotation_font_color="#C0392B",
        )

    for commodity in selected_commodities:
        sub = annual[annual["commodity"] == commodity].sort_values("year")
        if sub.empty:
            continue
        color = COLORS.get(commodity, "#888888")
        fig_trend.add_trace(go.Scatter(
            x=sub["year"], y=sub[price_col],
            mode="lines+markers",
            name=commodity,
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
            hovertemplate=f"<b>{commodity}</b><br>Year: %{{x}}<br>Price: {currency} %{{y:,.1f}}/kg<extra></extra>",
        ))

    fig_trend.update_layout(
        height=420,
        plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(
            gridcolor="#E8E6E0",
            title=f"Avg price ({currency}/kg)",
            tickprefix="$" if show_usd else "₨",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── YoY heatmap ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Year-on-year inflation heatmap (%)</div>',
                unsafe_allow_html=True)

    all_annual = (
        df[df["commodity"].isin(FOOD_COMMODITIES)]
        .groupby(["year", "commodity"])["price"]
        .mean().reset_index()
    )
    all_annual["yoy"] = (
        all_annual.groupby("commodity")["price"].pct_change() * 100
    )
    pivot = (
        all_annual[all_annual["year"] >= 2010]
        .pivot(index="commodity", columns="year", values="yoy")
        .round(1)
    )

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        zmin=-25, zmax=90,
        text_auto=".0f",
        aspect="auto",
        labels=dict(color="YoY %"),
    )
    fig_heat.update_layout(
        height=380,
        plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_colorbar=dict(title="YoY %", ticksuffix="%"),
        xaxis_title="", yaxis_title="",
    )
    fig_heat.update_traces(textfont_size=10)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(
        '<div class="insight-box">💡 <b>2023 row</b> is Pakistan\'s worst food inflation year on record. '
        'Wheat flour rose <b>+85.8%</b> in a single year.</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CRISIS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Crisis impact: 2020 baseline → 2023 peak</div>',
                unsafe_allow_html=True)

    crisis_sub = df[df["commodity"].isin(FOOD_COMMODITIES)]
    base  = crisis_sub[crisis_sub["year"] == 2020].groupby("commodity")["price"].mean()
    peak  = crisis_sub[crisis_sub["year"] == 2023].groupby("commodity")["price"].mean()
    recov = crisis_sub[crisis_sub["year"] == 2025].groupby("commodity")["price"].mean()

    impact_df = pd.DataFrame({
        "commodity"  : base.index,
        "price_2020" : base.values,
        "price_2023" : peak.reindex(base.index).values,
        "price_2025" : recov.reindex(base.index).values,
    }).dropna()
    impact_df["pct_crisis"]   = ((impact_df["price_2023"] - impact_df["price_2020"])
                                  / impact_df["price_2020"] * 100).round(1)
    impact_df["pct_recovery"] = ((impact_df["price_2025"] - impact_df["price_2023"])
                                  / impact_df["price_2023"] * 100).round(1)
    impact_df = impact_df.sort_values("pct_crisis", ascending=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_impact = go.Figure(go.Bar(
            x=impact_df["pct_crisis"],
            y=impact_df["commodity"],
            orientation="h",
            marker_color=[COLORS.get(c, "#888") for c in impact_df["commodity"]],
            text=[f"+{v:.0f}%  (₨{b:.0f}→₨{p:.0f})"
                  for v, b, p in zip(impact_df["pct_crisis"],
                                     impact_df["price_2020"],
                                     impact_df["price_2023"])],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Crisis rise: +%{x:.1f}%<extra></extra>",
        ))
        fig_impact.add_vline(x=100, line_dash="dash",
                             line_color="#C0392B", opacity=0.5,
                             annotation_text="Doubled", annotation_position="top")
        fig_impact.update_layout(
            height=380, title="Crisis rise (2020→2023)",
            plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
            xaxis=dict(ticksuffix="%", gridcolor="#E8E6E0"),
            yaxis=dict(showgrid=False),
            margin=dict(l=10, r=140, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_impact, use_container_width=True)

    with col_b:
        recov_df = impact_df.sort_values("pct_recovery")
        colors_r = ["#1D9E75" if v < 0 else "#C0392B" for v in recov_df["pct_recovery"]]
        fig_recov = go.Figure(go.Bar(
            x=recov_df["pct_recovery"],
            y=recov_df["commodity"],
            orientation="h",
            marker_color=colors_r,
            text=[f"{v:+.1f}%" for v in recov_df["pct_recovery"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Recovery: %{x:.1f}%<extra></extra>",
        ))
        fig_recov.add_vline(x=0, line_color="#555", line_width=1)
        fig_recov.update_layout(
            height=380, title="Recovery by 2025 (vs 2023 peak)",
            plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
            xaxis=dict(ticksuffix="%", gridcolor="#E8E6E0"),
            yaxis=dict(showgrid=False),
            margin=dict(l=10, r=80, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_recov, use_container_width=True)

    st.markdown(
        '<div class="warning-box">⚠️ <b>No commodity has fully returned to 2020 prices.</b> '
        'Sugar is <b>34% MORE expensive</b> in 2025 than at the 2023 crisis peak. '
        'The "recovery" headline masks ongoing structural damage.</div>',
        unsafe_allow_html=True
    )

    # ── PKR vs USD dual axis ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">PKR vs USD price — separating currency devaluation from real price rise</div>',
                unsafe_allow_html=True)

    wheat_all = df[(df["commodity"] == "Wheat flour") & (df["year"] >= 2020)]
    monthly_w = (
        wheat_all.groupby(["year", "month"])
        .agg(pkr=("price", "mean"), usd=("usdprice", "mean"))
        .reset_index()
    )
    monthly_w["date"] = pd.to_datetime(monthly_w[["year", "month"]].assign(day=1))
    monthly_w = monthly_w.sort_values("date")

    fig_dual = go.Figure()
    fig_dual.add_trace(go.Scatter(
        x=monthly_w["date"], y=monthly_w["pkr"],
        name="PKR price (left)", fill="tozeroy",
        fillcolor="rgba(192,57,43,0.08)",
        line=dict(color="#C0392B", width=2.5),
        hovertemplate="PKR: ₨%{y:,.1f}<extra></extra>",
    ))
    fig_dual.add_trace(go.Scatter(
        x=monthly_w["date"], y=monthly_w["usd"],
        name="USD price (right)", yaxis="y2",
        line=dict(color="#185FA5", width=2.5, dash="dash"),
        hovertemplate="USD: $%{y:.4f}<extra></extra>",
    ))
    if show_crisis:
        fig_dual.add_vrect(x0="2021-10-01", x1="2023-12-01",
                           fillcolor="#E74C3C", opacity=0.06,
                           layer="below", line_width=0)
    fig_dual.update_layout(
        height=380,
        plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(title="PKR / kg", tickprefix="₨",
                   gridcolor="#E8E6E0",
                   title_font=dict(color="#C0392B")),
        yaxis2=dict(title="USD / kg", tickprefix="$",
                    overlaying="y", side="right",
                    title_font=dict(color="#185FA5"),
                    showgrid=False),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_dual, use_container_width=True)
    st.markdown(
        '<div class="insight-box">💡 PKR price rose <b>+154%</b> (2020→2023). '
        'USD price rose only <b>+47%</b>. '
        'The remaining <b>~107 percentage points</b> were PKR devaluation — '
        'a domestic policy cost, not a global supply shock.</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROVINCE & CITY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Province comparison — select commodity & year</div>',
                    unsafe_allow_html=True)
        prov_commodity = st.selectbox(
            "Commodity", options=sorted(FOOD_COMMODITIES),
            key="prov_commodity",
            index=sorted(FOOD_COMMODITIES).index("Wheat flour")
        )
        prov_year = st.slider("Year", 2004, 2026, 2023, key="prov_year")

        prov_data = df[
            (df["commodity"] == prov_commodity) &
            (df["year"] == prov_year)
        ].groupby("admin1")["price"].mean().reset_index()
        prov_data["label"] = prov_data["admin1"].str.title().str.replace(
            "Khyber Pakhtunkhwa", "KPK"
        )
        prov_data = prov_data.sort_values("price", ascending=False)

        fig_prov = go.Figure(go.Bar(
            x=prov_data["label"],
            y=prov_data["price"],
            marker_color=[PROVINCE_COLORS.get(p, "#888") for p in prov_data["admin1"]],
            text=[f"₨{v:.0f}" for v in prov_data["price"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>₨%{y:.1f}/kg<extra></extra>",
        ))
        fig_prov.update_layout(
            height=340,
            plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#E8E6E0", tickprefix="₨",
                       title=f"Avg price (PKR/kg)"),
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_prov, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">City-level price trend</div>',
                    unsafe_allow_html=True)
        city_commodity = st.selectbox(
            "Commodity", options=sorted(FOOD_COMMODITIES),
            key="city_commodity",
            index=sorted(FOOD_COMMODITIES).index("Wheat flour")
        )

        city_annual = (
            df[df["commodity"] == city_commodity]
            .groupby(["year", "market"])["price"]
            .mean().reset_index()
        )

        fig_city = go.Figure()
        for city in df["market"].unique():
            sub = city_annual[city_annual["market"] == city].sort_values("year")
            if sub.empty:
                continue
            fig_city.add_trace(go.Scatter(
                x=sub["year"], y=sub["price"],
                name=city,
                mode="lines+markers",
                line=dict(color=CITY_COLORS.get(city, "#888"), width=2),
                marker=dict(size=4),
                hovertemplate=f"<b>{city}</b><br>Year: %{{x}}<br>₨%{{y:.0f}}/kg<extra></extra>",
            ))
        if show_crisis:
            fig_city.add_vrect(x0=2021.5, x1=2023.5,
                               fillcolor="#E74C3C", opacity=0.06,
                               layer="below", line_width=0)
        fig_city.update_layout(
            height=340,
            plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#E8E6E0", tickprefix="₨",
                       title="Avg price (PKR/kg)"),
            legend=dict(orientation="h", y=1.08),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_city, use_container_width=True)

    # ── Province heatmap ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Province × Year heatmap — wheat flour (PKR/kg)</div>',
                unsafe_allow_html=True)

    heat_commodity = st.selectbox(
        "Commodity for heatmap", options=sorted(FOOD_COMMODITIES),
        key="heat_commodity",
        index=sorted(FOOD_COMMODITIES).index("Wheat flour")
    )
    heat_data = (
        df[(df["commodity"] == heat_commodity) & (df["year"] >= 2010)]
        .groupby(["province_label", "year"])["price"]
        .mean().reset_index()
        .pivot(index="province_label", columns="year", values="price")
        .round(1)
    )
    fig_pheat = px.imshow(
        heat_data,
        color_continuous_scale="YlOrRd",
        text_auto=".0f",
        aspect="auto",
        labels=dict(color="PKR/kg"),
    )
    fig_pheat.update_layout(
        height=260,
        plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="", yaxis_title="",
    )
    fig_pheat.update_traces(textfont_size=11)
    st.plotly_chart(fig_pheat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Commodity price correlation matrix</div>',
                unsafe_allow_html=True)

    corr_year_range = st.slider(
        "Correlation period", 2004, 2026, (2004, 2026), key="corr_yr"
    )
    corr_data = (
        df[df["commodity"].isin(FOOD_COMMODITIES) &
           df["year"].between(*corr_year_range)]
        .groupby(["year", "commodity"])["price"]
        .mean().reset_index()
        .pivot(index="year", columns="commodity", values="price")
    )
    corr_matrix = corr_data.corr().round(2)

    fig_corr = px.imshow(
        corr_matrix,
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
        labels=dict(color="Pearson r"),
    )
    fig_corr.update_layout(
        height=520,
        plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="", yaxis_title="",
    )
    fig_corr.update_traces(textfont_size=9)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown(
        '<div class="insight-box">💡 Most food commodities show <b>r > 0.85</b> correlation. '
        'This means a single shock (fuel price, exchange rate) lifts <i>all</i> prices simultaneously. '
        'Pakistan\'s food supply chain has no firewall between sectors.</div>',
        unsafe_allow_html=True
    )

    # Top correlations table
    st.markdown('<div class="section-title">Strongest correlations</div>',
                unsafe_allow_html=True)
    corr_pairs = []
    cols_list  = corr_matrix.columns.tolist()
    for i in range(len(cols_list)):
        for j in range(i+1, len(cols_list)):
            corr_pairs.append({
                "Commodity A": cols_list[i],
                "Commodity B": cols_list[j],
                "Pearson r"  : corr_matrix.iloc[i, j],
            })
    corr_pairs_df = (
        pd.DataFrame(corr_pairs)
        .sort_values("Pearson r", ascending=False)
        .head(8)
        .reset_index(drop=True)
    )
    corr_pairs_df["Pearson r"] = corr_pairs_df["Pearson r"].apply(lambda x: f"{x:.3f}")
    st.dataframe(corr_pairs_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ARIMA FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">ARIMA price forecast</div>',
                unsafe_allow_html=True)
    st.caption(
        "Model: ARIMA(1,1,1) fitted on annual average prices. "
        "Shaded area = 95% confidence interval. "
        "For research purposes — not financial advice."
    )

    # Prepare series
    fc_series = (
        df[df["commodity"] == forecast_commodity]
        .groupby("year")["price"]
        .mean()
        .sort_index()
    )

    mean_fc, ci_fc = run_arima_forecast(fc_series.values, steps=forecast_years)

    if mean_fc is not None:
        future_years = list(range(int(fc_series.index[-1]) + 1,
                                  int(fc_series.index[-1]) + forecast_years + 1))

        fig_fc = go.Figure()

        # Historical
        fig_fc.add_trace(go.Scatter(
            x=list(fc_series.index), y=list(fc_series.values),
            name="Historical", mode="lines+markers",
            line=dict(color=COLORS.get(forecast_commodity, "#185FA5"), width=2.5),
            marker=dict(size=5),
            hovertemplate="Year: %{x}<br>Price: ₨%{y:,.1f}<extra></extra>",
        ))

        # Confidence interval shading
        ci_lower = list(ci_fc[:, 0]) if isinstance(ci_fc, np.ndarray) else list(ci_fc.iloc[:, 0])
        ci_upper = list(ci_fc[:, 1]) if isinstance(ci_fc, np.ndarray) else list(ci_fc.iloc[:, 1])
        fig_fc.add_trace(go.Scatter(
            x=future_years + future_years[::-1],
            y=ci_upper + ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(24,95,165,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% confidence interval",
            hoverinfo="skip",
        ))

        # Forecast line
        fig_fc.add_trace(go.Scatter(
            x=future_years, y=list(mean_fc),
            name="Forecast", mode="lines+markers",
            line=dict(color=COLORS.get(forecast_commodity, "#185FA5"),
                      width=2.5, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
            hovertemplate="Year: %{x}<br>Forecast: ₨%{y:,.1f}<extra></extra>",
        ))

        # Divider line
        fig_fc.add_vline(
            x=fc_series.index[-1] + 0.5,
            line_dash="dot", line_color="#888", line_width=1,
            annotation_text="Forecast starts",
            annotation_position="top",
        )

        fig_fc.update_layout(
            height=420,
            plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(gridcolor="#E8E6E0", tickprefix="₨",
                       title="Avg price (PKR/kg)"),
            legend=dict(orientation="h", y=1.08),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast table
        fc_table = pd.DataFrame({
            "Year"                  : future_years,
            "Forecast (PKR/kg)"     : [f"₨{v:,.1f}" for v in mean_fc],
            "Lower bound (95% CI)"  : [f"₨{v:,.1f}" for v in (ci_fc[:, 0] if isinstance(ci_fc, np.ndarray) else ci_fc.iloc[:, 0])],
            "Upper bound (95% CI)"  : [f"₨{v:,.1f}" for v in (ci_fc[:, 1] if isinstance(ci_fc, np.ndarray) else ci_fc.iloc[:, 1])],
        })
        st.dataframe(fc_table, use_container_width=True, hide_index=True)

        last_actual = fc_series.values[-1]
        fc_change   = (mean_fc[0] - last_actual) / last_actual * 100
        st.markdown(
            f'<div class="insight-box">📊 ARIMA forecast for <b>{forecast_commodity}</b>: '
            f'₨{mean_fc[0]:,.1f}/kg in {future_years[0]} '
            f'({fc_change:+.1f}% vs current ₨{last_actual:,.1f}/kg). '
            f'Wide confidence interval reflects high volatility in Pakistan\'s food prices.</div>',
            unsafe_allow_html=True
        )
    else:
        st.warning("Not enough data points to run ARIMA forecast for this commodity.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "📊 Data: World Food Programme (WFP) VAM Food Security Analysis · "
    "wfp_food_prices_pak.csv · 13,900 records · 2004–2026  |  "
    "Built with Python · Streamlit · Plotly · statsmodels (ARIMA)  |  "
    "Open source · MIT License"
)
