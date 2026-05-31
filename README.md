# 🌾 Pakistan Food Price Analysis (2004–2026)

> **20 years of WFP food price data — analysed, visualised, and open-sourced.**  
> A data science deep dive into Pakistan's food inflation, provincial disparities, and the 2022–2023 economic crisis.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Data: WFP](https://img.shields.io/badge/Data-WFP%20VAM-009EDB?style=flat-square)](https://data.humdata.org/dataset/wfp-food-prices-for-pakistan)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Records](https://img.shields.io/badge/Records-13%2C900-orange?style=flat-square)]()
[![Coverage](https://img.shields.io/badge/Coverage-2004--2026-red?style=flat-square)]()

---

## 📌 Project Overview

Pakistan has experienced some of the most volatile food prices in South Asia over the past two decades. This project uses the **World Food Programme (WFP)** dataset to:

- Track long-run price trends for 7 key food commodities
- Quantify the 2022–2023 economic crisis impact on household food costs
- Reveal inter-provincial price disparities across Balochistan, KPK, Punjab, and Sindh
- Identify which commodities move together through correlation analysis

**Why this matters:** Food insecurity is a critical strategic and humanitarian issue. Data-driven analysis helps policymakers, NGOs, and researchers understand where prices hurt the most — and when.

---

## 📊 Key Findings

| Commodity | Price 2004 | Price 2026 | Total Change | Worst Year |
|-----------|-----------|-----------|-------------|-----------|
| Wheat flour | ₨13/kg | ₨115/kg | **+770%** | 2023 (+85.8%) |
| Basmati rice | ₨20/kg | ₨239/kg | **+1,123%** | 2023 (+78.9%) |
| Lentils (masur) | — | ₨274/kg | — | 2022 (+57.1%) |
| Cooking oil | ₨214/kg | ₨604/kg | **+183%** | 2022 (+56.4%) |
| Eggs | — | ₨275/kg | — | 2023 (+46.8%) |
| Sugar | — | ₨156/kg | — | 2023 (+46.5%) |
| Poultry | — | ₨400/kg | — | 2023 (+43.6%) |

> 💡 **2023 was the worst year for food inflation in Pakistan's recorded history** — wheat flour alone rose 85.8% in a single year driven by currency collapse, fuel price shock, and import restrictions.

---

## 📁 Project Structure

```
pakistan-food-price-analysis/
│
├── 📄 pakistan_food_price_analysis.py   # Main analysis script
├── 📄 wfp_food_prices_pak.csv           # Raw dataset (WFP VAM)
├── 📄 README.md                         # This file
│
└── outputs/
    ├── trend_analysis.png               # 20-year price trend chart
    ├── yoy_inflation.png                # Year-on-year inflation heatmap
    ├── crisis_analysis.png              # 4-panel 2020–2026 crisis deep dive
    ├── province_comparison.png          # Inter-provincial comparison
    ├── correlation_matrix.png           # Commodity correlation heatmap
    └── summary_stats.csv               # Clean statistics table
```

---

## 📈 Visualisations

### 1. 20-Year Price Trend
![Price Trends](outputs/trend_analysis.png)
*Annual average prices (PKR/kg) for 7 key commodities. Crisis period (2022–23) shaded.*

---

### 2. Year-on-Year Inflation Heatmap
![YoY Inflation](outputs/yoy_inflation.png)
*Red = prices rising fast. Green = prices falling. 2022–23 row tells the whole story.*

---

### 3. Crisis Analysis (2020–2026)
![Crisis Analysis](outputs/crisis_analysis.png)
*4-panel deep dive: absolute prices, YoY inflation, cumulative change from 2020, and peak inflation per commodity.*

---

### 4. Province-wise Comparison
![Province Comparison](outputs/province_comparison.png)
*Punjab consistently has the lowest staple prices due to agricultural surplus. Balochistan pays the most — ~17% above Punjab for wheat.*

---

### 5. Commodity Correlation Matrix
![Correlation Matrix](outputs/correlation_matrix.png)
*Most commodities are highly correlated (r > 0.85), suggesting shared inflation drivers — fuel costs and import dependency.*

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading, cleaning, aggregation |
| `numpy` | Numerical computation (CAGR, pct change) |
| `matplotlib` | All chart generation |
| `seaborn` | Heatmaps (YoY inflation, correlation) |

---

## ▶️ How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/pakistan-food-price-analysis.git
cd pakistan-food-price-analysis
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn
```

**3. Run the analysis**
```bash
python pakistan_food_price_analysis.py
```

All 5 charts + summary CSV will be saved to `./outputs/`

---

## 📦 Dataset

**Source:** [World Food Programme — VAM Food Security Analysis](https://data.humdata.org/dataset/wfp-food-prices-for-pakistan)

| Field | Details |
|-------|---------|
| Records | 13,900 |
| Date range | January 2004 – April 2026 |
| Provinces | Balochistan, Khyber Pakhtunkhwa, Punjab, Sindh |
| Markets | Quetta, Peshawar, Lahore, Multan, Karachi |
| Commodities | 17 (cereals, pulses, oil, meat, dairy, non-food) |
| Currency | PKR (Pakistani Rupee) per kg |

**Citation:**
```
World Food Programme (WFP). Pakistan Food Prices Dataset.
VAM Food Security Analysis. Retrieved from:
https://data.humdata.org/dataset/wfp-food-prices-for-pakistan
```

---

## 💡 Insights for Policy & Research

1. **Food security risk is highest in Balochistan** — highest prices, remotest markets
2. **Cereals and oil prices are tightly correlated** — fuel/import shocks ripple instantly into kitchen costs
3. **The 2022–23 crisis was structural, not seasonal** — wheat prices took 2 years to partially correct
4. **Punjab's price advantage** makes it the most food-secure province — agricultural policy matters
5. **Basmati rice has never stopped rising** — currently at all-time high of ₨239/kg as of 2026

---

## 🗺️ Future Work

- [ ] Add USD price analysis (purchasing power over time)
- [ ] Time series forecasting (ARIMA / Prophet) for price predictions
- [ ] Seasonal decomposition — is there a harvest-cycle dip?
- [ ] Compare with CPI / inflation index data from SBP
- [ ] Interactive dashboard (Streamlit or Plotly Dash)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The dataset is provided by WFP under their open data policy. Please cite WFP when using this data in publications.

---

## 🤝 Connect

If you found this useful, let's connect!

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME)

*Star ⭐ the repo if this project helped you!*
