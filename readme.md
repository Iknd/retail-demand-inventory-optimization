# 📦 Multi-SKU Retail Demand Forecasting & Inventory Optimization

**Live Interactive Dashboard:** [https://retail-demand-inventory-optimization-ybvtekw9bhrv9h33q8bvg3.streamlit.app/]

## 📌 Business Problem
Retail supply chains face a constant trade-off between **stockouts (lost revenue)** and **overstocking (working capital & holding costs)**. Standard moving-average heuristics fail on SKU-level retail demand due to high intermittency (**55.2% zero-sales days**), weekend spikes (**+53% volume surge**), and promotional price elasticity.

This project builds an end-to-end machine learning forecasting and replenishment simulation pipeline across **3,049 SKUs** (Walmart M5 dataset, Store `CA_1`) to quantify the exact dollar savings of ML-driven inventory policies over traditional baselines.

---

## 🚀 Key Results (28-Day Out-of-Sample Validation)
* **Forecast Accuracy:** Reduced 28-day cumulative SKU WMAPE from **32.00%** (28-Day Moving Average) and **89.82%** (Seasonal Naive) down to **29.01%** using **LightGBM (Tweedie objective)**.
* **Stockout Reduction:** Cut unfulfilled demand by **31.5%** (preventing **4,139 lost unit sales** in a single store over 28 days).
* **Service Level Improvement:** Increased overall fill rate from **90.20% to 93.29%** (+309 bps) at a 95% target cycle service level.
* **Net Cost Savings:** Reduced total inventory cost by **33.2%** (**$6,291 saved per store/month**, scaling to **~$75K/year per store**), trading an $89.66 increase in holding cost for **$6,381.00 in avoided stockout penalties**.

---

## 🛠️ Methodology & Tech Stack
1. **Data Hygiene & EDA (`01_data_exploration.ipynb`):** Filtered out 1.13M pre-launch SKU-days (19.1% noise reduction) using dynamic price availability; identified `FOODS` as 69.9% of unit volume.
2. **Leakage-Free Feature Engineering (`02_baseline_and_features.ipynb`):** Engineered calendar flags, price-momentum/discount ratios, and 28-day shifted lags/rolling statistics (`lag_28`, `lag_35`, `rolling_mean_7/28`, `rolling_std_28`) to strictly simulate a 28-day forward planning horizon.
3. **Zero-Inflated Demand Modeling (`03_lightgbm_modeling.ipynb`):** Trained a **LightGBM Regressor** with `tweedie` loss (`variance_power=1.1`) tailored for right-skewed, intermittent retail demand.
4. **Inventory Policy Simulation (`04_inventory_optimization.ipynb`):** Built a daily replenishment simulator incorporating supplier lead times (3–5 days), dynamic Reorder Points (ROP), and safety stock buffers to measure fill rate vs. holding/stockout cost trade-offs.
5. **Interactive App (`app/app.py`):** Deployed an interactive **Streamlit + Plotly** web application for SKU-level forecast drill-downs and policy cost comparisons.

**Stack:** Python (`pandas`, `numpy`, `LightGBM`, `scikit-learn`), Streamlit, Plotly.
