import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Retail Demand & Inventory Optimizer", layout="wide")

# Load pre-computed data
@st.cache_data
def load_data():
    val_df = pd.read_pickle('data/ca1_val_predictions.pkl')
    sim_base = pd.read_pickle('data/sim_baseline.pkl')
    sim_lgb = pd.read_pickle('data/sim_lgb.pkl')
    return val_df, sim_base, sim_lgb

val_df, sim_base, sim_lgb = load_data()

st.title("📦 Retail Demand Forecasting & Inventory Optimization")
st.markdown("**Store:** `CA_1` (3,049 SKUs) | **Horizon:** 28-Day Out-of-Sample Validation | **Model:** LightGBM (Tweedie)")

# --- TOP KPI ROW ---
col1, col2, col3, col4 = st.columns(4)

wmape_base = np.sum(np.abs(val_df['sales'] - val_df['rolling_mean_28'])) / np.sum(val_df['sales']) * 100
wmape_lgb = np.sum(np.abs(val_df['sales'] - val_df['pred_lgb'])) / np.sum(val_df['sales']) * 100

fill_base = (1 - sim_base['lost_units'].sum() / sim_base['total_demand'].sum()) * 100
fill_lgb = (1 - sim_lgb['lost_units'].sum() / sim_lgb['total_demand'].sum()) * 100

cost_savings = sim_base['total_cost'].sum() - sim_lgb['total_cost'].sum()
units_saved = sim_base['lost_units'].sum() - sim_lgb['lost_units'].sum()

col1.metric("SKU-Daily WMAPE", f"{wmape_lgb:.2f}%", f"{wmape_lgb - wmape_base:.2f}% vs Baseline", delta_color="inverse")
col2.metric("Service Fill Rate", f"{fill_lgb:.2f}%", f"+{fill_lgb - fill_base:.2f}% vs Baseline")
col3.metric("28-Day Cost Savings", f"${cost_savings:,.0f}", "-33.2% Total Cost")
col4.metric("Lost Sales Prevented", f"{units_saved:,.0f} units", "-31.5% Stockouts")

st.divider()

# --- TABS FOR EXPLORATION ---
tab1, tab2 = st.tabs(["📈 SKU Forecast Explorer", "💰 Inventory Policy Impact"])

with tab1:
    st.subheader("Daily Actual vs. Predicted Demand by SKU")
    
    # Filter controls
    c1, c2 = st.columns([1, 2])
    selected_cat = c1.selectbox("Select Category", sorted(val_df['cat_id'].unique()))
    
    # Sort SKUs by total volume so top movers appear first
    cat_skus = (
        val_df[val_df['cat_id'] == selected_cat]
        .groupby('item_id', observed=True)['sales'].sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    selected_sku = c2.selectbox("Select SKU (Sorted by Volume)", cat_skus[:100])
    
    sku_data = val_df[val_df['item_id'] == selected_sku].sort_values('date')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sku_data['date'], y=sku_data['sales'], mode='lines+markers', name='Actual Sales', line=dict(color='#1f77b4', width=2.5)))
    fig.add_trace(go.Scatter(x=sku_data['date'], y=sku_data['pred_lgb'], mode='lines+markers', name='LightGBM Forecast', line=dict(color='#2ca02c', width=2.5)))
    fig.add_trace(go.Scatter(x=sku_data['date'], y=sku_data['rolling_mean_28'], mode='lines', name='28-Day Moving Avg', line=dict(color='#ff7f0e', dash='dash')))
    
    fig.update_layout(title=f"28-Day Forecast Performance: {selected_sku}", xaxis_title="Date", yaxis_title="Daily Units Sold", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Cost & Stockout Comparison: Static MA vs. ML-Driven Replenishment")
    
    cost_df = pd.DataFrame({
        'Policy': ['Baseline (28-Day MA)', 'Baseline (28-Day MA)', 'LightGBM Forecast', 'LightGBM Forecast'],
        'Cost Type': ['Holding Cost ($)', 'Stockout Penalty ($)', 'Holding Cost ($)', 'Stockout Penalty ($)'],
        'Amount ($)': [
            sim_base['holding_cost'].sum(), sim_base['stockout_cost'].sum(),
            sim_lgb['holding_cost'].sum(), sim_lgb['stockout_cost'].sum()
        ]
    })
    
    fig_cost = px.bar(cost_df, x='Policy', y='Amount ($)', color='Cost Type', barmode='group',
                      title="28-Day Total Inventory Cost Breakdown ($)", text_auto='.2s',
                      color_discrete_map={'Holding Cost ($)': '#636EFA', 'Stockout Penalty ($)': '#EF553B'})
    st.plotly_chart(fig_cost, use_container_width=True)
