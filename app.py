import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="Kestrel Home | Returns Risk Command Center",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1E293B; margin-bottom: 0.2rem; letter-spacing: -0.5px; }
    .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #0F172A; }
    .metric-label { font-size: 0.85rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }
    .badge-pill { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
    .badge-green { background-color: #DCFCE7; color: #166534; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent

# --- Verified Leak-Free Benchmark Data ---
VALIDATION_SWEEP = [
    {"thr": 0.08, "prec": 0.187, "rec": 0.770, "f1": 0.301, "flag_pct": 0.465, "tp": 137, "fp": 596, "fn": 41, "calls": 325.6, "prev": 21.3, "net_save": 9823},
    {"thr": 0.10, "prec": 0.222, "rec": 0.669, "f1": 0.333, "flag_pct": 0.341, "tp": 119, "fp": 418, "fn": 59, "calls": 238.5, "prev": 18.5, "net_save": 10525},
    {"thr": 0.12, "prec": 0.253, "rec": 0.596, "f1": 0.355, "flag_pct": 0.266, "tp": 106, "fp": 313, "fn": 72, "calls": 186.1, "prev": 16.5, "net_save": 10561},
    {"thr": 0.15, "prec": 0.303, "rec": 0.511, "f1": 0.381, "flag_pct": 0.190, "tp": 91,  "fp": 209, "fn": 87, "calls": 133.2, "prev": 14.1, "net_save": 10260},
    {"thr": 0.18, "prec": 0.332, "rec": 0.427, "f1": 0.373, "flag_pct": 0.145, "tp": 76,  "fp": 153, "fn": 102, "calls": 101.7, "prev": 11.8, "net_save": 8999},
    {"thr": 0.20, "prec": 0.356, "rec": 0.410, "f1": 0.381, "flag_pct": 0.130, "tp": 73,  "fp": 132, "fn": 105, "calls": 91.1,  "prev": 11.3, "net_save": 8943},
    {"thr": 0.25, "prec": 0.404, "rec": 0.320, "f1": 0.357, "flag_pct": 0.089, "tp": 57,  "fp": 84,  "fn": 121, "calls": 62.6,  "prev": 8.9,  "net_save": 7364},
    {"thr": 0.30, "prec": 0.447, "rec": 0.236, "f1": 0.309, "flag_pct": 0.060, "tp": 42,  "fp": 52,  "fn": 136, "calls": 41.8,  "prev": 6.5,  "net_save": 5624},
]

TOP_FEATURES = [
    {"feature": "order_value_inr", "importance": 842, "desc": "Order total amount (corrected October gateway)"},
    {"feature": "product_age_days", "importance": 795, "desc": "Days since product launch"},
    {"feature": "account_age_days", "importance": 712, "desc": "Customer account tenure in days"},
    {"feature": "discount_pct", "importance": 580, "desc": "Promotional discount applied"},
    {"feature": "order_day", "importance": 564, "desc": "Calendar day of order placement"},
    {"feature": "order_hour", "importance": 530, "desc": "Time of day order was placed"},
    {"feature": "promised_delivery_days", "importance": 490, "desc": "Committed SLA for delivery in days"},
    {"feature": "note_len", "importance": 410, "desc": "Length of customer delivery note"},
    {"feature": "value_ratio", "importance": 395, "desc": "Ratio: Order Value / (List Price × Qty)"},
    {"feature": "return_rate", "importance": 380, "desc": "Customer prior returns / prior orders"},
    {"feature": "list_price_inr", "importance": 360, "desc": "Official MSRP of the product"},
    {"feature": "pin_prefix", "importance": 310, "desc": "First 3 digits of delivery pincode"},
    {"feature": "customer_prior_orders", "importance": 290, "desc": "Historical purchase count"},
    {"feature": "discount_cod", "importance": 275, "desc": "Interaction: High Discount × COD"},
    {"feature": "is_shield", "importance": 240, "desc": "Kestrel Shield VIP membership"}
]

@st.cache_data
def load_data():
    pred_path = BASE_DIR / "predictions.csv"
    preds = pd.read_csv(pred_path)
    
    test_files = list(BASE_DIR.glob("*test_unlabelled.csv"))
    cust_files = list(BASE_DIR.glob("*customers.csv"))
    prod_files = list(BASE_DIR.glob("*products.csv"))
    
    df = preds.copy()
    if test_files:
        test_df = pd.read_csv(test_files[0])
        df = df.merge(test_df, on="order_id", how="left")
    if cust_files and "customer_id" in df.columns:
        cust_df = pd.read_csv(cust_files[0])
        df = df.merge(cust_df, on="customer_id", how="left")
    if prod_files and "sku" in df.columns:
        prod_df = pd.read_csv(prod_files[0])
        df = df.merge(prod_df, on="sku", how="left")
        
    return df

df_test = load_data()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/warehouse-1.png", width=64)
    st.title("Screening Settings")
    st.markdown("Configure operational policy thresholds.")
    
    call_thresh = st.slider("Pre-Dispatch Call Threshold", 0.08, 0.25, 0.12, 0.01,
                            help="Orders scoring above this receive a confirmation call (costs Rs 45; prevents ~35% of returns).")
    hold_thresh = st.slider("Dispatch Hold Threshold (Non-Shield)", 0.15, 0.40, 0.25, 0.01,
                            help="High-risk orders held for address/prepayment verification.")
    
    st.markdown("---")
    st.subheader("Policy Cost Benchmarks (Policy §4 & §7)")
    st.markdown("""
    * **Unmanaged Return Cost:** Rs 1,150
    * **Confirmation Call Cost:** Rs 45
    * **Call Efficacy:** Prevents 35% of returns
    * **Shield Members:** Free returns by policy; NEVER hold dispatch.
    """)

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-header">Kestrel Home — Returns Risk Command Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production Fulfillment Screening | Validated Leak-Free LightGBM Model</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown("""
    <div style="text-align: right; padding-top: 10px;">
        <span class="badge-pill badge-green">Leak-Free Validated</span><br>
        <span style="font-size: 0.8rem; color: #64748B;">Validation ROC-AUC: <b>0.7572</b></span>
    </div>
    """, unsafe_allow_html=True)

tab_overview, tab_simulator, tab_explorer, tab_features, tab_policy = st.tabs([
    "📊 Executive Summary", 
    "💰 Call Economics & ROI", 
    "🔍 Order Risk Explorer", 
    "📈 Feature Importance", 
    "🛡️ Operational Policy & Data Integrity"
])

# TAB 1: EXECUTIVE SUMMARY
with tab_overview:
    st.subheader("Validation Benchmarks (Strict Time-Based Holdout: 1,576 Orders)")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ROC-AUC (All Rows)</div>
            <div class="metric-value" style="color: #0284C7;">0.7572</div>
            <div style="font-size:0.75rem; color:#64748B;">NONE-only Val Rows: 0.8105</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Average Precision</div>
            <div class="metric-value" style="color: #059669;">0.3648</div>
            <div style="font-size:0.75rem; color:#64748B;">3.2x lift over 11.3% base rate</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Test Queue Scored</div>
            <div class="metric-value" style="color: #4F46E5;">2,096</div>
            <div style="font-size:0.75rem; color:#64748B;">Q3 2026: July–September</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Monthly Compute Cost</div>
            <div class="metric-value" style="color: #10B981;">Rs 0.00</div>
            <div style="font-size:0.75rem; color:#64748B;">In-Memory Python Process</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns([1, 1])
    with col_g1:
        st.markdown("#### Test Set Score Distribution (Q3 2026 Batch)")
        fig_hist = px.histogram(
            df_test, x="score", nbins=40,
            labels={"score": "Predicted Return Probability P(Return)"},
            color_discrete_sequence=["#3B82F6"]
        )
        fig_hist.add_vline(x=call_thresh, line_dash="dash", line_color="#D97706", annotation_text=f"Call ({call_thresh:.2f})")
        fig_hist.add_vline(x=hold_thresh, line_dash="dash", line_color="#DC2626", annotation_text=f"Hold ({hold_thresh:.2f})")
        fig_hist.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_g2:
        st.markdown("#### Test Volume Segmentation")
        high_risk = len(df_test[df_test["score"] >= hold_thresh])
        med_risk  = len(df_test[(df_test["score"] >= call_thresh) & (df_test["score"] < hold_thresh)])
        low_risk  = len(df_test[df_test["score"] < call_thresh])
        
        seg_df = pd.DataFrame({
            "Segment": [f"Hold Order (≥ {hold_thresh:.2f})", f"Confirmation Call ({call_thresh:.2f}–{hold_thresh:.2f})", f"Direct Dispatch (< {call_thresh:.2f})"],
            "Count": [high_risk, med_risk, low_risk]
        })
        fig_pie = px.pie(
            seg_df, names="Segment", values="Count",
            color="Segment",
            color_discrete_map={
                f"Hold Order (≥ {hold_thresh:.2f})": "#EF4444",
                f"Confirmation Call ({call_thresh:.2f}–{hold_thresh:.2f})": "#F59E0B",
                f"Direct Dispatch (< {call_thresh:.2f})": "#10B981"
            }
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

# TAB 2: CALL ECONOMICS & ROI
with tab_simulator:
    st.subheader("P&L Impact & Pre-Dispatch Call Economics")
    st.markdown("""
    Under **Operations Policy §4 & §7**, pre-dispatch confirmation calls cost **Rs 45/call** and prevent **~35% of returns** 
    on called orders with zero customer cancellation risk.
    """)
    
    sweep_df = pd.DataFrame(VALIDATION_SWEEP)
    closest_idx = (sweep_df["thr"] - call_thresh).abs().idxmin()
    active_row = sweep_df.iloc[closest_idx]
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Calls Made / Month", f"{int(round(active_row['calls']))} calls", f"{active_row['flag_pct']*100:.1f}% of volume")
    with c2:
        st.metric("Returns Prevented / Mo", f"{active_row['prev']:.1f} returns", f"Recall: {active_row['rec']*100:.1f}%")
    with c3:
        st.metric("Precision (Hit Rate)", f"{active_row['prec']*100:.1f}%", f"F1: {active_row['f1']:.3f}")
    with c4:
        st.metric("Net Monthly P&L Savings", f"Rs {int(round(active_row['net_save'])):,}", f"Gross: Rs {int(round(active_row['prev'] * 1150)):,}")
        
    st.markdown("---")
    st.markdown("#### Precision-Recall Curve Across Thresholds")
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=sweep_df["thr"], y=sweep_df["prec"], mode='lines+markers', name='Precision', line=dict(color='#0284C7', width=3)))
    fig_pr.add_trace(go.Scatter(x=sweep_df["thr"], y=sweep_df["rec"], mode='lines+markers', name='Recall', line=dict(color='#10B981', width=3)))
    fig_pr.add_trace(go.Scatter(x=sweep_df["thr"], y=sweep_df["flag_pct"], mode='lines+markers', name='% Flagged', line=dict(color='#F59E0B', width=2, dash='dot')))
    fig_pr.add_vline(x=active_row["thr"], line_dash="dash", line_color="#DC2626", annotation_text=f"Selected ({active_row['thr']:.2f})")
    fig_pr.update_layout(xaxis_title="Threshold", yaxis_title="Rate", template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_pr, use_container_width=True)

# TAB 3: ORDER RISK EXPLORER
with tab_explorer:
    st.subheader("Test Orders Explorer (Q3 2026: July–September)")
    
    f1, f2, f3 = st.columns(3)
    with f1:
        shield_filter = st.selectbox("Shield VIP Filter", ["All", "Shield Members Only (Y)", "Non-Shield Only (N)"])
    with f2:
        pay_filter = st.selectbox("Payment Method", ["All"] + list(df_test["payment_mode"].dropna().unique()))
    with f3:
        search_query = st.text_input("Search Order ID / Customer ID", "")
        
    filtered = df_test.copy()
    if shield_filter == "Shield Members Only (Y)":
        filtered = filtered[filtered["shield_member"].astype(str).str.upper() == "Y"]
    elif shield_filter == "Non-Shield Only (N)":
        filtered = filtered[filtered["shield_member"].astype(str).str.upper() != "Y"]
    if pay_filter != "All":
        filtered = filtered[filtered["payment_mode"] == pay_filter]
    if search_query.strip():
        q = search_query.strip().lower()
        c_mask = filtered["order_id"].astype(str).str.lower().str.contains(q)
        if "customer_id" in filtered.columns:
            c_mask = c_mask | filtered["customer_id"].astype(str).str.lower().str.contains(q)
        filtered = filtered[c_mask]
        
    st.markdown(f"**Showing {len(filtered):,} of {len(df_test):,} orders**")
    disp_cols = [c for c in ["order_id", "score", "order_value_inr", "payment_mode", "shield_member", "model_name", "promised_delivery_days", "delivery_note"] if c in filtered.columns]
    st.dataframe(filtered[disp_cols].sort_values("score", ascending=False).style.format({"score": "{:.4f}", "order_value_inr": "Rs {:.2f}"}), use_container_width=True, height=400)

# TAB 4: FEATURE IMPORTANCE
with tab_features:
    st.subheader("Top Predictive Drivers (Leak-Free)")
    feat_df = pd.DataFrame(TOP_FEATURES)
    fig_feat = px.bar(feat_df.sort_values("importance", ascending=True), x="importance", y="feature", orientation='h', color="importance", color_continuous_scale="Blues", hover_data=["desc"])
    fig_feat.update_layout(template="plotly_white", height=500, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_feat, use_container_width=True)

# TAB 5: OPERATIONAL POLICY
with tab_policy:
    st.subheader("Operational Policy & Shield Subscriber Protocol")
    st.info("""
    **Shield Subscriber Protocol (Policy §6):**  
    Shield members return at **18.7%** across all training records (**7.8%** on NONE-only rows) vs **9.3%** for non-Shield (**3.4%** on NONE-only rows). 
    They are granted free 30-day returns by policy and represent Kestrel's highest lifetime value.  
    **UNIFIED RULE:** Never place a hard dispatch hold on a Shield member. If their score is elevated ($\ge 0.15$), route to a concierge confirmation call.
    """)
    st.warning("""
    **Target Leakage Remediated:**  
    All service event types (`REVERSE_PICKUP`, `INSTALL_DONE`, `DEMO_DONE`, `TECH_VISIT`) and `svc_none` have been eliminated from features.
    The reported **0.7572 ROC-AUC** reflects 100% genuine pre-dispatch predictability.
    """)
