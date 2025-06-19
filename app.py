
#####################
import streamlit as st
import pandas as pd
import plotly.express as px
from multi_agents import run_agent_pipeline  # Make sure this is in the same directory or installed as a module
from Simulation import call_agent
from data_preparation_script import generate_daily_ui_data
import os
##Data loading ----------------------
data_dir = "Data"
os.makedirs(data_dir, exist_ok=True)

# Generate data
internal_df, external_df = generate_daily_ui_data()

# Save dataframes separately
internal_df.to_csv(os.path.join(data_dir, "internal_df.csv"), index=False)
external_df.to_csv(os.path.join(data_dir, "external_df.csv"), index=False)
###------------------------ 
st.set_page_config(layout="wide")

# Directly invoke the agent pipeline and cache the result
@st.cache_resource(show_spinner="Running agent pipeline...")
def get_agent_data():
    try:
        data = run_agent_pipeline()
        # print(data,"=============================")
        return data
    except Exception as e:
        st.error(f"Failed to retrieve data: {e}")
        return {}

data = get_agent_data()

# --------------------------------------------------
# 1️⃣  TREND ANALYSIS SECTION
# --------------------------------------------------
st.title("📊 Demand Sense")

trend_analysis_15day = data.get("trend_analysis_15day", {})
trends_data = trend_analysis_15day.get("trend_data", [])

trend_df = pd.DataFrame([
    {
        "Product": item["Product_Name"],
        "Category": item["Category"],
        "Region": item["Region"],
        "Platform": item["Platform"],
        "Trend %": item["trend_pct"],
        "Interest Trend": item["interest_trend"],
    }
    for item in trends_data
])

st.markdown(
    """
    <p style='margin-top:-10px;'>
        <span style='color:#4169e1;'>📡 Real‑time Data</span> &nbsp;
        <span style='color:#228B22;'>📈 Trend Analysis</span> &nbsp;
        <span style='color:#9932CC;'>📊 Performance Tracking</span>
    </p>
    <hr style='margin:10px 0;'>
    """,
    unsafe_allow_html=True,
)

# -- Global Filters
with st.expander("🌐 Global Filters", expanded=True):
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        selected_region = st.selectbox(
            "Region", ["All Regions"] + sorted(trend_df["Region"].unique())
        )
    with fcol2:
        selected_category = st.selectbox(
            "Category", ["All Categories"] + sorted(trend_df["Category"].unique())
        )
    with fcol3:
        selected_platform = st.selectbox(
            "Platform", ["All Platforms"] + sorted(trend_df["Platform"].unique())
        )

filtered_trend_df = trend_df.copy()
if selected_region != "All Regions":
    filtered_trend_df = filtered_trend_df[filtered_trend_df["Region"] == selected_region]
if selected_category != "All Categories":
    filtered_trend_df = filtered_trend_df[filtered_trend_df["Category"] == selected_category]
if selected_platform != "All Platforms":
    filtered_trend_df = filtered_trend_df[filtered_trend_df["Platform"] == selected_platform]

left_col, right_col = st.columns([3, 2])

# -- Trending Table
###----
with left_col:
    st.markdown(
        """
        <div style='background-color:#4169e1;color:white;padding:10px;border-radius:8px;margin-top:10px;'>
            <strong>📈 Trending Topics by Region &amp; Platform</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Helpers ───────────────────────────────────────────────────────────
    # Platform -> (SVG logo URL, badge background, text color)
    _PLAT_ICON = {
        "Instagram": ("https://cdn.jsdelivr.net/npm/simple-icons@9/icons/instagram.svg", "#fde3f3", "#c60f5d"),
        "Facebook":  ("https://cdn.jsdelivr.net/npm/simple-icons@9/icons/facebook.svg",  "#e5efff", "#0e70ff"),
        "X":         ("https://cdn.jsdelivr.net/npm/simple-icons@9/icons/x.svg",         "#e9e9ea", "#000000"),
        "TikTok":    ("https://cdn.jsdelivr.net/npm/simple-icons@9/icons/tiktok.svg",    "#D3D3D3", "#000000"),
    }

    def _trend_html(v: str) -> str:
        """Color the % Trend value."""
        return f"<span style='color:{'green' if '+' in v else 'red'};font-weight:600;'>{v}</span>"

    def _plat_html(p: str) -> str:
        """Return pill with SVG icon + platform name."""
        logo, bg, fg = _PLAT_ICON.get(p, ("", "#f0f0f0", "#333"))
        img = f"<img src='{logo}' width='14' style='margin-right:4px;vertical-align:middle;'>" if logo else "❓"
        return (
            f"<div style='background:{bg};color:{fg};padding:2px 8px;border-radius:12px;display:inline-flex;"
            f"align-items:center;font-size:12px;'>{img}{p}</div>"
        )

    # ── Build DataFrame (no Category) ─────────────────────────────────────
    # table_df = (
    #     filtered_trend_df[["Product", "Region", "Trend %", "Platform"]]
    #     .rename(columns={"Product": "Product", "Trend %": "% Trend"})
    # )

    table_df = (
        filtered_trend_df[["Product", "Region", "Trend %", "Platform"]]
        .rename(columns={
            "Product": "Product",
            "Region" : "Region",
            "Trend %": "% Trend",
            "Platform": "Platform",
        })
        # 🔀 Randomise order so one product’s rows aren’t clumped together
        .sample(frac=1, random_state=None)   # remove random_state for true shuffle each load
        .reset_index(drop=True)
    )
    # table_df["_sort"] = (
    # table_df["% Trend"]
    #     .str.replace(r"[+%]", "", regex=True)   # "-12%" → "-12", "+40%" → "40"
    #     .astype(float)
    # )
    # table_df = table_df.sort_values("_sort", ascending=False).drop(columns="_sort")
    # ── Convert DF ➜ HTML ────────────────────────────────────────────────
    def _df_to_html(df: pd.DataFrame) -> str:
        th = "padding:6px 10px;text-align:left;font-weight:bold;background:#fafafa;"
        td = "padding:6px 10px;border-bottom:1px solid #f0f0f0;font-size:13px;"
        html = ["<table style='border-collapse:collapse;width:100%;'>"]
        html.append("<thead><tr>" + "".join([f"<th style='{th}'>{c}</th>" for c in df.columns]) + "</tr></thead><tbody>")
        for _, row in df.iterrows():
            html.append("<tr>")
            for col, val in row.items():
                if col == "% Trend":
                    cell = _trend_html(val)
                elif col == "Platform":
                    cell = _plat_html(val)
                else:
                    cell = val
                html.append(f"<td style='{td}'>{cell}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")
        return "".join(html)

    # ── Scrollable container (fixed 420 px height) ───────────────────────
    st.markdown(
        f"<div style='height:420px;overflow-y:auto;padding-right:4px;'>" + _df_to_html(table_df) + "</div>",
        unsafe_allow_html=True,
    )

with right_col:
    # 1️⃣ Product selector (unique names only)
    prod_choice = st.selectbox(
        "Select a product to view its aggregated Interest Trend:",
        sorted(filtered_trend_df["Product"].unique()),
        key="sel_interest_product",
    )

    # 2️⃣ Grab all rows for that product
    prod_rows = filtered_trend_df[filtered_trend_df["Product"] == prod_choice]

    # 3️⃣ Build one long DataFrame of (day, interest) across all regions/platforms
    long_df_list = []
    for _, r in prod_rows.iterrows():
        tmp = pd.DataFrame(r["Interest Trend"])        # columns: day, interest
        tmp["day"] = pd.to_datetime(tmp["day"])
        long_df_list.append(tmp)

    merged_df = pd.concat(long_df_list)

    # 4️⃣  Aggregate per day → mean  (could be sum/max if preferred)
    agg_df = (
        merged_df.groupby("day", as_index=False)["interest"]
        .mean()
        .sort_values("day")
    )

    # 5️⃣  Plot
    st.markdown(
        f"<div style='background-color:#228B22;padding:10px;border-radius:10px;color:white;text-align:center;'>"
        f"📊 Interest Trend ({prod_choice}) – All Regions / Platforms</div>",
        unsafe_allow_html=True,
    )

    fig = px.line(agg_df, x="day", y="interest", markers=True, line_shape="spline",
                  color_discrete_sequence=["#228B22"])
    fig.update_layout(height=300, margin=dict(t=30, r=10, l=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 6️⃣ Inline insight card (compare first vs. last day for a quick % change)
    delta_pct = (
        (agg_df["interest"].iloc[-1] - agg_df["interest"].iloc[0])
        / agg_df["interest"].iloc[0] * 100
    )
    delta_pct_str = f"{delta_pct:+.0f}%"
    bg = "#e6f9e6" if delta_pct > 0 else "#f9e6e6"

    st.markdown(
        f"<div style='background-color:{bg};padding:15px;border-radius:10px;text-align:center;margin-top:10px;'>"
        f"<strong>Insight:</strong> Overall interest for <em>{prod_choice}</em> changed by "
        f"<strong>{delta_pct_str}</strong> over the selected period.</div>",
        unsafe_allow_html=True,
    )
# --------------------------------------------------
# 🔻 Separator
# --------------------------------------------------
st.markdown("<hr style='margin:40px 0;'>", unsafe_allow_html=True)

# # --------------------------------------------------
# # 2️⃣  PRODUCT RANKING & ANOMALIES SECTION
# # --------------------------------------------------
# Fetch all required slices once
matrix       = data.get("matrix", {})
ranking_data = data.get("ranking", [])
top_mover    = data.get("top_mover", {})
# 🔝 Breakout & Rising – sort by numeric % desc and slice top‑5
b_raw = data.get("breakout_products", [])
breakout = sorted(
    b_raw,
    key=lambda x: float(str(x.get("breakout_lift_pct", "0").replace("%", "").replace(",", ""))),
    reverse=True,
)[:5]

r_raw = data.get("rising_queries", [])
rising = sorted(
    r_raw,
    key=lambda x: float(str(x.get("trends_lift_pct", "0").replace("%", "").replace(",", ""))),
    reverse=True,
)[:5]


# Create three responsive columns: [Anomaly Matrix] – [Ranking] – [Breakout/Rising]
acol, pcol, qcol = st.columns([1.35, 1.1, 1.1])

# ── 2.A  Anomaly Intensity Matrix  ────────────────────────────────────────────
color_map = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}

with acol:
    st.markdown(
        """<div style='background-color:#C24B05;color:white;padding:12px 10px;border-radius:8px;display:flex;align-items:center;'>
        ⚠️ <strong style='margin-left:4px;'>Anomaly Intensity Matrix</strong></div>""",
        unsafe_allow_html=True,
    )
    # Legend row
    st.markdown(
        "<p style='font-size:12px;margin:6px 0 4px 4px;'>"
        "<span style='color:green;'>● Low</span> &nbsp;"
        "<span style='color:orange;'>● Medium</span> &nbsp;"
        "<span style='color:red;'>● High</span></p>",
        unsafe_allow_html=True,
    )
    # Category cards
    for cat, regions in matrix.items():
        st.markdown(
            f"<div style='background-color:#f6f8fb;border:1px solid #e1e4e8;border-radius:10px;padding:8px;margin-bottom:8px;'>"
            f"<strong style='font-size:14px;'>{cat}</strong><br>",
            unsafe_allow_html=True,
        )
        r_cols = st.columns(len(regions))
        for rc, (region, intensity) in zip(r_cols, regions.items()):
            icon = color_map.get(intensity, "")
            rc.markdown(
                f"<div style='text-align:center;font-size:12px;'>"
                f"<strong>{region}</strong><br>{icon} {intensity}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

# ── 2.B  Product Ranking Tracker  ─────────────────────────────────────────────

with pcol:
    st.markdown(
        """<div style='background-color:#6B23D3;color:white;padding:12px 10px;border-radius:8px;display:flex;align-items:center;'>
        🏆 <strong style='margin-left:4px;'>Product Ranking Tracker</strong></div>""",
        unsafe_allow_html=True,
    )
    for itm in ranking_data:
        rank, prev, dir_ = itm["current_rank"], itm["prev_week_rank"], itm["direction"]

        # 🔄  NEW icons – plain arrows, no blue square
        icon  = "↑" if dir_ == "improved" else ("↓" if dir_ == "worsened" else "—")
        color = "green" if dir_ == "improved" else ("red"   if dir_ == "worsened" else "black")
        weight = "600" if icon == "—" else "400"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"border:1px solid #e1e4e8;padding:8px 10px;border-radius:8px;margin-top:6px;"
            f"background-color:#ffffff;'>"
            f"<span style='font-weight:600;'>#{rank} {itm['Product_Name']}</span>"
            f"<span style='font-size:13px;'>Prev #{prev} "
            f"<span style='color:{color};font-size:16px;'>{icon}</span></span>"
            "</div>",
            unsafe_allow_html=True,
        )
    # # Individual ranking rows
    # for itm in ranking_data:
    #     rank, prev, dir_ = itm["current_rank"], itm["prev_week_rank"], itm["direction"]
    #     icon = "🔼" if dir_ == "improved" else ("🔽" if dir_ == "worsened" else "➖")
    #     color = "green" if dir_ == "improved" else ("red" if dir_ == "worsened" else "black")
    #     st.markdown(
    #         f"<div style='display:flex;justify-content:space-between;align-items:center;border:1px solid #e1e4e8;"
    #         f"padding:8px 10px;border-radius:8px;margin-top:6px;background-color:#ffffff;'>"
    #         f"<span style='font-weight:600;'>#{rank} {itm['Product_Name']}</span>"
    #         f"<span style='font-size:13px;'>Prev #{prev} <span style='color:{color};font-size:16px;'>{icon}</span></span>"
    #         "</div>",
    #         unsafe_allow_html=True,
    #     )

    if top_mover:
        st.markdown(
            f"<div style='background-color:#ede7ff;padding:8px;border-radius:8px;margin-top:8px;font-size:12px;'>"
            f"<strong>Top Mover:</strong> {top_mover['Product_Name']} climbed from #{top_mover['from_rank']} → #{top_mover['to_rank']} this week!"
            "</div>",
            unsafe_allow_html=True,
        )

# ── 2.C  Breakout Products & Rising Queries  ──────────────────────────────────

with qcol:
    st.markdown(
        """<div style='background-color:#C60F5D;color:white;padding:12px 10px;border-radius:8px;display:flex;align-items:center;'>💡 <strong style='margin-left:4px;'>Breakout Products & Rising Queries</strong></div>""",
        unsafe_allow_html=True,
    )

    # Breakout Products
    # ── Breakout Products
    st.markdown(
        "<h5 style='text-align:center;color:#C60F5D;margin-top:6px;'>Breakout Products</h5>",
        unsafe_allow_html=True,
    )

    b_html = "<div class='scroll-list'>"
    for itm in breakout:
        b_html += (
            f"<div style='background:#fff2f8;border-radius:8px;padding:6px 10px;"
            f"margin-bottom:4px;display:flex;justify-content:space-between;"
            f"align-items:center;'>"
            f"<span>{itm['Product_Name']}</span>"
            f"<span style='color:#C60F5D;font-weight:600;'>{itm['breakout_lift_pct']}</span>"
            "</div>"
        )
    b_html += "</div>"
    st.markdown(b_html, unsafe_allow_html=True)

    # ── Rising Queries
    st.markdown(
        "<h5 style='text-align:center;color:#0E70FF;margin-top:10px;'>Rising Queries</h5>",
        unsafe_allow_html=True,
    )

    r_html2 = "<div class='scroll-list'>"
    for itm in rising:
        r_html2 += (
            f"<div style='background:#eef6ff;border-radius:8px;padding:6px 10px;"
            f"margin-bottom:4px;display:flex;justify-content:space-between;"
            f"align-items:center;'>"
            f"<span>{itm['Product_Name']}</span>"
            f"<span style='color:#0E70FF;font-weight:600;'>{itm['trends_lift_pct']}</span>"
            "</div>"
        )
    r_html2 += "</div>"
    st.markdown(r_html2, unsafe_allow_html=True)



# --------------------------------------------------
# 🔻 Separator
# --------------------------------------------------
st.markdown("<hr style='margin:40px 0;'>", unsafe_allow_html=True)


# --------------------------------------------------
# 3️⃣  PROMOTION SIMULATOR SECTION
# --------------------------------------------------
# ────────────────────────────────────────────────────────────────────────────────
# 3️⃣  PROMOTION / PRICING SIMULATOR  (🪄 Interactive)
# ────────────────────────────────────────────────────────────────────────────────

# ▸▸ Promotion Simulator  (styled card) ▸▸
st.markdown(
    "<div style='background-color:#4B3FFF;color:white;padding:12px 16px;"
    "border-radius:8px;font-size:18px;font-weight:600;display:flex;gap:6px;"
    "align-items:center;'>🗒️ Promotion Simulator</div>",
    unsafe_allow_html=True,
)
# put (or replace) this immediately after the “## Promotion Simulator” header
st.markdown("""
<style>
:root{ --promo:#4B3FFF; --promoRail:#e7e6ff; }

/* ▶ Run button – unchanged */
div.stButton>button:first-child{
    background:var(--promo)!important;color:#fff!important;border:none!important;
    border-radius:6px!important;height:42px;font-weight:600;width:100%;
}
div.stButton>button:first-child:hover{background:#3a33e5!important;}

/* ▶ Rail (full length, light lavender) */
div.stSlider div[data-baseweb="slider"]>div:first-child{
    background:var(--promoRail)!important;
}

/* ▶ Active track (left segment) */
div.stSlider div[data-baseweb="slider"]>div:first-child>div:nth-child(1){
    background:var(--promo)!important;
}

/* ▶ Inactive track (right segment) */
div.stSlider div[data-baseweb="slider"]>div:first-child>div:nth-child(2){
    background:var(--promoRail)!important;
}

/* ▶ Knob */
div.stSlider div[role="slider"]{
    background:#fff!important;border:2px solid var(--promo)!important;
}
</style>
""", unsafe_allow_html=True)
# ── Parameter Inputs ──────────────────────────────────────────────────────────
# Product selection
product_list = ['Eye Mask', 'Ice Bar', 'Yoga Mat', 'Earbuds','Detergent']
selected_product = st.selectbox("Select Product", product_list, key="product_selector")

# Two side-by-side sliders
col_l, col_r = st.columns(2)

with col_l:
    discount = st.slider(
        "Discount Depth (%)",
        min_value=0,
        max_value=50,
        value=5,
        help="Percentage discount to apply to the product price",
        key="discount_slider",
    )

with col_r:
    social_amp = st.slider(
        "Social Amplification (%)",
        min_value=0,
        max_value=100,
        value=75,
        help="Percentage increase in social media engagement and trends",
        key="social_amp_slider",
    )

# Display current parameters
st.info(f"**Simulation Parameters:** {selected_product} with {discount}% discount and {social_amp}% social amplification")

# ── Run Simulation Button ─────────────────────────────────────────────────────
# if st.button("🚀 Run Simulation", type="primary"):
if st.button("Run Simulation", type="primary", use_container_width=True):
        # Show loading spinner
        with st.spinner('Running simulation...'):
            query = f"Can you run a sales forecast for the '{selected_product}' with a {discount}% discount and a {social_amp}% social media amplification?"
            
            try:
                
                simulation_result = call_agent(query)
                
                # Validate the simulation result
                if isinstance(simulation_result, dict) and "Sales Volume (Units)" in simulation_result:
                    sales_data = simulation_result["Sales Volume (Units)"]
                    profit_data = simulation_result["Profit Margin (%)"]
                    
                    # st.write("### 📊 Simulation Results")
                    st.markdown("""
                    <div style='background-color:#008080;padding:10px 15px;border-radius:10px;margin-bottom:10px;'>
                        <h3 style='color:white;margin:0;'>📊 Simulation Results</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    # Create DataFrames for visualization
                    sales_df = pd.DataFrame({
                        'Week': ['Current', 'Week 1', 'Week 2'],
                        'Sales Volume': [sales_data['Current'], sales_data['Week 1'], sales_data['Week 2']]
                    })
                    
                    profit_df = pd.DataFrame({
                        'Week': ['Current', 'Week 1', 'Week 2'],
                        'Profit Margin (%)': [profit_data['Current'], profit_data['Week 1'], profit_data['Week 2']]
                    })
                    
                    # Display key metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Current Sales", 
                            f"{sales_data['Current']:,} units"
                        )
                    
                    with col2:
                        st.metric(
                            "Projected Sales", 
                            f"{sales_data['Week 1']:,} units",
                            delta=f"{sales_data['Week 1'] - sales_data['Current']:,}"
                        )
                    
                    with col3:
                        st.metric(
                            "Current Profit Margin", 
                            f"{profit_data['Current']:.1f}%"
                        )
                    
                    with col4:
                        st.metric(
                            "Projected Profit Margin", 
                            f"{profit_data['Week 1']:.1f}%",
                            delta=f"{profit_data['Week 1'] - profit_data['Current']:.1f}%"
                        )
                    
                    # Charts in two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_sales = px.bar(
                            sales_df, 
                            x='Week', 
                            y='Sales Volume', 
                            title='📈 Sales Volume Comparison',
                            color='Week',
                            color_discrete_sequence=['#008080', '#008080', '#008080']
                        )
                        fig_sales.update_layout(showlegend=False)
                        st.plotly_chart(fig_sales, use_container_width=True)
                    
                    with col2:
                        fig_profit = px.bar(
                            profit_df, 
                            x='Week', 
                            y='Profit Margin (%)', 
                            title='💰 Profit Margin Comparison',
                            color='Week',
                            color_discrete_sequence=['#008080', '#008080', '#008080']
                        )
                        fig_profit.update_layout(showlegend=False)
                        st.plotly_chart(fig_profit, use_container_width=True)
                    
                    # Impact summary with appropriate styling
                    impact_summary = simulation_result["Impact Summary"]
                    if "increase" in impact_summary and not impact_summary.startswith("-"):
                        st.success(f"✅ {impact_summary}")
                    elif "decrease" in impact_summary or impact_summary.startswith("-"):
                        st.warning(f"⚠️ {impact_summary}")
                    else:
                        st.info(f"ℹ️ {impact_summary}")
                    
                    # Additional insights
                    st.write("### 💡 Key Insights")
                    
                    sales_change = sales_data['Week 1'] - sales_data['Current']
                    profit_change = profit_data['Week 1'] - profit_data['Current']
                    
                    insights = []
                    
                    if sales_change > 0:
                        insights.append(f"• Sales volume is projected to increase by {sales_change:,} units")
                    else:
                        insights.append(f"• Sales volume may decrease by {abs(sales_change):,} units")
                    
                    if profit_change > 0:
                        insights.append(f"• Profit margin is expected to improve by {profit_change:.1f} percentage points")
                    else:
                        insights.append(f"• Profit margin may decline by {abs(profit_change):.1f} percentage points")
                    
                    if discount > 20:
                        insights.append("• High discount depth may significantly impact profitability")
                    
                    if social_amp > 80:
                        insights.append("• Strong social amplification can drive substantial awareness")
                    
                    for insight in insights:
                        st.write(insight)
                    
                elif isinstance(simulation_result, dict) and "error" in simulation_result:
                    st.error(f"❌ Simulation Error: {simulation_result['error']}")
                
                else:
                    st.error("❌ Unexpected response format from simulation. Please try again.")
                    st.write("Debug - Received:", simulation_result)
            
            except Exception as e:
                st.error(f"❌ An error occurred during simulation: {str(e)}")
                st.write("Please check your data setup and try again.")

