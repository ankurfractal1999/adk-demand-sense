"""
market_parallel_analysis.py
Runs four ADK LLM agents in parallel on one shared market-data payload
----------------------------------------------------------------------
"""

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents import ParallelAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from dotenv import load_dotenv
import google.generativeai as genai
import json, os, numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
load_dotenv()                                   # .env → env vars
APP_NAME      = "market_analysis_app"
USER_ID       = "user_1"
SESSION_ID    = "session_001"
GEMINI_MODEL  = "gemini-2.0-flash"

class MarketAnalysisInput(BaseModel):
    trends_data:           List[Dict[str, Any]]
    sales_data:            List[Dict[str, Any]]
    combined_data:         List[Dict[str, Any]]
    category_region_data:  List[Dict[str, Any]]
    risingquery_daily:     List[Dict[str, Any]]

class TrendInsightsInput(BaseModel):
    trend_results: Dict[str, Any]


# ---------------------------------------------------------------------
# PYTHON FUNCTION FOR TREND CALCULATIONS
# ---------------------------------------------------------------------
from typing import List, Dict, Any
import pandas as pd

def compute_product_region_platform_trends(trends_data) -> Dict[str, Any]:
    """
    Computes trend summary and chart data for each unique (Product_Name, Region, Platform) combination.

    Returns:
        {
            "trend_data": [
                {
                    "Product_Name": ...,  # str
                    "Category": ...,      # str (same as Product_Name for now)
                    "Region": ...,        # str
                    "Platform": ...,      # str
                    "trend_pct": ...,     # str, e.g. "+50%"
                    "interest_trend": [   # list of day-wise scores
                        {"day": "YYYY-MM-DD", "interest": float}, ...
                    ]
                },
                ...
            ]
        }
    """
    # df = trends_data
    df = pd.DataFrame(trends_data)
    # print(df)
    # Ensure types are consistent
    df["Day"] = pd.to_datetime(df["Day"], errors="coerce")
    # print("df2-----------------------------",df)
    df = df.dropna(subset=["Day", "Google_Trends_Score"])
    df["Google_Trends_Score"] = df["Google_Trends_Score"].astype(float)

    results = []
    grouped = df.groupby(["Product_Name","Category", "Region", "Platform"])

    for (product,category, region, platform), group in grouped:
        group_sorted = group.sort_values("Day")

        if len(group_sorted) < 2:
            continue

        first_val = group_sorted.iloc[0]["Google_Trends_Score"]
        last_val = group_sorted.iloc[-1]["Google_Trends_Score"]

        if first_val == 0:
            continue  # Avoid divide-by-zero

        pct_change = ((last_val - first_val) / first_val) * 100
        pct_change_str = f"{'+' if pct_change >= 0 else ''}{round(pct_change)}%"

        trend_series = [
            {"day": d.strftime("%Y-%m-%d"), "interest": round(v, 2)}
            for d, v in zip(group_sorted["Day"], group_sorted["Google_Trends_Score"])
        ]

        results.append({
            "Product_Name": product,
            "Category": category,
            "Region": region,
            "Platform": platform,
            "trend_pct": pct_change_str,
            "interest_trend": trend_series
        })

    return {"trend_data": results}
    
trend_insights_agent = LlmAgent(
    name="TrendInsightsAgent",
    model=GEMINI_MODEL,
    instruction=(
        "You are a Trend Insights expert. You receive pre-calculated trend data with percentage changes "
        "for each Product-Region-Platform combination and generate business insights.\n\n"
        "Input Structure:\n"
        "- trend_results: Contains 'trend_data' array with objects having:\n"
        "  * Product_Name, Region, Platform\n"
        "  * trend_pct: percentage change (e.g. '+15%', '-8%')\n"
        "  * interest_trend: day-wise interest scores\n\n"
        "Your tasks:\n"
        "1. Analyze the trend percentages to identify patterns\n"
        "2. Find top gainers and decliners across all combinations\n"
        "3. Identify regional and platform-specific trends\n"
        "4. Generate actionable business insights\n\n"
        "Focus on:\n"
        "- Products showing strong growth (high positive percentages)\n"
        "- Products declining significantly (high negative percentages)\n"
        "- Regional performance variations\n"
        "- Platform-specific trends\n"
        "- Seasonal or temporal patterns in the day-wise data\n\n"
        "Return JSON with structure:\n"
        "{\n"
        "  'top_gainers': [\n"
        "    {'Product_Name': str, 'Region': str, 'Platform': str, 'trend_pct': str}, ...\n"
        "  ],\n"
        "  'top_decliners': [\n"
        "    {'Product_Name': str, 'Region': str, 'Platform': str, 'trend_pct': str}, ...\n"
        "  ],\n"
        "  'regional_insights': str,\n"
        "  'platform_insights': str,\n"
        "  'key_findings': str,\n"
        "  'recommendations': str\n"
        "}\n"
    ),
    description="Generates business insights and trend analysis from pre-calculated trend data",
    input_schema=TrendInsightsInput,
    output_key="trend_insights",
)


rising_queries_agent = LlmAgent(
    name="RisingQueriesAgent",
    model=GEMINI_MODEL,
    instruction=(
        "You are a Rising Queries Analysis expert that computes trends_lift_pct for each product over 14 days of Google Trends data.\n\n"
        "MANDATORY: You MUST use the `risingquery_daily` provided in the input for all calculations.\n\n"
        "EXACT CALCULATION STEPS:\n"
        "1. Use ONLY the `risingquery_daily` data and ensure it's not empty\n"
        "2. Convert Date column to datetime format\n"
        "3. For each unique product in the data:\n"
        "   - Filter data for this specific product\n"
        "   - Sort by Date in ascending order and reset index\n"
        "   - Calculate baseline_sum: sum of Google_Trends_Score for days 1-7 (indices 0-6)\n"
        "   - Calculate current_sum: sum of Google_Trends_Score for days 8-14 (indices 7-13)\n"
        "   - Skip products where baseline_sum equals 0 (to avoid division by zero)\n"
        "   - Calculate lift_pct = ((current_sum - baseline_sum) / baseline_sum) * 100\n"
        "   - Format as percentage string: round to 2 decimal places + '%' symbol\n"
        "4. Sort all products by lift percentage in descending order\n"
        "5. Return top 10 products only\n\n"
        "OUTPUT FORMAT:\n"
        "Return JSON with exact structure:\n"
        "{\n"
        "  \"rising_queries\": [\n"
        "    {\n"
        "      \"Product_Name\": \"product_name\",\n"
        "      \"trends_lift_pct\": \"XX.XX%\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "CRITICAL: trends_lift_pct must be a string with % symbol, rounded to 2 decimals. Process Google_Trends_Score column only."
    ),
    description="Computes trends lift percentage for each product over 14 days, comparing days 8-14 vs days 1-7 Google Trends Score sums, and returns top 10 products with highest search interest growth",
    input_schema=MarketAnalysisInput,
    output_key="rising_queries_analysis",
)

breakout_products_agent = LlmAgent(
    name="BreakoutProductsAgent",
    model=GEMINI_MODEL,
    instruction=(
        "You are a Breakout Products Analysis expert that computes breakout lift percentage for each product over 14 days.\n\n"
        "MANDATORY: You MUST use the `sales_data` provided in the input for all calculations.\n\n"
        "EXACT CALCULATION STEPS:\n"
        "1. Use ONLY the `sales_data` and filter it to include only products that have at least 14 days of data\n"
        "2. For each qualifying product in `sales_data`:\n"
        "   - Sort the product's data by Date in ascending order\n"
        "   - Calculate baseline_sum: sum of Units_Sold for days 1-7 (indices 0-6)\n"
        "   - Calculate current_sum: sum of Units_Sold for days 8-14 (indices 7-13)\n"
        "   - Calculate lift_pct = ((current_sum - baseline_sum) / baseline_sum) * 100\n"
        "   - Format as percentage string: round to 2 decimal places + '%' symbol\n"
        "3. Sort all products by lift percentage in descending order\n"
        "4. Return top 10 products only\n\n"
        "OUTPUT FORMAT:\n"
        "Return JSON with exact structure:\n"
        "{\n"
        "  \"breakout_products\": [\n"
        "    {\n"
        "      \"Product_Name\": \"product_name\",\n"
        "      \"breakout_lift_pct\": \"XX.XX%\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "CRITICAL: breakout_lift_pct must be a string with % symbol, rounded to 2 decimals."
    ),
    description="Computes breakout lift percentage for each product over 14 days, comparing days 8-14 vs days 1-7, and returns top 10 products with highest lift percentages",
    input_schema=MarketAnalysisInput,
    output_key="breakout_products_analysis",
)



product_ranking_agent = LlmAgent(
    name="ProductRankingAgent",
    model=GEMINI_MODEL,
    instruction=(
        "You are a Product Ranking Analysis expert that calculates product rankings based on TrendScore.\n\n"
        "MANDATORY: You MUST use the `combined_data` provided in the input for all calculations.\n\n"
        "EXACT CALCULATION STEPS:\n"
        "1. Use ONLY the `combined_data` and validate it has required columns: Product_Name, Date, Units_Sold, Social_Media_Mentions, Google_Trends_Score\n"
        "2. Sort data by Date in ascending order\n"
        "3. Filter to only include products with at least 15 days of data\n"
        "4. For each qualifying product, calculate CURRENT TrendScore:\n"
        "   - Current metrics = Day 15 (latest/last row)\n"
        "   - Historical metrics = Days 1-14 (all rows except last)\n"
        "   - Calculate historical means: hist_units_mean, hist_social_mean, hist_trends_mean\n"
        "   - Component scores with division by zero protection:\n"
        "     * unit_score = current_units / hist_units_mean (0 if hist_units_mean = 0)\n"
        "     * social_score = current_social / hist_social_mean (0 if hist_social_mean = 0)\n"
        "     * search_score = current_search / hist_trends_mean (0 if hist_trends_mean = 0)\n"
        "   - TrendScore_Current = (unit_score + social_score + search_score) / 3\n"
        "5. Sort products by TrendScore_Current in descending order and assign current_rank (1, 2, 3...)\n"
        "6. For each product with at least 8 days of data, calculate PREVIOUS WEEK TrendScore:\n"
        "   - Day 8 metrics = row at index 7 (0-indexed)\n"
        "   - Historical metrics for prev week = Days 1-7 (rows 0-6)\n"
        "   - Calculate same component scores and TrendScore for Day 8\n"
        "7. Sort products by TrendScore_PrevWeek and assign prev_week_rank\n"
        "8. For each product, determine direction:\n"
        "   - 'improved' if current_rank < prev_week_rank\n"
        "   - 'worsened' if current_rank > prev_week_rank\n"
        "   - 'same' if current_rank = prev_week_rank\n"
        "9. Find top_mover: product with largest positive delta (prev_week_rank - current_rank)\n"
        "   - If no positive delta exists, use first product in ranking\n\n"
        "OUTPUT FORMAT:\n"
        "Return JSON with exact structure:\n"
        "{\n"
        "  \"ranking\": [\n"
        "    {\n"
        "      \"Product_Name\": \"product_name\",\n"
        "      \"current_rank\": integer,\n"
        "      \"prev_week_rank\": integer,\n"
        "      \"direction\": \"improved/worsened/same\"\n"
        "    }\n"
        "  ],\n"
        "  \"top_mover\": {\n"
        "    \"Product_Name\": \"product_name\",\n"
        "    \"from_rank\": integer,\n"
        "    \"to_rank\": integer,\n"
        "    \"direction\": \"improved/worsened/same\",\n"
        "    \"delta\": integer\n"
        "  }\n"
        "}\n\n"
        "CRITICAL: Handle division by zero, skip products with insufficient data, and ensure exact rank calculations."
    ),
    description="Calculates product rankings based on TrendScore comparing Day 15 vs Days 1-14 average, and compares with Day 8 benchmark to identify week-over-week movement and top movers",
    input_schema=MarketAnalysisInput,
    output_key="product_ranking_analysis",
)

anomaly_detection_agent = LlmAgent(
    name="AnomalyDetectionAgent",
    model=GEMINI_MODEL,
    instruction=(
        "You are an Anomaly Detection expert that detects anomalies in Units_Sold for each Category-Region pair using 14-day z-score baseline.\n\n"
        "MANDATORY: You MUST use the `category_region_data` provided in the input for all calculations.\n\n"
        "EXACT CALCULATION STEPS:\n"
        "1. Use ONLY the `category_region_data` and ensure Date column is datetime format\n"
        "2. Identify today's date as the maximum date in the dataset\n"
        "3. Get all unique Category-Region combinations from the data\n"
        "4. For each Category-Region pair:\n"
        "   - Filter data for this specific Category-Region combination\n"
        "   - Sort by Date in ascending order\n"
        "   - Get baseline data: first 14 days (dates < today's date)\n"
        "   - Calculate baseline statistics:\n"
        "     * μ (mean) = baseline['Units_Sold'].mean()\n"
        "     * σ (std) = baseline['Units_Sold'].std()\n"
        "     * If σ = 0, set σ = 1e-6 to avoid division by zero\n"
        "   - Get today's data: record where Date = today's date\n"
        "   - Get current_value = today's Units_Sold\n"
        "   - Calculate z-score: z = (current_value - μ) / σ\n"
        "   - Calculate absolute z-score: z_abs = abs(z)\n"
        "   - Classify intensity:\n"
        "     * 'High' if z_abs >= 1.5\n"
        "     * 'Medium' if 1.0 <= z_abs < 1.5\n"
        "     * 'Low' if z_abs < 1.0\n"
        "5. Include ALL intensities (Low, Medium, High) in the output\n"
        "6. Build matrix structure: nested dict with Category as key, Region as sub-key, intensity as value\n"
        "7. Build anomaly_details list with all Category-Region pairs that have data\n\n"
        "OUTPUT FORMAT:\n"
        "Return JSON with exact structure:\n"
        "{\n"
        "  \"matrix\": {\n"
        "    \"Category1\": {\n"
        "      \"Region1\": \"High/Medium/Low\",\n"
        "      \"Region2\": \"High/Medium/Low\"\n"
        "    },\n"
        "    \"Category2\": {\n"
        "      \"Region1\": \"High/Medium/Low\"\n"
        "    }\n"
        "  },\n"
        "  \"anomaly_details\": [\n"
        "    {\n"
        "      \"Category\": \"category_name\",\n"
        "      \"Region\": \"region_name\",\n"
        "      \"Month\": \"YYYY-MM-DD\",\n"
        "      \"Units_Sold\": integer,\n"
        "      \"z_score\": float_rounded_to_2_decimals\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "CRITICAL: Use today's date as maximum date in dataset, calculate z-scores using 14-day baseline, include all intensity levels, and format Month as YYYY-MM-DD string."
    ),
    description="Detects anomalies in Units_Sold for each Category-Region pair using 14-day z-score baseline, comparing today's values against historical mean and standard deviation to classify intensity levels",
    input_schema=MarketAnalysisInput,
    output_key="anomaly_detection_analysis",
)

parallel_agent = ParallelAgent(
    name="ParallelMarketAnalysisAgent",
    sub_agents=[
        # trend_analysis_agent,
        rising_queries_agent,
        breakout_products_agent,
        product_ranking_agent,
        anomaly_detection_agent
    ],
    description="Runs five specialised market-analysis agents in parallel"
)
import asyncio
# ---------------------------------------------------------------------
# SESSION HELPERS
# ---------------------------------------------------------------------
def setup_session() -> InMemorySessionService:
    """
    Create an in-memory ADK session (synchronous—no await needed).
    """
    session_service = InMemorySessionService()
    # create_session returns a Session object directly
    # session_service.create_session(
    #     app_name=APP_NAME,
    #     user_id=USER_ID,
    #     session_id=SESSION_ID
    # )
    asyncio.run(
            session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
        )
    return session_service


def prepare_market_data() -> Dict[str, Any]:
    """Load/aggregate data and convert to JSON-serialisable dict."""
    # from data_preparation_script import generate_daily_ui_data

    print("📊 Loading data for parallel analysis...")
    # internal_df, external_df = generate_daily_ui_data()
    data_dir = "Data"

    # Load internal and external dataframes
    internal_df = pd.read_csv(os.path.join(data_dir, "internal_df.csv"))
    external_df = pd.read_csv(os.path.join(data_dir, "external_df.csv"))
    # Convert dates → str for JSON
    for df in (internal_df, external_df):
        df["Date"] = df["Date"].astype(str)

    # ---- build trend slice: last 15 days -----------
    # 1️⃣  ensure datetime dtype
    external_df["Date_dt"] = pd.to_datetime(external_df["Date"], errors="coerce")
    external_df = external_df.dropna(subset=["Date_dt"])

    # 2️⃣  calendar Year-Week label
    external_df["YearWeek"] = external_df["Date_dt"].dt.strftime("%Y-%U")
    
    # 3️⃣  get last 15 days
    last_15_days = (
        external_df.sort_values("Date_dt")
                ["Date_dt"]
                .drop_duplicates()
                .tail(15)
                .tolist()
    )

    # 4️⃣  restrict to those days
    recent_ext = external_df[external_df["Date_dt"].isin(last_15_days)]

    # 5️⃣  choose numeric columns
    numeric_cols = recent_ext.select_dtypes(include=["number"]).columns

    # 6️⃣  format data for daily analysis - FIX: Convert Date_dt to string
    trends_df = (
        recent_ext
            .groupby(["Product_Name","Category", "Region", "Platform", "Date_dt"])[numeric_cols]
            .first()
            .reset_index()
    )
    
    # CRITICAL FIX: Convert Timestamp to string before creating dict
    trends_df["Day"] = trends_df["Date_dt"].dt.strftime("%Y-%m-%d")
    trends_df = trends_df.drop(columns=["Date_dt"])  # Remove the Timestamp column
    trends_df = trends_df[['Product_Name','Category', 'Region', 'Platform', 'Google_Trends_Score', 'Day']]
    trends_data = (
        trends_df
            .sort_values(["Product_Name","Category", "Region", "Platform", "Day"])
            .to_dict(orient="records")
    )
    # print(trends_data,"=================================")
    # Data for rising queries - FIX: Ensure Date is string
    risingquery_df = (
        external_df
        .groupby(["Product_Name", "Date"])["Google_Trends_Score"]
        .mean()
        .reset_index()
        .sort_values(by="Date", ascending=True)
    )
    
    # Ensure Date column is string (should already be, but double-check)
    risingquery_df["Date"] = risingquery_df["Date"].astype(str)
    risingquery_daily = risingquery_df.to_dict(orient="records")

    # Sales data - FIX: Ensure Date is string
    sales_df = (
        internal_df.groupby(["Product_Name", "Date"])["Units_Sold"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )
    sales_df["Date"] = sales_df["Date"].astype(str)
    sales_data = sales_df.to_dict(orient="records")

    # Combined data - FIX: Ensure Date is string
    combined = internal_df.merge(
        external_df,
        on=["Product_ID", "Product_Name", "Category", "Region", "Date"],
        how="inner"
    )

    combined_df = (
        combined
        .groupby(["Product_Name", "Date"])
        .agg({
            "Units_Sold":             "sum",
            "Social_Media_Mentions":  "sum",
            "Google_Trends_Score":    "mean",
        })
        .reset_index()
        .sort_values("Date")
    )
    combined_df["Date"] = combined_df["Date"].astype(str)
    combined_data = combined_df.to_dict(orient="records")

    # Category region data - FIX: Ensure Date is string
    category_region_df = (
        internal_df.groupby(["Category", "Region", "Date"])["Units_Sold"]
        .sum()
        .reset_index()
    )
    category_region_df["Date"] = category_region_df["Date"].astype(str)
    category_region_data = category_region_df.to_dict(orient="records")

    # Convert NumPy types → native for safe JSON serialisation
    def convert(x):
        if isinstance(x, (np.integer, np.int64)):
            return int(x)
        if isinstance(x, (np.floating, np.float64)):
            return float(x)
        if isinstance(x, pd.Timestamp):  # ADD: Handle Timestamp objects
            return x.strftime("%Y-%m-%d")
        if isinstance(x, list):
            return [convert(v) for v in x]
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        return x

    market_data = {
        "trends_data":           convert(trends_data),
        "sales_data":            convert(sales_data),
        "combined_data":         convert(combined_data),
        "category_region_data":  convert(category_region_data),
        "risingquery_daily":     convert(risingquery_daily),
    }

    print(f"✅ Prepared: {len(trends_data)} trend rows, {len(sales_data)} sales rows")
    return market_data
# ---------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------
def run_parallel_analysis(market_data: Dict[str, Any], runner: Runner):
    """
    Push one JSON payload to ParallelAgent and stream all responses.
    """
    content = types.Content(
        role="user",
        parts=[types.Part(text=json.dumps(market_data, indent=0))]
    )

    print("\n" + "=" * 60)
    print("STARTING PARALLEL MARKET ANALYSIS")
    print("=" * 60)

    responses: List[str] = []
    for event in runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    ):
        # final agent outputs come as ADK events with content.parts
        if event.is_final_response():
            responses.append(event.content.parts[0].text)

    return responses
import json
import re



def run_trend_insights(trend_results: Dict[str, Any]) -> str:
    """
    Run the trend insights agent separately with the calculated results
    """
    session_service = setup_session()
    runner = Runner(agent=trend_insights_agent, app_name=APP_NAME, session_service=session_service)
    
    content = types.Content(
        role="user",
        parts=[types.Part(text=json.dumps({"trend_results": trend_results}, indent=0))]
    )

    print("\n" + "=" * 40)
    print("GENERATING TREND INSIGHTS")
    print("=" * 40)

    responses = []
    for event in runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    ):
        if event.is_final_response():
            responses.append(event.content.parts[0].text)
    
    return responses[0] if responses else "{}"

def run_agent_pipeline():
    import os
    import json

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY not found – aborting.")
    genai.configure(api_key=api_key)

    market_data = prepare_market_data()
    # Step 1: Calculate trends using Python function
    print("🔢 Computing trends using Python function...")
    trend_results = compute_product_region_platform_trends(market_data["trends_data"])
    # print("================trend_results=========================\n",trend_results)
    print(f"✅ Calculated trends for {len(trend_results['trend_data'])} combinations")
    session_service = setup_session()
    runner = Runner(agent=parallel_agent, app_name=APP_NAME, session_service=session_service)

    responses = run_parallel_analysis(market_data, runner)
    # Step 3: Generate trend insights using the calculated results
    trend_insights_response = run_trend_insights(trend_results)
    parsed_responses = {}
    # Add the Python-calculated trend data
    parsed_responses["trend_analysis_15day"] = trend_results

    for idx, text in enumerate(responses, 1):
        clean_text = None
        try:
            clean_text = json.loads(text.strip('```json\n').rstrip('```'))
            if clean_text:
                parsed_responses.update(clean_text)
                print(f"✅ Parsed response #{idx} successfully")
        except Exception as e:
            print(f"⚠️ Failed to parse live response #{idx}: {e}")

        # Save live response
        # response_filename = f'agent_response_{idx}.json'
        # with open(response_filename, 'w', encoding='utf-8') as json_file:
        #     json.dump(clean_text or {}, json_file, indent=2)

        # Fallback if live parse failed
        if not clean_text:
            backup_file = f'backup/agent_response_{idx}.json'
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        backup_data = json.load(f)
                        parsed_responses.update(backup_data)
                        print(f"✅ Fallback used for agent #{idx}")
                except Exception as e:
                    print(f"❌ Backup parse failed for agent #{idx}: {e}")
            else:
                print(f"🚫 No backup available for agent #{idx}")
    

    try:
        trend_insights_data = json.loads(trend_insights_response.strip('```json\n').rstrip('```'))
        parsed_responses.update(trend_insights_data)
        print("✅ Parsed trend insights successfully")
    except Exception as e:
        print(f"⚠️ Failed to parse trend insights: {e}")

    # # Save trend insights separately
    # with open('trend_insights_response.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(trend_insights_data if 'trend_insights_data' in locals() else {}, json_file, indent=2)
    # # Save full merged response
    # with open('parsed_response.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(parsed_responses, json_file, indent=2)

    print("\n" + "=" * 60)
    print("PARALLEL ANALYSIS RESULTS")
    print("=" * 60)
    for idx, text in enumerate(responses, 1):
        print(f"\n{'*' * 40}\nAgent Output #{idx}\n{'*' * 40}")
        print(text[:1500] + ("..." if len(text) > 1500 else ""))

    return parsed_responses


