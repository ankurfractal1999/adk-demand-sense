import pandas as pd
import numpy as np
import asyncio
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import content_types
import os
from datetime import datetime, timedelta
load_dotenv()
os.getenv("GOOGLE_API_KEY")
# --- Constants ---
APP_NAME = "sales_simulation_agent"
USER_ID = "user_abc"
SESSION_ID = "session_123"
MODEL_ID = "gemini-2.0-flash"

# --- 1. Data Loading and Preparation ---

def load_and_prepare_data(internal_df, external_df):
    """
    Loads, merges, and prepares the internal and external data for simulation.
    """
    df = pd.merge(internal_df, external_df, on=['Product_ID','Product_Name','Category','Region', 'Date'])
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

# Global variable to hold our prepared data
DATA_DF = None
def run_sales_simulation(product_name: str, discount_percentage: float, social_amplification: float) -> dict:
    global DATA_DF
    if DATA_DF is None:
        return {"error": "Data not loaded. Please ensure the data files are available."}

    # Ensure 'Date' column is datetime
    DATA_DF['Date'] = pd.to_datetime(DATA_DF['Date'])

    # Filter rows for the product
    product_rows = DATA_DF[DATA_DF['Product_Name'].str.lower() == product_name.lower()]
    if product_rows.empty:
        return {"error": f"Product '{product_name}' not found in the dataset."}

    # Find latest date in data
    latest_date = product_rows['Date'].max()
    week_start = latest_date - timedelta(days=6)  # Last 7 days including latest

    # Filter for last 7 days
    last_week_data = product_rows[(product_rows['Date'] >= week_start) & (product_rows['Date'] <= latest_date)]
    if last_week_data.empty:
        return {"error": f"No data available for the last 7 days for product '{product_name}'."}

    # Aggregate current week metrics
    current_sales = int(last_week_data['Units_Sold'].sum())
    avg_base_price = last_week_data['Base_Price_INR'].mean()
    avg_sentiment = last_week_data['Social_Media_Sentiment'].mean()
    avg_trend_score = last_week_data['Google_Trends_Score'].mean()

    # Use latest row for cost assumptions
    latest_row = last_week_data.sort_values('Date', ascending=False).iloc[0]

    # --- Simulation Logic ---
    discount_effect = 1 + (discount_percentage / 100) * 1.5
    amplified_sentiment = avg_sentiment * (1 + social_amplification / 100)
    amplified_trends = avg_trend_score * (1 + social_amplification / 100)

    sentiment_effect = 1 + (amplified_sentiment - 0.5) * 0.2
    trends_effect = 1 + (amplified_trends / 100) * 0.1

    simulated_units_sold = int(current_sales * discount_effect * sentiment_effect * trends_effect)

    cost_per_unit = avg_base_price * 0.6
    discounted_price = avg_base_price * (1 - discount_percentage / 100)
    revenue = simulated_units_sold * discounted_price
    total_cost = simulated_units_sold * cost_per_unit
    profit = revenue - total_cost
    profit_margin = (profit / revenue) * 100 if revenue != 0 else 0

    current_profit_margin = ((avg_base_price - cost_per_unit) / avg_base_price) * 100

    sales_increase_pct = ((simulated_units_sold - current_sales) / current_sales) * 100 if current_sales else 0
    profit_margin_improvement = profit_margin - current_profit_margin

    return {
        "Sales Volume (Units)": {
            "Current": current_sales,
            "Week 1": simulated_units_sold,
            "Week 2": simulated_units_sold
        },
        "Profit Margin (%)": {
            "Current": round(current_profit_margin, 2),
            "Week 1": round(profit_margin, 2),
            "Week 2": round(profit_margin, 2)
        },
        "Impact Summary": f"{round(sales_increase_pct, 2)}% sales increase with {round(profit_margin_improvement, 2)}% profit margin improvement projected."
    }

# --- 2. Tool Definition ---

# Wrap the function with FunctionTool
sales_tool = FunctionTool(func=run_sales_simulation)

# --- 3. Agent Definition ---

sales_agent = Agent(
    model=MODEL_ID,
    name='SalesSimulator',
    instruction=(
        "You are a Sales Simulation Agent. Your task is to run sales simulations. "
        "When a user asks for a forecast, you MUST use the 'run_sales_simulation' tool. "
        "Extract the 'product_name', 'discount_percentage', and 'social_amplification' from the user's query. "
        "Example: for the query 'run a sales forecast for the 'Laptop' with a 20% discount and a 50% social media amplification', "
        "call the tool with product_name='Laptop', discount_percentage=20, and social_amplification=50. "
        "If any parameter is missing, ask the user for it."
    ),
    tools=[sales_tool]
)

# --- 4. Session and Runner Setup ---
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


# --- 5. Agent Interaction ---

def call_agent(query: str):
    """Handles sending a query to the agent and returns the simulation results dictionary."""
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    data_dir = "Data"

    # Load internal and external dataframes
    internal_df = pd.read_csv(os.path.join(data_dir, "internal_df.csv"))
    external_df = pd.read_csv(os.path.join(data_dir, "external_df.csv"))
    # Load data into the global variable
    global DATA_DF
    try:
        DATA_DF = load_and_prepare_data(internal_df, external_df)
    except FileNotFoundError:
        print(f"Error: Make sure data files are in the same directory.")
        return

    print(f"\nUser Query: '{query}'")
    print("-" * 20)
    
    # Parse query manually for parameters
    import re
    product_name_match = re.search(r"for our '([^']*)' product", query) or re.search(r"for the '([^']*)'", query)
    discount_percentage_match = re.search(r"gave a (\d+)% discount", query) or re.search(r"with a (\d+)% discount", query)
    social_amplification_match = re.search(r"only had (\d+)% social amplification", query) or re.search(r"and a (\d+)% social media amplification", query)

    if product_name_match and discount_percentage_match and social_amplification_match:
        # Direct simulation call - return the dictionary result
        product_name = product_name_match.group(1)
        discount_percentage = float(discount_percentage_match.group(1))
        social_amplification = float(social_amplification_match.group(1))
        
        result = run_sales_simulation(product_name, discount_percentage, social_amplification)
        print(result)  # Print for debugging
        return result  # Return the dictionary
    
    else:
        # Agent-based approach
        session_service = setup_session()
        runner = Runner(agent=sales_agent, app_name=APP_NAME, session_service=session_service)
        
        content = content_types.to_content(query)
        events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

        simulation_result = None
        
        for event in events:
            if event.is_final_response():
                final_response = event.content.parts[0].text
                print("Agent Response: ", final_response)
                # If no tool was called, return the text response
                if simulation_result is None:
                    return final_response
                else:
                    return simulation_result
                    
            elif event.type == "tool_output":
                # Capture the tool output (simulation result)
                tool_output = event.data
                print("Tool Output:", tool_output)
                
                # Parse the tool output to extract the simulation result
                # The tool output should contain the dictionary result
                if isinstance(tool_output, dict):
                    simulation_result = tool_output
                else:
                    # If it's a string representation, try to evaluate it
                    try:
                        import ast
                        simulation_result = ast.literal_eval(str(tool_output))
                    except:
                        simulation_result = tool_output
                        
            elif event.type == "tool_code":
                print("Tool Call:", event.data)
        
        # Return the simulation result if we got one
        return simulation_result if simulation_result is not None else "No simulation result obtained"


