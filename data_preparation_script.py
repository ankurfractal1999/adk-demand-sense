import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_daily_ui_data(num_products: int = 10,
                           days_back: int = 15,
                           seed: int = 789):
    """Generate day-level internal & external synthetic data for a curated
    product→category mapping. Returns the last `days_back` days (rolling window)
    for every (product, region) combination.

    Columns exactly match the final UI schema. Rerun daily to refresh dates.
    """

    np.random.seed(seed)

    # Curated master mapping (first 10 entries kept) -------------------------
    curated_products = [
        # Food (2)
        ('Ice Bar', 'Food'), ('Keto Bar', 'Food'),
        # Beauty (2)
        ('Eye Mask', 'Beauty'), ('Natural Deo', 'Beauty'),
        # Fitness (2)
        ('Yoga Mat', 'Fitness'), ('Fitness Band', 'Fitness'),
        # Electronics (2)
        ('Earbuds', 'Electronics'), ('Smartwatch', 'Electronics'),
        # Household (2)
        ('Detergent', 'Household'), ('Floor Cleaner', 'Household')
    ]

    # Trim / extend to requested num_products
    if num_products <= len(curated_products):
        curated_products = curated_products[:num_products]
    else:
        extra = num_products - len(curated_products)
        curated_products += [(f'Product_{i}', 'Misc') for i in range(extra)]

    product_names, categories = zip(*curated_products)

    # Date range: last `days_back` days up to today --------------------------
    today  = datetime.now().date()
    dates  = [today - timedelta(days=i) for i in range(days_back-1, -1, -1)]  # chronological

    regions = ['North', 'South', 'East', 'West']

    internal_records, external_records = [], []

    for idx, (name, cat) in enumerate(curated_products):
        pid        = f"UIDPROD{idx+1:03d}"
        base_price = np.random.randint(100, 5000)

        for region in regions:
            for d in dates:
                # -------- Internal data row --------
                discount = np.random.choice([0, 5, 10, 15, 20])
                internal_records.append({
                    'Product_ID': pid,
                    'Product_Name': name,
                    'Category': cat,
                    'Region': region,
                    'Date': d,
                    'Base_Price_INR': base_price,
                    'Current_Discount_%': discount,
                    'Discounted_Price_INR': round(base_price * (1 - discount / 100)),
                    'Inventory_Level': np.random.randint(50, 500),
                    'Units_Sold': np.random.randint(50, 100),
                    'Restocked_Units': np.random.choice([0, 50, 100]),
                    'Avg_Customer_Rating': round(np.random.uniform(3.0, 5.0), 1),
                    'Number_of_Reviews': np.random.randint(5, 500),
                    'Return_Rate_%': round(np.random.uniform(0.5, 10.0), 2),
                    'Fulfillment_Center': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 'Kolkata'])
                })

                # -------- External data row --------
                trend_val = np.random.choice([50, 33, 28, 67, 124, -8, -15],
                                             p=[0.2, 0.15, 0.1, 0.1, 0.1, 0.2, 0.15])
                external_records.append({
                    'Product_ID': pid,
                    'Product_Name': name,
                    'Category': cat,
                    'Region': region,
                    'Date': d,
                    'Social_Media_Mentions': np.random.randint(100, 10000),
                    'Social_Media_Sentiment': round(np.random.uniform(-1, 1), 2),
                    'Trending_Hashtags': np.random.choice(['#TrendingNow', '#ViralProduct', '#ShopNova', '#HotPick']),
                    'Competitor_Price_INR': base_price + np.random.randint(-500, 500),
                    'Competitor_Promotion': np.random.choice(['None', 'Buy 1 Get 1', 'Cashback ₹200', 'Free Shipping']),
                    'Google_Trends_Score': np.random.randint(10, 100),
                    'Influencer_Marketing': np.random.choice(['Yes', 'No'], p=[0.4, 0.6]),
                    'Influencer_Sentiment_Score': round(np.random.uniform(0.5, 1.0), 2),
                    'Economic_Confidence_Index': round(np.random.uniform(85.0, 120.0), 2),
                    'Seasonal_Factor': np.random.choice(['Diwali', 'Monsoon', 'New Year', 'Back to School', 'None']),
                    'Platform': np.random.choice(['Instagram', 'TikTok', 'Facebook', 'X']),
                    'Percent_Trend': trend_val
                })

    internal_df = pd.DataFrame(internal_records)
    external_df = pd.DataFrame(external_records)
    return internal_df, external_df


# internal_df, external_df = generate_daily_ui_data()
# internal_df.to_csv('15days_internal_data.csv',index=False)
# external_df.to_csv('15days_external_data.csv',index=False)