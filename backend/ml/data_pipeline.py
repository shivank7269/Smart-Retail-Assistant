import pandas as pd
import os

def load_and_clean(path="data/retail_data.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file {path} not found")

    # read csv
    df = pd.read_csv(path)
    print(f"Data Pipeline: Raw dataset shape is {df.shape}")

    # All useful columns needed for both Demand and Anomaly models
    useful_cols = [
        "Sales",
        "Quantity",
        "Discount",
        "Profit",
        "Outlet Type",
        "City Type",
        "Category of Goods",
        "Region",
        "State",
        "Segment",
        "Ship Mode",
        "Sub-Category",
        "Order Date",
    ]

    useful_cols = [c for c in useful_cols if c in df.columns]
    df = df[useful_cols]

    # convert dates
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=False, errors="coerce")

    # Remove missing rows
    df.dropna(inplace=True)

    # remove invalid rows
    df = df[df["Sales"] > 0]
    df = df[df["Quantity"] > 0]

    # extract time feature from Order Date
    df["month"]       = df["Order Date"].dt.month
    df["day_of_week"] = df["Order Date"].dt.dayofweek
    df["quarter"]     = df["Order Date"].dt.quarter
    df["year"]        = df["Order Date"].dt.year

    # encode categories
    encode_cols = [
        "Outlet Type",
        "City Type",
        "Category of Goods",
        "Region",
        "State",
        "Segment",
        "Ship Mode",
        "Sub-Category",
    ]

    encode_cols = [c for c in encode_cols if c in df.columns]

    for col in encode_cols:
        df[col + "_enc"] = df[col].astype("category").cat.codes

    os.makedirs("data", exist_ok=True)

    print(f"Data Pipeline: Cleaned dataset shape is {df.shape}")
    return df

if __name__ == "__main__":
    cleaned_df = load_and_clean()
    cleaned_df.to_csv("data/clean_retail.csv", index=False)
    print("Data Pipeline: Saved clean_retail.csv successfully.")