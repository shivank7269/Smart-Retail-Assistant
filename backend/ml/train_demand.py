import pandas as pd
import numpy as np
import pickle
import os, json
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_demand():
    path = "data/clean_retail.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing. Run data pipeline first.")

    df = pd.read_csv(path)
    print(f"Loaded dataset shape: {df.shape}")

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    festival_months = [1, 3, 8, 10, 11, 12]

    # aggregate daily demand
    daily = df.groupby(
        ["Order Date", "Category of Goods_enc"]
    ).agg(
        total_qty=("Quantity", "sum"),
        avg_discount=("Discount", "mean"),
        avg_sales=("Sales", "mean")
    ).reset_index()

    print(f"Daily aggregated rows: {daily.shape}")

    # full date features
    daily["month"] = daily["Order Date"].dt.month
    daily["quarter"] = daily["Order Date"].dt.quarter
    daily["day_of_week"] = daily["Order Date"].dt.dayofweek
    daily["week_of_year"] = daily["Order Date"].dt.isocalendar().week.astype(int)
    daily["day_of_month"] = daily["Order Date"].dt.day
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["festival_pressure"] = daily["month"].isin(festival_months).astype(int)

    daily = daily.sort_values(["Category of Goods_enc", "Order Date"]).reset_index(drop=True)

    # full lag features
    grp = daily.groupby("Category of Goods_enc")["total_qty"]
    daily["lag_1"] = grp.shift(1)
    daily["lag_7"] = grp.shift(7)
    daily["lag_14"] = grp.shift(14)
    daily["lag_30"] = grp.shift(30)

    # full rolling averages
    daily["rolling_mean_7"] = grp.transform(lambda x: x.shift(1).rolling(7).mean())
    daily["rolling_mean_30"] = grp.transform(lambda x: x.shift(1).rolling(30).mean())

    daily.dropna(inplace=True)
    print(f"Rows after lag creation: {daily.shape}")

    # log transform target
    daily["total_qty"] = np.log1p(daily["total_qty"])

    features = [
        "month", "quarter", "week_of_year", "day_of_month", "day_of_week",
        "is_weekend", "festival_pressure", "Category of Goods_enc",
        "avg_discount", "avg_sales",
        "lag_1", "lag_7", "lag_14", "lag_30",
        "rolling_mean_7", "rolling_mean_30"
    ]
    target = "total_qty"

    X = daily[features].values
    y = daily[target].values

    split_index = int(len(daily) * 0.90)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # gradient boosting regression model
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # predictions — reverse log transform
    y_train_actual = np.expm1(y_train)
    y_test_actual = np.expm1(y_test)
    y_pred_train = np.expm1(model.predict(X_train))
    y_pred_test = np.expm1(model.predict(X_test))

    def calculate_metrics(label, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"[{label:>5}]  MAE:{mae:6.2f} | RMSE:{rmse:6.2f} | R²:{r2:+.4f}")
        return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}

    print("\n--- Demand Forecast Evaluation ---")
    train_metrics = calculate_metrics("Train", y_train_actual, y_pred_train)
    test_metrics = calculate_metrics("Test", y_test_actual, y_pred_test)

    os.makedirs("backend/ml/trained_models", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    # save model
    with open("backend/ml/trained_models/demand_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # save metrics
    metrics_summary = {
        "model": "GradientBoostingRegressor",
        "forecasting_type": "Demand Forecasting",
        "features": features,
        "train": train_metrics,
        "test": test_metrics
    }
    with open("docs/demand_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)

    # save feature importances (safely handled for GradientBoosting)
    if hasattr(model, 'feature_importances_'):
        importances = dict(zip(features, model.feature_importances_.round(4)))
        with open("docs/demand_feature_importance.json", "w") as f:
            json.dump(importances, f, indent=4)

    # visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(y_test_actual[:100], label="Actual", linewidth=1.5)
    axes[0].plot(y_pred_test[:100], label="Predicted", linewidth=1.5, linestyle="--")
    axes[0].set_title("Demand Forecast Accuracy")
    axes[0].legend()

    residuals = y_test_actual - y_pred_test
    axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Residual Error Distribution")

    plt.tight_layout()
    plt.savefig("docs/demand_forecast_plot.png", dpi=150)
    plt.close()

    print("Demand Engine: Forecast training completed successfully.")


if __name__ == "__main__":
    train_demand()