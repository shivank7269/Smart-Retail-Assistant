import pandas as pd
import pickle, os
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def train_anomaly():
    path = "data/clean_retail.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing. Run Data Pipeline First")

    df = pd.read_csv(path)

    # Uses the full set of financial features for high-quality anomaly detection
    features = ["Sales", "Quantity", "Discount", "Profit"]

    X = df[features].copy()

    # scales so that large ranges do not dominate the calculation of isolation forest
    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42  # maintain state for testing
            )
        )
    ])

    pipeline.fit(X)
    df["anomaly"] = pipeline.predict(X)

    df["anomaly_flag"] = df["anomaly"].apply(
        lambda x: 1 if x == -1 else 0
    )

    anomalies = df[df["anomaly_flag"] == 1]

    print(f"The number of anomalies is {len(anomalies)} out of {len(df)}")

    os.makedirs("backend/ml/trained_models", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    with open("backend/ml/trained_models/anomaly_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    # export anomalies in csv format for analysis
    anomalies.to_csv("data/anomalies.csv", index=False)

    # generate visualization
    plt.figure(figsize=(10, 4))
    plt.scatter(
        df.index,
        df["Sales"],
        c=df["anomaly_flag"],
        cmap="coolwarm",
        s=5,
        alpha=0.6
    )
    plt.title("Anomaly Detection - Sales Outlier Map")
    plt.xlabel("Record Index")
    plt.ylabel("Sales (INR)")
    plt.colorbar(label="Normal(0) vs Anomaly(1)")
    plt.savefig("docs/anomaly_plot.png")
    plt.close()

    print("Anomaly Engine: Plot and model successfully saved to disk.")


if __name__ == "__main__":
    train_anomaly()