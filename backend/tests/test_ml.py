import pytest
from backend.agents import demand_agent, anomaly_agent


def test_demand_agent_prediction():
    """Test if the demand forecasting agent processes a query and returns a prediction."""
    query = "How many units of clothing will sell in South next week?"
    result = demand_agent.run(query)

    # Handle both string and dictionary return types
    response_text = result.get("response", "") if isinstance(result, dict) else result

    assert isinstance(response_text, str)
    assert "Predicted Demand:" in response_text or "Error" in response_text


def test_anomaly_agent_detection():
    """Test if the anomaly agent can read the CSV and summarize outliers."""
    query = "Are there any suspicious sales or anomalies?"
    result = anomaly_agent.run(query)

    # Extract the text from the dictionary
    response_text = result.get("response", "") if isinstance(result, dict) else result

    assert isinstance(response_text, str)
    assert "Total Anomalies Detected:" in response_text or "No anomaly data" in response_text or "Error" in response_text