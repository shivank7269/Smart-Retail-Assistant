import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_demand():
    payload = {"query": "How many units of clothing will sell in South next week?"}
    response = client.post("/api/predict/demand", json=payload)
    assert response.status_code == 200, f"API crashed with: {response.text}"
    data = response.json()
    assert "prediction" in data

def test_predict_anomaly():
    payload = {"query": "Are there any suspicious sales or anomalies?"}
    response = client.post("/api/predict/anomaly", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data

def test_search_knowledge_base():
    payload = {"query": "What is the return policy for electronics?"}
    response = client.post("/api/search/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data

def test_agent_chat():
    payload = {"message": "What is the return policy for laptops?"}
    response = client.post("/api/agent/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "response" in data