from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json

import logging
logger = logging.getLogger(__name__)

# Import your ML agents
from backend.agents import demand_agent, anomaly_agent

# Import your MongoDB models and insert function
from backend.database import PredictionLog, insert_prediction_log

router = APIRouter()


# -----------------------------------
# Request Schemas
# -----------------------------------
class DemandInput(BaseModel):
    query: str


class AnomalyInput(BaseModel):
    query: str


# -----------------------------------
# Routes
# -----------------------------------
@router.get("/metrics")
def get_ml_metrics():
    """Fetches the latest evaluation metrics for the ML models."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(current_dir, "..", "..", "docs", "demand_metrics.json")

    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")

    with open(metrics_path, "r") as f:
        return json.load(f)


@router.post("/demand")
def predict_demand(data: DemandInput):
    """Direct API endpoint for demand forecasting."""
    # 1. Run the ML model
    result = demand_agent.run(data.query)

    if "Error" in result:
        raise HTTPException(status_code=500, detail=result)

    # 2. Prepare the database log
    log_entry = PredictionLog(
        prediction_type="demand",
        input_data=data.model_dump_json(),
        prediction_result=result
    )

    # 3. Save to MongoDB
    insert_prediction_log(log_entry)

    return {"prediction": result}


@router.post("/anomaly")
def predict_anomaly(data: AnomalyInput):
    try:
        result = anomaly_agent.run(data.query)
        response_text = result.get("response", "")

        if "Error" in response_text:
            raise HTTPException(status_code=500, detail=response_text)

        log_entry = PredictionLog(
            prediction_type="anomaly",
            input_data=data.model_dump_json(),
            prediction_result=response_text
        )
        insert_prediction_log(log_entry)

        return {"prediction": response_text}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly Route Error: {e}", exc_info=True)  # ← full traceback
        raise HTTPException(status_code=500, detail=str(e))