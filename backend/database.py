import os
from datetime import datetime
from pymongo import MongoClient
from pydantic import BaseModel, Field

# 1. Pydantic Models

class PredictionLog(BaseModel):
    prediction_type: str
    input_data: str
    prediction_result: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now().replace(microsecond=0)
    )


class InteractionLog(BaseModel):
    intent: str
    query: str
    response: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now().replace(microsecond=0)
    )


# -----------------------------------
# 2. MongoDB Connection Setup
# -----------------------------------
_mongo_client = None
_db_instance = None


def get_database():
    global _mongo_client, _db_instance

    if _db_instance is not None:
        return _db_instance

    MONGO_URL = os.getenv('MONGO_URL')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'retail_database')

    if not MONGO_URL:
        raise ValueError("MONGO_URL is not set — ensure load_dotenv() is called in main.py before imports")

    _mongo_client = MongoClient(MONGO_URL)
    _db_instance = _mongo_client[DATABASE_NAME]
    _mongo_client.admin.command('ping')
    print(f"✅ Successfully connected to MongoDB -> DB: {DATABASE_NAME}")

    return _db_instance


# -----------------------------------
# 3. Database Operations (CRUD)
# -----------------------------------
def insert_prediction_log(log: PredictionLog) -> str:
    db = get_database()
    log_dict = log.model_dump()
    log_dict['log_type'] = 'prediction'
    result = db['logs'].insert_one(log_dict)
    return str(result.inserted_id)


def insert_interaction_log(log: InteractionLog) -> str:
    db = get_database()
    log_dict = log.model_dump()
    log_dict['log_type'] = 'interaction'
    result = db['logs'].insert_one(log_dict)
    return str(result.inserted_id)


# -----------------------------------
# 4. Quick Testing (run directly)
# -----------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # only needed when running this file standalone

    pred_id = insert_prediction_log(PredictionLog(
        prediction_type="demand",
        input_data='{"month": 6, "region": 1}',
        prediction_result='{"predicted_quantity": 45.2}'
    ))
    print(f"✅ Prediction inserted: {pred_id}")

    int_id = insert_interaction_log(InteractionLog(
        intent="qa",
        query="What is the return policy?",
        response="You have 30 days to return electronics."
    ))
    print(f"✅ Interaction inserted: {int_id}")