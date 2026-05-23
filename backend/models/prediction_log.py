from pydantic import BaseModel, Field
from datetime import datetime
class PredictionLog(BaseModel):
    prediction_type: str
    input_data: str
    prediction_result: str

    timestamp: datetime = Field( # field autogenerate fresh timestamp
        default_factory=lambda: datetime.now().replace(microsecond=0)
    )