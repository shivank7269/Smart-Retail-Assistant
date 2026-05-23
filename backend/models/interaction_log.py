from pydantic import BaseModel, Field
from datetime import datetime

class InteractionLog(BaseModel):
    intent: str
    query: str
    response: str

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now().replace(microsecond=0)
    )
