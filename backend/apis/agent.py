from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from backend.agents.orchestrator import run
from backend.database import InteractionLog, insert_interaction_log,get_database

router = APIRouter()
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    message: str

# Route
@router.post("/chat")
def chat_with_assistant(data: ChatMessage):

    try:
        get_database()
        result = run(data.message)
        log_entry = InteractionLog(
            intent=result.get("intent", "unknown"),
            query=result.get("query", ""),
            response=result.get("response", "")
        )
        insert_interaction_log(log_entry)

        return result

    except Exception as e:
        logger.error(f"Agent Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))