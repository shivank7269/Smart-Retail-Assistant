from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import logging
from backend.agents.master_agent import agent_executor
from backend.database import insert_interaction_log, InteractionLog

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        result = agent_executor.invoke({"input": request.message})
        final_response = result["output"]

        log_entry = InteractionLog(
            intent="multi_agent",
            query=request.message,
            response=final_response,
            timestamp=datetime.utcnow()
        )

        insert_interaction_log(log_entry)

        return {
            "intent": "multi_agent",
            "query": request.message,
            "response": final_response
        }

    except Exception as e:
        logger.error(f"Chat Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))