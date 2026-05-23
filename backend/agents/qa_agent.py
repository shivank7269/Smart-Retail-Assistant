import logging
from backend.rag.rag_chain import rag_answer

logger = logging.getLogger(__name__)

def run(query: str) -> str:
    try:
        answer = rag_answer(query)
        return answer.strip()
    except Exception as e:
        logger.error(f"QA Agent Error: {e}")
        return f"QA Agent Error: {str(e)}"