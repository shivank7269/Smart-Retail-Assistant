from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.rag.rag_chain import rag_answer

router = APIRouter()

class SearchQuery(BaseModel):
    query: str

@router.post("/")
def search_knowledge_base(data: SearchQuery):
    """Directly queries the RAG knowledge base."""
    try:
        answer = rag_answer(data.query)
        return {"query": data.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))