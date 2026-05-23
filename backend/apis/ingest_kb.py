from fastapi import APIRouter, UploadFile, File, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.rag.vectorstore import load_vectorstore
import os

router = APIRouter()


@router.post("/upload")
async def upload_knowledge_document(file: UploadFile = File(...)):
    """Uploads text documents and embeds them directly into the Chroma Vector DB."""
    if not file.filename.endswith('.txt'):
        # For PDFs, you would use PyPDFLoader here. Keeping it to .txt for simplicity.
        raise HTTPException(status_code=400, detail="Currently only .txt files are supported for KB ingestion.")

    try:
        contents = await file.read()
        text = contents.decode("utf-8")

        # Split the new document into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(text)

        # Add to existing ChromaDB
        vectorstore = load_vectorstore()
        vectorstore.add_texts(texts=chunks)

        return {
            "message": f"Knowledge base updated successfully with {file.filename}.",
            "chunks_embedded": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))