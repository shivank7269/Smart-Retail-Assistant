from dotenv import load_dotenv
import os
from fastapi import FastAPI
from backend.apis import ingest_data, ingest_kb, predict, search, agent
import logging


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="Smart Retail Assistant API",
    description="Multi-agent backend for retail analytics, forecasting, and RAG.",
    version="1.0.0"
)

# Wire up all the separate routers
app.include_router(ingest_data.router, prefix="/api/data",    tags=["Retail Data"])
app.include_router(ingest_kb.router,   prefix="/api/kb",      tags=["Knowledge Base"])
app.include_router(predict.router,     prefix="/api/predict", tags=["ML Predictions"])
app.include_router(search.router,      prefix="/api/search",  tags=["Document Search"])
app.include_router(agent.router,       prefix="/api/agent",   tags=["Agent Chat"])

@app.get("/health", tags=["System"])
def health_check():
    """System health check endpoint."""
    return {"status": "healthy", "service": "Smart Retail Assistant API"}

@app.get("/")
def root():
    return {"message":"welcome to digital retail assistant"}