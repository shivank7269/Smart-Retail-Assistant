from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io
import os

router = APIRouter()
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")


@router.post("/upload")
async def upload_retail_csv(file: UploadFile = File(...)):
    """Uploads and saves raw retail transaction CSV data."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted for retail data.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Save to the data folder for the ML pipeline to process later
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, "retail_data.csv")
        df.to_csv(file_path, index=False)

        return {
            "message": "Retail data uploaded successfully.",
            "rows_ingested": len(df),
            "next_steps": "Run the data_pipeline to clean this data before ML training."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))