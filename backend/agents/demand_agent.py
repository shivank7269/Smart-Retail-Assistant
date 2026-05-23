import os
import pickle
import logging
import numpy as np
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# BULLETPROOF PATHS
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "ml")
MODEL_PATH = os.path.join(ML_DIR, "trained_models", "demand_model.pkl")

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    temperature=0.1
)


def load_ml_assets():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def run(query: str) -> str:
    try:
        model = load_ml_assets()

        # Azure-Compliant Extraction Prompt
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful data processing assistant. Please extract or estimate the following 16 numerical features from the user's query: month, quarter, week_of_year, day_of_month, day_of_week, is_weekend, festival_pressure, Category of Goods_enc, avg_discount, avg_sales, lag_1, lag_7, lag_14, lag_30, rolling_mean_7, rolling_mean_30. "
             "If a value is missing, please use these defaults: 6, 2, 24, 15, 1, 0, 0, 0, 0.1, 500, 50, 45, 45, 40, 48, 45. "
             "Format your response as a single comma-separated list of 16 numbers with no additional text."),
            ("human", "{query}")
        ])

        raw_extraction = llm.invoke(extract_prompt.format(query=query)).content.strip()

        try:
            parts = [float(x.strip()) for x in raw_extraction.split(",")]
            if len(parts) != 16:
                raise ValueError
            features = parts
        except:
            features = [6, 2, 24, 15, 1, 0, 0, 0, 0.1, 500, 50, 45, 45, 40, 48, 45]

        raw_features = [features]

        predicted_val = round(max(0.0, float(np.expm1(model.predict(raw_features)[0]))), 1)

        # Azure-Compliant Insight Prompt
        insight_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a retail data analyst. The ML model predicts {predicted_val} units of demand based on the user's query. "
             "Please provide a brief, 1-to-2 sentence business insight summarizing this prediction. Keep it concise and professional."),
            ("human", "{query}")
        ])

        insight = llm.invoke(insight_prompt.format(predicted_val=predicted_val, query=query)).content.strip()

        return f"Predicted Demand: {predicted_val} units\n{insight}"

    except FileNotFoundError:
        return "Demand Agent Error: ML models not found. Please train the model first."
    except Exception as e:
        logger.error(f"Demand Agent Error: {e}")
        return f"Demand Agent Error: {str(e)}"