import os
import pandas as pd
import logging
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATA_PATH = "data/anomalies.csv"

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    temperature=0.1
)


def run(query: str) -> str:
    try:
        # Step 1: Read the anomaly data from the CSV file
        if not os.path.exists(DATA_PATH):
            return "No anomaly data found. Please run the anomaly detection model first."

        anomaly_df = pd.read_csv(DATA_PATH)
        total_anomalies = len(anomaly_df)

        # Grab a small sample of the most severe outliers for the LLM context
        sample_data = anomaly_df[["Category of Goods", "Region", "Sales", "Profit"]].head(5).to_string(index=False)

        # Step 2: Generate concise insight
        insight_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an ML Expert. The system found {total} anomalies. Here is a sample:\n{sample}\n\n under stand the question then give answer accordingly"
             "if asked for some data then give him some data from sample else if asked for analysis analyse the data then give answer"),
            ("human", "{query}")
        ])

        insight = llm.invoke(
            insight_prompt.format(total=total_anomalies, sample=sample_data, query=query)).content.strip()

        return f"Total Anomalies Detected: {total_anomalies}\n{insight}"

    except Exception as e:
        logger.error(f"Anomaly Agent Error: {e}")
        return f"Anomaly Agent Error: {str(e)}"