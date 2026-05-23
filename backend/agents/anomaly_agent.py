import os
import pandas as pd
import logging
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

DATA_PATH = "data/anomalies.csv"

RELEVANT_COLUMNS = [
    "Sales", "Quantity", "Discount", "Profit",
    "Outlet Type", "City Type", "Category of Goods",
    "Region", "State", "Segment", "Ship Mode",
    "Sub-Category", "Order Date", "month", "quarter", "year"
]


def get_llm():
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-15-preview",
        temperature=0.1
    )


def run(query: str) -> dict:
    try:
        if not os.path.exists(DATA_PATH):
            return {
                "intent": "anomaly_detection",
                "query": query,
                "response": "No anomaly data found. Please run the anomaly detection model first."
            }

        anomaly_df = pd.read_csv(DATA_PATH)

        # Keep only anomalies (anomaly_flag == 1)
        anomaly_df = anomaly_df[anomaly_df["anomaly_flag"] == 1]
        total_anomalies = len(anomaly_df)

        # Use only human-readable columns
        display_df = anomaly_df[RELEVANT_COLUMNS].reset_index(drop=True)

        # Pass top 100 rows so LLM has enough data to filter/answer correctly
        sample_data = display_df.head(100).to_string(index=False)

        insight_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a retail anomaly detection expert. The ML model detected {total_anomalies} anomalies.

Available columns: Sales, Quantity, Discount, Profit, Outlet Type, City Type, Category of Goods, Region, State, Segment, Ship Mode, Sub-Category, Order Date, month, quarter, year

Anomaly data sample (up to 100 rows):
{sample_data}

Rules:
- Answer ONLY what the user asked, nothing more
- If user asks for data or rows → filter the above data and return matching rows in a clean table
- If user asks for analysis → give 2-3 bullet points max
- If user asks for count → just return the number
- If user asks about sales/quantity/profit greater or less than a value → filter and show those rows
- If user asks about a region/category/state → filter by that column and show results
- If a column does not exist in the data, say so clearly
- Never add unnecessary explanation or preamble
- Be concise and direct"""),
            ("human", "{query}")
        ])

        llm = get_llm()
        insight = llm.invoke(
            insight_prompt.format(
                total_anomalies=total_anomalies,
                sample_data=sample_data,
                query=query
            )
        ).content.strip()

        return {
            "intent": "anomaly_detection",
            "query": query,
            "response": f"Total Anomalies Detected: {total_anomalies}\n{insight}"
        }

    except Exception as e:
        logger.error(f"Anomaly Agent Error: {e}", exc_info=True)
        return {
            "intent": "anomaly_detection",
            "query": query,
            "response": f"Anomaly Agent Error: {str(e)}"
        }