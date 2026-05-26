import os
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


def run_data_exploration(query: str) -> str:
    df = pd.read_csv("data/clean_retail.csv")

    # Updated LLM initialization with all required Azure credentials
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-15-preview",
        temperature=0.0
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        agent_type="openai-tools",
        allow_dangerous_code=True
    )

    result = agent.invoke({"input": query})
    return result["output"]