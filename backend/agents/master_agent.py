import os
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate

from backend.agents.demand_agent import run as run_demand
from backend.agents.anomaly_agent import run as run_anomaly
from backend.agents.qa_agent import run as run_qa
from backend.agents.data_agent import run_data_exploration

@tool
def predict_demand_tool(query: str) -> str:
    """Use THIS tool ONLY to PREDICT or FORECAST FUTURE demand. Do NOT use this to count existing rows, past sales, or dataset size."""
    raw_result = run_demand(query)
    if isinstance(raw_result, dict):
        return raw_result.get("response", str(raw_result))
    return str(raw_result)

@tool
def detect_anomalies_tool(query: str) -> str:
    """Use THIS tool to find outliers, unusual transactions, and anomalies."""
    raw_result = run_anomaly(query)
    if isinstance(raw_result, dict):
        return raw_result.get("response", str(raw_result))
    return str(raw_result)

@tool
def retail_policy_qa_tool(query: str) -> str:
    """Use THIS tool to answer questions about store policies, returns, and shipping."""
    raw_result = run_qa(query)
    if isinstance(raw_result, dict):
        return raw_result.get("response", str(raw_result))
    return str(raw_result)

@tool
def explore_dataset_tool(query: str) -> str:
    """Use THIS tool to count rows, calculate historical sales, or answer questions like 'how many are in the dataset'."""
    return run_data_exploration(query)

tools = [
    predict_demand_tool,
    detect_anomalies_tool,
    retail_policy_qa_tool,
    explore_dataset_tool
]

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    temperature=0.1
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Smart Retail Assistant. You must choose the correct tool based strictly on the tool descriptions. Do not guess or use the prediction tool for existing dataset queries."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)