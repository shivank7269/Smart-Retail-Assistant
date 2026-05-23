import os
import sys
import warnings
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from backend.agents import demand_agent, qa_agent, anomaly_agent

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    temperature=0.1
)

class RouteDecision(BaseModel):
    agent: str = Field(description="Must be strictly one of: 'demand', 'qa', 'anomaly'")

def run(query: str) -> dict:
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", "Route the user query to the most appropriate agent based on these strict rules:\n"
                   "- demand: For queries about numbers, historical sales, forecasting, or demand predictions.\n"
                   "- qa: For queries about store policies, rules, returns, shipping, or general text knowledge.\n"
                   "- anomaly: For queries about unusual transactions, outliers, fraud, or anomalies."),
        ("human", "{query}")
    ])

    structured_llm = llm.with_structured_output(RouteDecision)

    try:
        decision = structured_llm.invoke(router_prompt.format(query=query))

        if decision.agent == "demand":
            response = demand_agent.run(query)
        elif decision.agent == "qa":
            response = qa_agent.run(query)
        elif decision.agent == "anomaly":
            response = anomaly_agent.run(query)
        else:
            response = "Orchestrator failed to route the request."
            decision.agent = "unknown"

        return {
            "intent": decision.agent,
            "query": query,
            "response": response
        }

    except Exception as e:
        return {
            "intent": "error",
            "query": query,
            "response": f"Orchestration Error: {str(e)}"
        }


if __name__ == "__main__":
    queries = [
        "What is the return policy?",
        "How many units of clothing will sell in South next week?",
        "Are there any suspicious sales or anomalies?"
    ]

    for q in queries:
        # Save the dictionary returned by the orchestrator
        result = run(q)

        # Extract and print just the query and the response text
        print(f"Question: {result['query']}")
        print(f"Answer: {result['response']}\n")
        print("-" * 50)  # Optional: adds a dividing line between Q&As