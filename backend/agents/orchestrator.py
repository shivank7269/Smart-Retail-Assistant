import os
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from backend.rag.rag_chain import rag_answer

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    temperature=0.1
)


def data_analyst_agent(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Data Analyst Agent for a retail store. Answer questions regarding historical sales data, revenue metrics, inventory counts, and SQL databases. Be precise and analytical."),
        ("human", "{query}")
    ])
    response = llm.invoke(prompt.format(query=query))
    return f"[Data Analyst Agent] {response.content}"


def ml_expert_agent(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an ML Expert Agent. Answer questions about predictive modeling, demand forecasting, anomaly detection, and AI insights. Explain complex ML concepts clearly."),
        ("human", "{query}")
    ])
    response = llm.invoke(prompt.format(query=query))
    return f"[ML Expert Agent] {response.content}"


def document_assistant_agent(query: str) -> str:
    answer = rag_answer(query)
    return f"[Document Assistant Agent] {answer}"


class RouteDecision(BaseModel):
    agent: str = Field(description="Must be strictly one of: 'data_analyst', 'document_assistant', 'ml_expert'")


def orchestrator(query: str) -> str:
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", "Route the user query to the most appropriate agent based on these strict rules:\n"
                   "- data_analyst: For queries about numbers, historical sales, revenue, databases, or inventory counts.\n"
                   "- document_assistant: For queries about store policies, rules, returns, shipping, or general text knowledge.\n"
                   "- ml_expert: For queries about future predictions, AI models, forecasting, or machine learning algorithms."),
        ("human", "{query}")
    ])

    structured_llm = llm.with_structured_output(RouteDecision)

    try:
        decision = structured_llm.invoke(router_prompt.format(query=query))

        if decision.agent == "data_analyst":
            return data_analyst_agent(query)
        elif decision.agent == "document_assistant":
            return document_assistant_agent(query)
        elif decision.agent == "ml_expert":
            return ml_expert_agent(query)
        else:
            return "Orchestrator failed to route the request."

    except Exception as e:
        return f"Orchestration Error: {str(e)}"


if __name__ == "__main__":
    queries = [
        "What is the return policy for electronics?",
        "What was our total revenue for the third quarter?",
        "Why did our demand forecasting model predict a drop in sales next week?"
    ]

    for q in queries:
        print(f"User: {q}")
        print(f"{orchestrator(q)}\n")