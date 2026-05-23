import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from backend.rag.vectorstore import load_vectorstore

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01",
    temperature=0.2
)

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Smart Retail Assistant for an Indian retail chain.\n\n"
               "Provide a clear, structured, and easy-to-read answer using ONLY the provided context.\n"
               "Use bullet points if listing multiple conditions, policies, or rules.\n"
               "If the exact answer is not contained in the context, output exactly: 'I do not have information on that.'\n\n"
               "Context:\n{context}"),
    ("human", "{question}")
])

def rag_answer(question: str) -> str:
    try:
        docs = retriever.invoke(question)

        print("\n========== RETRIEVED DOCS ==========\n")

        for i, doc in enumerate(docs):
            print(f"\n--- DOC {i+1} ---")
            print(doc.page_content)

        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.format(
            context=context,
            question=question
        )

        response = llm.invoke(final_prompt)

        return response.content

    except Exception as e:
        return f"Error processing request: {str(e)}"

if __name__ == "__main__":
    query = "What is the return policy ?"
    answer = rag_answer(query)
    print(answer)