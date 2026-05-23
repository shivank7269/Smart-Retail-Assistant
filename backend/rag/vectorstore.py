from backend.rag.embeddings import get_embeddings, CHROMA_PATH
from langchain_chroma import Chroma


# Load Chroma Vector Database
def load_vectorstore():
    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vectorstore


# Retrieve Similar Context
def retrieve_context(query: str, k=1):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    context = ""
    for i, doc in enumerate(docs):
        context += f"\nDocument {i + 1}:\n{doc.page_content}\n"

    return context


# Testing
if __name__ == "__main__":
    print("Testing Vector Store Retrieval...\n")

    result = retrieve_context(
        "What is the return policy for electronics?"
    )

    print(result)