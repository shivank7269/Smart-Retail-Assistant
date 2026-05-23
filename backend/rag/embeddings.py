from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load configuration variables from .env
load_dotenv()

# Define where the vector database will be saved locally
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")


def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-01"
    )


def build_vectorstore(doc_path="docs/retail_knowledge.txt"):
    # 1. Verify the knowledge base exists
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Knowledge file {doc_path} not found. Create it first.")
    with open(doc_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # split the data
    # chunk_size of 300 and overlap of 50 ensures relevant paragraphs
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(raw_text)
    print(f"RAG Embeddings: Splitting documents into {len(chunks)} text chunks.")

    # 4. Initialize Azure OpenAI Embeddings
    embeddings = get_embeddings()

    # 5. Build and persist the vector store
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"RAG Embeddings: Chroma vectorstore successfully generated at {CHROMA_PATH}")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
