import pytest
from backend.rag.vectorstore import retrieve_context
from backend.rag.rag_chain import rag_answer


def test_vectorstore_retrieval():
    """Test if the vector database successfully finds relevant context."""
    query = "What is the return policy?"
    context = retrieve_context(query, k=2)

    # Should return a string containing the retrieved documents
    assert isinstance(context, str)

    # Even if no exact match is found, it shouldn't crash (might return empty string)
    assert context is not None


def test_rag_generation():
    """Test if the LLM successfully reads the context and generates an answer."""
    query = "What is the return policy for electronics?"
    answer = rag_answer(query)

    assert isinstance(answer, str)
    assert len(answer) > 0