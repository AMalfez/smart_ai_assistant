from langchain.tools import tool
from chroma import vector_store
from llm import model
from langgraph.config import get_stream_writer

@tool
def get_summary(query: str):
    """Given the context, Get a concise summary for a given query."""
    writer = get_stream_writer()
    writer('🔍 Searching relevant sections in ChromaDB...\n')
    retrieved_docs = vector_store.similarity_search(query, k=4)
    combined_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    writer(f'📚 Retrieved {len(combined_text)} chunks\n')
    prompt = f"""
    Summarize the following text based on the query: "{query}".
    Focus only on relevant details, concise and clear.
    ---
    {combined_text}
    """

    writer('🧩 Using LangChain agent to summarize...\n')
    response = model.invoke(prompt)
    text = response.content
    writer(f"💬 Answer: {text}\n")

