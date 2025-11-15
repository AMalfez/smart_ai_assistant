from langchain.tools import tool, ToolRuntime
from chroma import vector_store
from llm import model
from langgraph.config import get_stream_writer

@tool
def get_summary(query: str, runtime: ToolRuntime):
    """Given the context, Get a concise summary for a given query."""
    writer = get_stream_writer()
    writer('🔍 Searching relevant sections in ChromaDB...\n')
    writer(f'{runtime.state['messages']}\n')
    retrieved_docs = vector_store.similarity_search(query, k=4)
    if not retrieved_docs:
        writer("⚠️ No relevant documents found.\n")
        
    combined_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    writer(f'📚 Retrieved {len(retrieved_docs)} chunks\n')
    prompt = f"""
    Summarize the following text based on the query: "{query}".
    Focus only on relevant details, concise and clear.
    ---
    context:{combined_text}
    previous_conversations:{runtime.state['messages']}
    """

    writer('🧩 Using LangChain agent to summarize...\n')
    response = model.invoke(prompt)
    text = response.content
    writer(f"💬 Answer: {text}\n")

