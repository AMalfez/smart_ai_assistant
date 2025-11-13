from langchain.tools import tool
from chroma import vector_store
from llm import model

@tool
def get_summary(query: str):
    """Given the context, Get a concise summary for a given query."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    combined_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
    Summarize the following text based on the query: "{query}".
    Focus only on relevant details, concise and clear.
    ---
    {combined_text}
    """
    response = model.invoke(prompt)
    return response.content

@tool
def do_math(query: str):
    """
    Retrieves relevant docs from vectordb, extracts numeric facts, and computes the answer.
    Returns step-by-step computation and the final numeric value.
    """
    retrieved_docs = vector_store.similarity_search(query)
    combined_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    You are an assistant that extracts numeric facts from text and performs precise numeric calculations.

    User question:
    {query}

    Documents (only use numeric facts from these to compute):
    {combined_text}

    Tasks (strictly do these, in order):
    1) Extract numeric facts from the documents and present them as a JSON array named "facts".
    - Each fact should be an object with fields: "text", "value", "unit" (unit can be null), "context".
    - Example: {{ "text": "Out of 100 students, 80 were present", "value": 80, "unit": "students", "context": "present" }}

    2) Using only those facts, perform the arithmetic or reasoning needed to answer the user question.
    - Show every step of the arithmetic (e.g., "80 * 0.40 = 32").
    - If percentages are involved, convert them properly (e.g., 40% -> 0.40).

    3) Provide a clear final answer labeled "Final Answer:" followed by a single numeric result and unit if applicable.
    - Also include the minimal expression used (e.g., "80 * 0.40 = 32").

    Important constraints:
    - Only use facts that are actually present in the documents above.
    - If necessary facts are missing, state clearly which fact is missing and why you cannot compute the final number.
    - Do not hallucinate extra facts.

    Output format (important — produce valid JSON + steps + final answer):
    ----
    FACTS_JSON:
    <json array here>

    STEPS:
    1) ...
    2) ...

    Final Answer: <number> <unit or empty>
    ----
    Now, perform the task.
    """
    resp = model.invoke(prompt)
    return resp.content
