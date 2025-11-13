import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from chroma import AddDocument
from langchain.agents import create_agent
from tools import get_summary, do_math
from dotenv import load_dotenv
from llm import model
load_dotenv()

folder_path = "./example_data"
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        loader = PyPDFLoader(file_path)
        docs=loader.load()
        AddDocument(docs)
    
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        loader = TextLoader(file_path)
        docs=loader.load()
        AddDocument(docs)


tools = [get_summary, do_math]
# If desired, specify custom instructions
prompt = (
    """
You are a RAG assistant with two tools:
- get_summary(query)
- do_math(query)

Decide which tool(s) to call based on the user's question. If the question needs numeric computation from documents, prefer math_tool. If it is summarization, prefer summarize_tool. If trivial, answer directly.

User Query: {content}
"""
)

agent = create_agent(model, tools, system_prompt=prompt)

# if __name__ == "__main__":
# query = "calculate the number of boys present on saturday."
# res = agent.invoke({"messages": [{"role": "user", "content": query}]})
# final_output = res["messages"][-1].content
# print(final_output)

def answer_query(query: str):
    for chunk in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode='custom'):
        print(chunk)