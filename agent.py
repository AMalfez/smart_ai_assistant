import os
from langchain.agents import create_agent
from tools.summarizer import get_summary
from tools.mathtool import do_math
from dotenv import load_dotenv
from llm import model
load_dotenv()

tools = [get_summary, do_math]

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