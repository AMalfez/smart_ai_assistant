from langchain.agents import create_agent
from tools.summarizer import get_summary
from tools.mathtool import do_math
from dotenv import load_dotenv
from llm import model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware

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

agent = create_agent(model, tools, middleware=[SummarizationMiddleware(model='google_genai:gemini-2.5-flash-lite', max_tokens_before_summary=100, messages_to_keep=2)], system_prompt=prompt, checkpointer=InMemorySaver())