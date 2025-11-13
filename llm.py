from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from schema.mathtool_schema import json_schema

load_dotenv()

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

model_with_structure = model.with_structured_output(
    json_schema,
    method="json_schema",
)
