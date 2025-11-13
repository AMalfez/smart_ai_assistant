from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# for structured output
json_schema = {
    "type": "object",
    "description": "Response format of final answer after using MathTool.",
    "properties": {
        "final_answer": {
            "type": "string",
            "description": "The final numeric answer computed by MathTool, including unit if applicable.",
        }
    },
    "required": ["final_answer"],
}


model = init_chat_model("google_genai:gemini-2.5-flash-lite")

model_with_structure = model.with_structured_output(
    json_schema,
    method="json_schema",
)
