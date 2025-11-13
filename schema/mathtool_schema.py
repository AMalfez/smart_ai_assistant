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