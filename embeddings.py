import os
from langchain_huggingface import HuggingFaceEmbeddings
os.environ["HF_TOKEN"] = "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"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")