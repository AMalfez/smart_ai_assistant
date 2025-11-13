import os
import streamlit as st
import tempfile
from agent import agent
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from chroma import AddDocument

# --- Global Setup ---
st.set_page_config(page_title="RAG Stream App", page_icon="🤖", layout="wide")
st.title("📘 RAG Stream Demo with Gemini + HuggingFace + Tools")

# --- File Upload ---
uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded.")
    all_docs = []
    # Save uploaded files to temp dir for PyPDFLoader/TextLoader
    with tempfile.TemporaryDirectory() as tmp_dir:
        for file in uploaded_files:
            file_path = os.path.join(tmp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            else:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()

            all_docs.extend(docs)
            

    # Build embeddings + vector store
    st.info("🔄 Building vector store...")
    AddDocument(all_docs)
    st.success("✅ Knowledge base ready!")

    # --- Query Section ---
    query = st.text_area("Enter your question/query:")
    if st.button("🚀 Generate Response"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            st.write("### 🔎 Streaming Output")
            st_div = st.empty()

            # --- Stream ---
            full_output = ""
            for chunk in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="custom"):
                full_output += chunk
                st_div.markdown(f"```\n{full_output}\n```")

            st.success("✅ Done!")
