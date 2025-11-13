from langchain_chroma import Chroma
from embeddings import embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)

def AddDocument(docs):
    final_documents = text_splitter.split_documents(docs)
    vector_store.add_documents(final_documents)