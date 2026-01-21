import os
import glob
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src import config

def load_documents(data_dir: str) -> List:
    """
    Loads all PDF documents from the specified directory.
    """
    print(f"Loading documents from {data_dir}...")
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    if not documents:
        print("No PDF documents found in the data directory.")
        return []
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents: List) -> List:
    """
    Splits documents into smaller chunks.
    """
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def create_vector_db():
    """
    Main function to ingest data and create/save the vector database.
    """
    # 1. Load Documents
    documents = load_documents(config.DATA_DIR)
    if not documents:
        return

    # 2. Split Documents
    chunks = split_documents(documents)

    # 3. Initialize Embeddings
    print(f"Initializing embedding model: {config.EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}, # Force CPU for compatibility, change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True}
    )

    # 4. Create FAISS Index
    print("Creating FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 5. Save Index
    print(f"Saving FAISS index to {config.VECTOR_DB_DIR}...")
    vector_store.save_local(config.VECTOR_DB_DIR)
    print("Vector database created successfully!")

if __name__ == "__main__":
    create_vector_db()
