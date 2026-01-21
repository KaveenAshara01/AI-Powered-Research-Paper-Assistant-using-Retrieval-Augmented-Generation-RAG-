import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
VECTOR_DB_DIR = os.path.join(os.path.dirname(BASE_DIR), "vector_store")
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Model Config
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# Retrieval Config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 3

# Generation Config
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
