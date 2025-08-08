from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL="all-MiniLM-L6-v2"
VECTOR_DB_URL="http://localhost:8080"  # change for streamlit cloud if needed
WEAVIATE_INDEX_NAME="PatentArticles"
CHUNK_SIZE=500
CHUNK_OVERLAP=100
WEAVIATE_URL="http://localhost:8080"  # or your actual Weaviate URL