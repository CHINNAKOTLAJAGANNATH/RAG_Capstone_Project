from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from backend.config import VECTOR_DB_URL, WEAVIATE_INDEX_NAME

client = WeaviateClient(
    connection_params=ConnectionParams.from_url(VECTOR_DB_URL, grpc_port=50051)
)

client.connect()

# Define schema
schema = {
    "class": WEAVIATE_INDEX_NAME,
    "description": "Patent-related content",
    "vectorizer": "none",
    "properties": [{"name": "text", "dataType": ["text"]}]
}

# Clean up if needed
if client.schema.exists(WEAVIATE_INDEX_NAME):
    client.schema.delete_class(WEAVIATE_INDEX_NAME)

client.schema.create_class(schema)
print(f"✅ Schema '{WEAVIATE_INDEX_NAME}' created successfully.")