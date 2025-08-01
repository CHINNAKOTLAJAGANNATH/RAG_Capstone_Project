# from langchain_weaviate import WeaviateVectorStore
# from weaviate import WeaviateClient
# from weaviate.connect import ConnectionParams

# from backend.config import VECTOR_DB_URL, WEAVIATE_INDEX_NAME
# from backend.embedding import get_embedder

# def get_retriever():
#     client = WeaviateClient(
#         connection_params=ConnectionParams.from_url(
#             url=VECTOR_DB_URL,
#             grpc_port=50051
#         )
#     )
#     client.connect()  # <-- REQUIRED: Connect the client
    
#     embedder = get_embedder()
    
#     # Add the `text_key` argument — this must match the property in your Weaviate schema
#     retriever = WeaviateVectorStore(
#         client=client,
#         index_name=WEAVIATE_INDEX_NAME,
#         text_key="text",  # <-- MUST match your schema's text field
#         embedding=embedder,
#     )
#     return retriever.as_retriever()

# ------------------------------------------
from langchain_community.vectorstores import WeaviateVectorStore
import weaviate
from weaviate.util import get_valid_uuid

from backend.config import WEAVIATE_CLOUD_URL, WEAVIATE_INDEX_NAME
from backend.embedding import get_embedder


def get_retriever():
    # Connect to Weaviate Cloud instance using the proper client
    client = weaviate.Client(
        url=WEAVIATE_CLOUD_URL,
        additional_headers={
            "X-OpenAI-Api-Key": "",  # Optional: add your OpenAI key if you're using modules
        }
    )

    # Ensure the client is ready
    if not client.is_ready():
        raise ConnectionError("Weaviate client failed to connect.")

    embedder = get_embedder()

    retriever = WeaviateVectorStore(
        client=client,
        index_name=WEAVIATE_INDEX_NAME,
        text_key="text",  # Must match the property name in your schema
        embedding=embedder,
    )

    return retriever.as_retriever()
