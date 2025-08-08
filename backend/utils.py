# utils.py

import os
import weaviate

def init_weaviate():
    client = weaviate.Client(
        url="http://localhost:8080",
        additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
    )
    return client
