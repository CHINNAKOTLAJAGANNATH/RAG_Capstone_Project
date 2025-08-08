# # backend/ingest.py

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_weaviate import WeaviateVectorStore
# from weaviate import WeaviateClient
# from weaviate.connect import ConnectionParams
# from backend.config import WEAVIATE_URL, WEAVIATE_INDEX_NAME
# import os

# def chunk_and_upload(text, filename="unknown.txt", class_name=WEAVIATE_INDEX_NAME):
#     # Chunk the text
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     chunks = splitter.split_text(text)

#     # Print chunks
#     # for i, chunk in enumerate(chunks):
#     #     print(f"\n--- Chunk {i+1} ---\n{chunk}\n")

#     # Embeddings
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     # Connect to Weaviate
#     client = WeaviateClient(
#         connection_params=ConnectionParams.from_url(
#             url=WEAVIATE_URL,
#             grpc_port=50051
#         )
#     )
#     client.connect()

#     # Use langchain_weaviate vector store (✅ Correct version)
#     vectorstore = WeaviateVectorStore(
#         client=client,
#         index_name=class_name,
#         text_key="text",
#         embedding=embeddings
#     )

#     # Upload chunks with metadata
#     metadatas = [{"source": filename} for _ in chunks]
#     vectorstore.add_texts(texts=chunks, metadatas=metadatas)

#     print(f"✅ Uploaded {len(chunks)} chunks from {filename}")


# backend/ingest.py

# from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams

from backend.config import WEAVIATE_URL, WEAVIATE_INDEX_NAME
import os

def get_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def chunk_documents_from_folder(folder_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # Sliding window chunking with 500 size and 100 overlap
    # The RecursiveCharacterTextSplitter is a document chunking strategy from LangChain 
    # that splits long text into smaller overlapping chunks for better retrieval, intelligently by using recursive rules (like sentence > paragraph > character level). 
    # When combined with sliding window techniques, it ensures contextual continuity across chunks.

    all_chunks = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".txt") and os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                document = Document(page_content=text, metadata={"source": filename})
                chunks = splitter.split_documents([document])
                all_chunks.extend(chunks)

    return all_chunks

def chunk_and_upload_all(verbose=False):
    chunks = chunk_documents_from_folder("data")

    if verbose:
        for i, chunk in enumerate(chunks):
            print(f"--- Chunk {i+1} from {chunk.metadata['source']} ---\n{chunk.page_content}\n")

    embeddings = get_embedder()
    client = WeaviateClient(
        connection_params=ConnectionParams.from_url(
            url=WEAVIATE_URL,
            grpc_port=50051
        )
    )
    client.connect()

    vectorstore = WeaviateVectorStore(
        client=client,
        index_name=WEAVIATE_INDEX_NAME,
        text_key="text",
        embedding=embeddings
    )

    # Add only text and metadata, not full Document objects
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    vectorstore.add_texts(texts=texts, metadatas=metadatas)

    print(f"✅ Uploaded {len(chunks)} chunks to Weaviate class: {WEAVIATE_INDEX_NAME}")


# # backend/ingest.py
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Weaviate
# from backend.config import WEAVIATE_URL, WEAVIATE_INDEX_NAME
# import weaviate
# import os

# def chunk_and_upload(text, class_name=WEAVIATE_INDEX_NAME):
#     # Sliding window chunking
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     chunks = splitter.split_text(text)

#     # Print chunks
#     for i, chunk in enumerate(chunks):
#         print(f"\n--- Chunk {i+1} ---\n{chunk}\n")

#     # Embed and upload to Weaviate
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     client = weaviate.Client(WEAVIATE_URL)

#     vectorstore = Weaviate(
#         client=client,
#         index_name=class_name,
#         text_key="text",
#         embedding=embeddings,
#     )

#     vectorstore.add_texts(chunks)

#     print(f"\n✅ Uploaded {len(chunks)} chunks to Weaviate class: {class_name}")
#     print(f"📦 Total Chunks: {len(chunks)}")
#     print(f"🔢 Embedding Vector Size: {len(embeddings.embed_documents([chunks[0]])[0])}")
