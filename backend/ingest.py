from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
from backend.config import WEAVIATE_URL, WEAVIATE_INDEX_NAME
import weaviate

def chunk_and_upload(text, class_name=WEAVIATE_INDEX_NAME):
    # Sliding window chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Print chunks
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---\n{chunk}\n")

    # Embed and upload to Weaviate
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    client = weaviate.Client(WEAVIATE_URL)

    vectorstore = Weaviate(
        client=client,
        index_name=class_name,
        text_key="text",
        embedding=embeddings,
    )

    vectorstore.add_texts(chunks)

    print(f"\n✅ Uploaded {len(chunks)} chunks to Weaviate class: {class_name}")
    print(f"📦 Total Chunks: {len(chunks)}")
    print(f"🔢 Embedding Vector Size: {len(embeddings.embed_documents([chunks[0]])[0])}")
