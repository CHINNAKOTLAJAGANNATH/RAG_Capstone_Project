from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Load all .txt files from the data folder
print(">>> Loading all .txt files from data/")
data_dir = "data"
all_text = ""

for filename in sorted(os.listdir(data_dir)):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()
            all_text += file_content + "\n"
        print(f"✔ Loaded {filename}")

print(">>> All files loaded and combined.")

# Split the combined text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.create_documents([all_text])
print(f">>> Total chunks created: {len(chunks)}")

# Show the first chunk
print("\n🧩 First Chunk:")
print(chunks[0].page_content)

# Initialize embedder
print("\n🧬 Embedding all chunks...")
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Extract text from all chunks
chunk_texts = [chunk.page_content for chunk in chunks]

# Embed all chunks
vectors = embedder.embed_documents(chunk_texts)

# Show vector for first chunk
print(f"Vector (truncated): {str(vectors[0][:10])}...")  # First 10 dimensions

# Store in FAISS
print("\n🔗 Adding all chunks to FAISS vector store...")
vectorstore = FAISS.from_documents(chunks, embedder)
print(">>> Vector store created successfully")

# Prepare for FastAPI route
combined = [
    {
        "chunk": i + 1,
        "text": chunk.page_content,
        "vector": vectors[i]
    }
    for i, chunk in enumerate(chunks)
]
print(">>> Chunks ready for FastAPI route (/debug/chunks)")