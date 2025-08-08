from langchain.text_splitter import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter to chunk your documents into small overlapping parts (for better retrieval).
from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,         # = 500
        chunk_overlap=CHUNK_OVERLAP    # = 100
    )
    return splitter.create_documents(docs)
