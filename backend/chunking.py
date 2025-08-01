from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,         # = 500
        chunk_overlap=CHUNK_OVERLAP    # = 100
    )
    return splitter.create_documents(docs)

# Uses Sliding Window via RecursiveCharacterTextSplitter.
# This is Sliding Window chunking, because it splits text 
# with overlap (100 tokens between chunks), allowing context to slide across boundaries.

# SWC: Creates overlapping windows with a fixed stride 
# (common in NLP tasks, more repetitive but captures local context better)

# RecursiveCharacterTextSplitter: Splits text based on character limits with semantic-aware breaking (good general-purpose)