from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_groq import ChatGroq

from backend.config import GROQ_API_KEY
from backend.retriever import get_retriever

def build_rag():
    # Step 1: Define your Chain-of-Thought Prompt
    cot_prompt = PromptTemplate.from_template(
        """You are a patent assistant helping researchers understand patents and AI topics.

        Question: {question}

        Let's think step by step and explain thoroughly before giving the final answer.
        Relevant Information:
        {context}

        Answer:"""
    )

    # The line Let's think step by step and explain thoroughly... 
    # is a classic CoT-style instruction, prompting the LLM to reason 
    # step-by-step before finalizing the answer.



    # Step 2: LLM
    llm = ChatGroq(model="llama3-70b-8192", groq_api_key=GROQ_API_KEY)
    
    # Step 3: Define LLMChain with CoT prompt
    llm_chain = LLMChain(llm=llm, prompt=cot_prompt)

    # Step 4: StuffDocumentsChain combines retrieved context into prompt
    combine_docs_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    # Step 5: Use RetrievalQA with custom combine_documents_chain
    retriever = get_retriever()
    rag_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=combine_docs_chain,
        return_source_documents=True
    )

    return rag_chain
