from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_groq import ChatGroq

from backend.config import GROQ_API_KEY
from backend.retriever import get_retriever


class RAGPipeline:
    def __init__(self):
        # CoT Prompt
        self.prompt = PromptTemplate.from_template(
            """You are a patent assistant helping researchers understand patents and AI topics.

            Question: {question}

            Let's think step by step and explain thoroughly before giving the final answer.
            Relevant Information:
            {context}

            Answer:"""
        )

        # load the LLM
        self.llm = ChatGroq(model="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

        # Chain setup - Create a chain using the LLM and a custom prompt
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

        # 3. Combine document chunks into a single context
        # StuffDocumentsChain(LLMChain + Prompt): Combines multiple retrieved chunks into one big context and sends it to llm_chain.
        self.combine_docs_chain = StuffDocumentsChain(
            llm_chain=self.llm_chain,
            document_variable_name="context"    # insert the chunks into {context} inside the prompt.
        )

        # 4. Setup retriever (fetch relevant chunks)
        self.retriever = get_retriever()

        # 5. Build the RetrievalQA chain
        self.rag_chain = RetrievalQA(                       # 1. Takes the user question
            retriever=self.retriever,                       # 2. Uses the retriever to find docs
            combine_documents_chain=self.combine_docs_chain, # 3. Uses combine_documents_chain to feed them to LLM
            return_source_documents=True                     # 4. Returns the LLM’s answer + the source documents
        )
    
    def get_answer_with_sources(self, query: str):
        result = self.rag_chain.invoke({"query": query})
        answer = result["result"]
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content
            }
            for doc in result.get("source_documents", [])
        ]
        return answer, sources
    
    def get_llm_only_response(self, query: str) -> str:
        return self.llm.invoke(query).content


def build_rag():
    return RAGPipeline()

# def invoke(self, inputs: dict):
    #     result = self.rag_chain.invoke(inputs)

    #     # Add source document info into result
    #     result["sources"] = []
    #     for doc in result.get("source_documents", []):
    #         result["sources"].append({
    #             "source": doc.metadata.get("source", "Unknown"),
    #             "content": doc.page_content
    #         })

    #     return result

# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.chains.llm import LLMChain
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# from langchain_groq import ChatGroq

# from backend.config import GROQ_API_KEY
# from backend.retriever import get_retriever

# def get_llm_only_response(self, query: str) -> str:
#     return self.llm.invoke(query).content

# def build_rag():
#     # Step 1: Define your Chain-of-Thought Prompt
#     cot_prompt = PromptTemplate.from_template(
#         """You are a patent assistant helping researchers understand patents and AI topics.

#         Question: {question}

#         Let's think step by step and explain thoroughly before giving the final answer.
#         Relevant Information:
#         {context}

#         Answer:"""
#     )

#     # The line Let's think step by step and explain thoroughly... 
#     # is a classic CoT-style instruction, prompting the LLM to reason 
#     # step-by-step before finalizing the answer.



#     # Step 2: LLM
#     llm = ChatGroq(model="llama3-70b-8192", groq_api_key=GROQ_API_KEY)
    
#     # Step 3: Define LLMChain with CoT prompt
#     llm_chain = LLMChain(llm=llm, prompt=cot_prompt)

#     # Step 4: StuffDocumentsChain combines retrieved context into prompt
#     combine_docs_chain = StuffDocumentsChain(
#         llm_chain=llm_chain,
#         document_variable_name="context"
#     )

#     # Step 5: Use RetrievalQA with custom combine_documents_chain
#     retriever = get_retriever()
#     rag_chain = RetrievalQA(
#         retriever=retriever,
#         combine_documents_chain=combine_docs_chain,
#         return_source_documents=True
#     )

#     return rag_chain

