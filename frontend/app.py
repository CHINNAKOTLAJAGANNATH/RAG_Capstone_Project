import sys
import os
import json
import requests
import streamlit as st
import shutil

# Setup path to access backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import backend modules
from backend.rag_pipeline import build_rag
from backend.memory_store import store_memory, recall_memory
from backend.evaluate import compute_f1
from frontend.login import login
from backend.ingest import chunk_and_upload_all

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("D:\\Data-Engineering\\Project\\Tasks\\rag_project\\frontend\\styles.css")


data_folder = "data"

def main():
    # Set Streamlit page config
    st.set_page_config(page_title="Patent RAG App")
    st.title("👨‍💻 Patent Research Assistant (RAG)")

    # Run login
    login()

    # Check if logged in
    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        st.sidebar.success("Logged in")

        # Logout button in sidebar
        if st.sidebar.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

        # Load test set for automatic F1 scoring
        TEST_FILE = os.path.join("backend", "test_queries.json")
        with open(TEST_FILE, "r") as f:
            test_pairs = json.load(f)
        test_lookup = {item["query"]: item["answer"] for item in test_pairs}

        # Chain-of-Thought Prompt Template
        cot_prompt_template = """
        Use the following retrieved context to answer the question with step-by-step reasoning.

        Context:
        {context}

        Question:
        {question}

        Answer:
        Let's think step by step.
        """
        
        chunk_and_upload_all(verbose=True)
        
        # Query input
        query = st.text_input("Ask a question about patents:")

        # Default values
        rag_response = None
        llm_only_response = None

        if query:
            # ✅ Always build RAG
            rag_pipeline = build_rag()

            # Recall from memory
            memory = recall_memory(query)
            
            if memory:
                # st.info("📌 From Memory:")
                st.markdown("<h4>🧠 From Memory:</h4>", unsafe_allow_html=True)
                st.markdown(f"<div class='chunk-box'>{memory}</div>", unsafe_allow_html=True)
                # st.write(memory)
                rag_response = memory
            else:
                answer, sources = rag_pipeline.get_answer_with_sources(query)
                rag_response = answer
                store_memory(query, answer)
                
                # st.success("💬 RAG Answer:")
                st.markdown("<h4>💬 RAG Answer:</h4>", unsafe_allow_html=True)
                st.markdown(f"<div class='chunk-box'>{answer}</div>", unsafe_allow_html=True)
                # st.write(answer)
                
                if sources:
                    st.markdown("### 📄 Retrieved Source Documents")

                    seen = set()
                    unique_sources = []
                    for src in sources:
                        key = (src['source'], src['content'][:300])
                        if key not in seen:
                            seen.add(key)
                            unique_sources.append(src)

                    for idx, src in enumerate(unique_sources):
                        # st.markdown(f"**🔹 Source {idx + 1}: {src['source']}**")
                        # st.write(src["content"][:300])
                        st.markdown(f"<h5>🔹 Source {idx + 1}: {src['source']}</h5>", unsafe_allow_html=True)
                        st.markdown(f"<div class='chunk-box'>{src['content'][:300]}</div>", unsafe_allow_html=True)
                        st.markdown("---")
                else:
                    st.info("No source documents were retrieved.")

            # ✅ Get baseline LLM-only response
            # print("DEBUG TYPE:", type(rag_pipeline))
            llm_only_response = rag_pipeline.get_llm_only_response(query)

            # st.info("🧪 LLM-Only Baseline Answer:")
            # st.write(llm_only_response)
            st.markdown("<h4>🧪 LLM-Only Baseline Answer:</h4>", unsafe_allow_html=True)
            st.markdown(f"<div class='chunk-box'>{llm_only_response}</div>", unsafe_allow_html=True)

            # Auto evaluation
            if query in test_lookup:
                true_ans = test_lookup[query]
                if rag_response and llm_only_response:
                    rag_f1 = compute_f1(rag_response, true_ans)
                    llm_f1 = compute_f1(llm_only_response, true_ans)

                    st.markdown("### 📊 F1-Score Comparison")
                    col1, col2 = st.columns(2)
                    col1.metric("RAG F1-Score", f"{rag_f1:.2f}")
                    col2.metric("LLM Only F1-Score", f"{llm_f1:.2f}")
            else:
                st.info("ℹ️ No predefined answer available for evaluation.")

        # Chunks and vector display
        st.markdown("---")
        st.subheader("🧮 View Chunks and Vectors")

        if st.button("🔍 Show All Indexed Chunks"):
            with st.spinner("Fetching indexed chunks..."):
                response = requests.get("http://localhost:8000/debug/chunks")
                if response.status_code == 200:
                    # st.success(f"Total Chunks Retrieved: {len(chunk_data['chunks'])}")
                    chunk_data = response.json()
                    for idx, chunk in enumerate(chunk_data["chunks"]):
                        st.markdown(f"**Chunk {idx + 1} (Source: {chunk.get('metadata', {}).get('source', 'Unknown')}):**")
                        st.write(chunk["text"])
                        st.code(chunk["vector"][:10], language="python")  # Show first 10 dimensions
                        st.markdown("---")
                else:
                    st.error("Failed to fetch chunks from backend.")
    else:
        st.warning("Please log in to use the application.")


if __name__ == "__main__":
    main()



# 🧠 Added rag.get_llm_only_response(query)
# 📈 Shows both RAG and LLM-only answers
# 🔬 Computes and displays F1-scores for both if the ground truth is found

