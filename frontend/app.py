
import sys
import os
import json
import requests
import streamlit as st

# Setup path to access backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import backend modules
from backend.rag_pipeline import build_rag
from backend.memory_store import store_memory, recall_memory
from backend.evaluate import compute_f1
from frontend.login import login

def main():
    # Set Streamlit page config
    st.set_page_config(page_title="Patent RAG App")
    st.title("🧠 Patent Research Assistant (RAG)")

    # Run login
    login()

    # Check if logged in
    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        st.sidebar.success("Logged in")
        
        # Logout button in sidebar
        if st.sidebar.button("🚪 Logout"):
            st.session_state.clear()
            # st.experimental_rerun()
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

        # Query input
        query = st.text_input("Ask a question about patents:")

        if query:
            memory = recall_memory(query)
            if memory:
                st.info("📌 From Memory:")
                st.write(memory)
                response = memory
            else:
                rag = build_rag()
                # Ask RAG to return context for manual prompt crafting
                result = rag.invoke({"query": query, "return_context": True})
                retrieved_context = result.get("context", "")

                # Use Chain-of-Thought Prompt
                cot_prompt = cot_prompt_template.format(context=retrieved_context, question=query)

                 # Re-run RAG with CoT prompt (assuming build_rag uses OpenAI/Huggingface)
                response = rag.invoke({"query": query, "prompt": cot_prompt})["result"]

                store_memory(query, response)
                st.success("💬 Answer:")
                st.write(response)

            # Auto evaluation
            if query in test_lookup:
                true_ans = test_lookup[query]
                f1 = compute_f1(response, true_ans)
                st.metric("F1-Score", f"{f1:.2f}")
            else:
                st.info("ℹ️ No predefined answer available for evaluation.")

        # Chunks and vector display
        st.markdown("---")
        st.subheader("🧮 View Chunks and Vectors")

        if st.button("🔍 Show All Indexed Chunks"):
            with st.spinner("Fetching indexed chunks..."):
                response = requests.get("http://localhost:8000/debug/chunks")
                if response.status_code == 200:
                    chunk_data = response.json()
                    for idx, chunk in enumerate(chunk_data["chunks"]):
                        st.markdown(f"**Chunk {idx + 1}:**")
                        st.write(chunk["text"])
                        st.code(chunk["vector"][:10], language="python")  # Show first 10 dimensions
                        st.markdown("---")
                else:
                    st.error("Failed to fetch chunks from backend.")
    else:
        st.warning("Please log in to use the application.")

if __name__ == "__main__":
    main()
