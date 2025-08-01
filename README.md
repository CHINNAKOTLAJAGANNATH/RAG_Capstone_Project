# 🧠 Patent Research Assistant – Retrieval-Augmented Generation (RAG) App

## 👤 Author Details

* **Full Name:** Chinnakotla Jagannath
* **Email ID:** [jagannath.chinnakotla@nexturn.com](mailto:jagannath.chinnakotla@nexturn.com)

## 🎯 Objective

Build a full-stack Retrieval-Augmented Generation (RAG) application focused on the topic **"Patent Research Article"** that:

* Accepts user queries
* Retrieves relevant context from patent documents
* Generates intelligent answers using LLMs
* Displays evaluation metrics like F1-score
* Remembers past interactions (memory)
* Print the chunks and vector indexes
* Has a web interface (deployed on Streamlit Cloud)

---

## 📚 Dataset & Preprocessing

* **Documents Used:** 10 `.txt` files (\~5–10 pages each) covering patent-related topics.
* **Data Preparation:**

  * Cleaned and curated for noise removal
  * Chunked using **Sliding Window** method for better context preservation

---

## 🔧 Tech Stack and Tools Used

| Component                | Tool / Library                            |
| ------------------------ | ----------------------------------------- |
| **Framework**            | LangChain                                 |
| **Embedding Model**      | `all-MiniLM-L6-v2` (HuggingFace)          |
| **Vector DB & Indexing** | Weaviate (`HNSW`)                         |
| **Chunking Method**      | Sliding Window                            |
| **Prompting Technique**  | Chain-of-Thought                          |
| **LLM Provider**         | Groq API                                  |
| **Frontend**             | Streamlit + HTML + CSS + JavaScript       |
| **Deployment**           | Streamlit Cloud                           |
| **Evaluation Metric**    | F1-Score                                  |
| **Memory Support**       | ConversationalMemory (LangChain)          |
| **Login Tracing**        | MLflow (for tracking login/user behavior) |

---

## 🧱 Architecture

1. **Data Ingestion** – 10 patent-related documents are loaded and chunked using a sliding window.
2. **Embedding Generation** – Each chunk is embedded using `all-MiniLM-L6-v2`.
3. **Vector Indexing** – Chunks are indexed into **Weaviate** using HNSW algorithm.
4. **RAG Pipeline**:

   * User submits a query.
   * Most relevant chunks are retrieved from Weaviate.
   * Chunks + question are passed to LLM via Chain-of-Thought prompting.
   * Generated answer is returned.
5. **Evaluation** – F1-Score is calculated comparing model output with ground-truth.
6. **Frontend** – Streamlit UI to interact with the app in real-time.

---

## 📊 Evaluation Metrics

* **F1-Score**:

  * Precision: how many of the generated answers were correct
  * Recall: how much of the ground truth was retrieved
  * F1 = Harmonic mean of precision & recall
* **Metric Usage**: Helps measure the factual correctness of answers in patent context

---

## 💡 Memory Integration

* Uses **LangChain Conversational Buffer Memory**
* Remembers previous questions and answers
* Enables contextual follow-up queries

---

## 🚀 How to Run

### 1. Backend (FastAPI + Weaviate)
    Open Docker and start engine:   
        After opening docker:
        http://localhost:8080/v1

    then run:
    # 1. Start Weaviate
        docker-compose up -d
        docker start weaviate

    Go to .\venv\Scripts\activate
    uvicorn backend.main:app --reload --port 8000


Make sure Docker container for Weaviate is running (using `docker-compose up`).

### 2. Frontend (Streamlit)

    Open another terminal:
        Go to .\venv\Scripts\activate
        streamlit run frontend/app.py


### 3. Access the Application

* Localhost URL: `http://localhost:8501`
* Deployed: [Streamlit Cloud](https://share.streamlit.io/...) *(replace with your deployed link)*

---

## 🔐 Login & Tracing (Optional)

* Users can log in (simulated via MLflow tracking)
* User sessions can be traced and monitored

---

## ✅ Deliverables

* ✅ 10 processed patent documents
* ✅ RAG pipeline with chunking, embedding, indexing, LLM generation
* ✅ UI interface on Streamlit
* ✅ Evaluation via F1-score
* ✅ Memory-enabled QA system
* ✅ Deployed version online
* ✅ Codebase with documentation

---

