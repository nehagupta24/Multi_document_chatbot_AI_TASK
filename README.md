# Multi_document_chatbot_AI_TASK

## 📌 Overview

  
This project demonstrates a complete **RAG pipeline** that uses a **vector database** and an **instruction-tuned LLM** to answer user queries based on a provided document.

Key features:
- ✅ Document preprocessing, cleaning & chunking
- ✅ Semantic embeddings with FAISS vector store
- ✅ Retrieval-Augmented Generation (retriever + generator)
- ✅ **Real-time streaming responses** via Streamlit
- ✅ Displays source passages for each answer
- ✅ Sidebar showing model name & chunk count
- ✅ Clear chat/reset functionality

---

## 🗂️ Folder Structure
├── 📁 docs/ # 📄 Raw uploaded document files (PDF, DOCX, PPTX)

├── 📁 rag_faiss_store/ # 🗂️ Saved FAISS vector database for semantic search

├── 📄 app.py # ✅ Streamlit app with real-time streaming responses

├── 📄 tools.py # ✅ Core helper: run_agent() and tool registrations

├── 📄 rag_index.py # ✅ Document ingestion, chunking, embeddings, vector store builder

├── 📄 main_chat.py # 🧪 Local test script for the agent (optional)

├── 📄 requirements.txt # ✅ Python dependencies

├── 📄 README.md # 📖 Full project instructions & screenshots

├── 📄 .env.example # ✅ Sample environment config (no real keys!)


---

## ⚙️ **Project Architecture & Flow**

1. **Document Ingestion**  
   - Upload PDF, DOCX, or PPTX via Streamlit.
   - Extract text and clean if needed.

2. **Chunking & Embeddings**  
   - Text is chunked into 100–300 word segments with slight overlaps.
   - Embeddings are generated using **AzureOpenAI Embeddings**.
   - Embeddings are stored in a **FAISS vector store** for fast semantic search.

3. **RAG Pipeline**
   - **Retriever**: Uses FAISS to find top-k relevant chunks.
   - **Generator**: Retrieved context + user query are injected into the LLM prompt.
   - The LLM produces answers grounded in the retrieved context.

4. **Streaming Response & UI**
   - The answer is streamed **sentence-by-sentence** for a smooth user experience.
   - The app displays relevant source chunks.
   - Sidebar shows model in use & chunks indexed.
   - Clear/reset chat available.

---

## 🧩 **Models & Embeddings Used**

- **Embedding Model**: `AzureOpenAIEmbeddings` for generating vector representations.
- **Vector DB**: **FAISS**, stored locally for fast retrieval.
- **LLM**: Small open-source or instruction-optimized model, e.g., `mistral-7b-instruct` / `llama-3` / `zephyr-7b`.

---

## 🚀 **How to Run the Project**

### 1️⃣ **Install Requirements**


pip install -r requirements.txt

## 🔐 Environment Setup

---

### ✅ **1️⃣ Create a `.env` file**


OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_MODEL=gpt-4o

Azure OpenAI Embeddings
AZURE_OPENAI_API_KEY=YOUR_AZURE_API_KEY
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE_NAME.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-XX-XX   # replace with your version, e.g., 2023-03-15-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=YOUR_EMBEDDING_DEPLOYMENT_NAME
---

🙌RUN STREAMLIT APP

streamlit run app.py
---
