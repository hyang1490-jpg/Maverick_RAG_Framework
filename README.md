# Project Icarus: Local-First RAG Pipeline for Cognitive Bias Detection

![Project Status](https://img.shields.io/badge/Status-MVP_Completed-brightgreen)
![Architecture](https://img.shields.io/badge/Architecture-100%25_Local_RAG-blue)
![LLM](https://img.shields.io/badge/LLM-Qwen2.5_14B-red)
![VectorDB](https://img.shields.io/badge/Database-ChromaDB-orange)

Project Icarus is an experimental, **100% offline, privacy-first Retrieval-Augmented Generation (RAG) sandbox**. It demonstrates a full-stack implementation capable of analyzing user input to detect underlying cognitive biases and matching them against historical archetypes.

*(Note: The current dataset uses public-domain historical figures like Howard Hughes and Nikola Tesla to safely demonstrate system accuracy without legal or privacy risks).*

## ⚡ Demo: The "Icarus" Protocol in Action

![System Demo](./demo.gif)

## 🏗 System Architecture

The entire pipeline runs locally (optimized for RTX 5080 16GB VRAM), ensuring zero data leakage to third-party APIs.

```text
[User Input: High-Risk Thought Process] 
       │
       ▼ (REST API / FastAPI)
[Embedding Layer: nomic-embed-text (Ollama)] ──▶ Vectorization
       │
       ▼
[Retrieval Layer: ChromaDB] ────────────────────▶ Semantic Search (Top-K Matching)
       │
       ▼ (Constructs Context-Aware Prompt)
[Generation Layer: Ollama + Qwen2.5:14b] ──────▶ Local LLM Inference
       │
       ▼ (JSON Response)
[FastAPI Middleware (api_server.py)] ───────────▶ REST API Gateway (Port 8000)
       │
       ▼
[Frontend UI: HTML/JS + Tailwind] ─────────────▶ Threat Rating & Streaming Output
```

## 🛠 Tech Stack

* **Backend:** Python, FastAPI, Uvicorn
* **API Gateway:** FastAPI 中间层 (`api_server.py`, Port 8000)
* **Vector Database:** ChromaDB (Local Persistent Storage)
* **Embeddings:** `nomic-embed-text` via Ollama (本地推理)
* **LLM Inference:** Ollama running `qwen2.5:14b`
* **Frontend:** Vanilla JavaScript, HTML5, Tailwind CSS (via CDN)

## 🚀 How to Run Locally

1. **Prerequisites:**
   * Install Python 3.10+
   * Install [Ollama](https://ollama.com/) and pull the models:
     ```bash
     ollama pull qwen2.5:14b
     ollama pull nomic-embed-text
     ```

2. **Install Dependencies:**
   ```bash
   pip install fastapi uvicorn chromadb ollama tqdm
   ```

3. **Initialize the Vector Database:**
   ```bash
   python ingest_v2.py
   ```

4. **Start the Server:**
   ```bash
   python api_server.py
   ```

5. **Access the UI:** Open `index.html` in your browser (API runs on `http://localhost:8000`).