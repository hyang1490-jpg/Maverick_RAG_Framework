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
[Embedding Layer: bge-large-zh-v1.5] ──▶ Vectorization
       │
       ▼
[Retrieval Layer: ChromaDB] ───────────▶ Semantic Search (Top-K Matching)
       │
       ▼ (Constructs Context-Aware Prompt)
[Generation Layer: Ollama + Qwen2.5] ──▶ Local LLM Inference
       │
       ▼ (JSON Response)
[Frontend UI: HTML/JS + Tailwind] ─────▶ Threat Rating & Streaming Output
```

## 🛠 Tech Stack

* **Backend:** Python, FastAPI, Uvicorn
* **Vector Database:** ChromaDB (Local SQLite)
* **Embeddings:** `SentenceTransformer` (`bge-large-zh-v1.5`)
* **LLM Inference:** Ollama running `qwen2.5:14b`
* **Frontend:** Vanilla JavaScript, HTML5, Tailwind CSS (via CDN)

## 🚀 How to Run Locally

1. **Prerequisites:** * Install Python 3.10+
   * Install [Ollama](https://ollama.com/) and pull the model: `ollama run qwen2.5:14b`

2. **Install Dependencies:**
   ```bash
   pip install fastapi uvicorn chromadb sentence-transformers
   ```

3. **Initialize the Vector Database:**
   ```bash
   python icarus_core.py
   ```

4. **Start the Server:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080
   ```

5. **Access the UI:** Open `http://localhost:8080` in your browser.