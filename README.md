# 📈 StockSage — RAG-Powered Real-Time Financial Assistant

StockSage is a financial assistant that answers user questions using recent news articles via a powerful Retrieval-Augmented Generation (RAG) pipeline. It combines semantic chunking, hybrid search (BM25 + embeddings + cross-encoder reranking), and LLMs for accurate and contextual responses.

---

## 🚀 Features

- 🔍 **Semantic Chunking**: Splits articles into meaningful sentence groups based on semantic similarity.
- 📚 **Hybrid Retrieval**: Uses cosine similarity and cross-encoder reranking for relevant chunk selection.
- 🤖 **LLM-Backed Answering**: Generates natural language answers using `mistral` or other language models.
- 🧠 **Context-Aware Q&A**: Uses only retrieved context (no hallucination).
- 🌐 **Source Linking**: Displays clickable sources for transparency.

---

## 🏗️ Tech Stack

- **Backend**: Python 3.10+
- **LLMs**: Mistral (for final answer generation), SentenceTransformers (`all-MiniLM-L6-v2`), Cross-Encoder (`ms-marco`)
- **NLP Tools**: `spaCy` (`en_core_web_sm`)
- **Vector Search**: Cosine similarity + reranking
- **Database**: PostgreSQL, ChromaDB
- **Dependencies**:
  - `sentence-transformers`
  - `spacy`
  - `psycopg2`
  - `numpy`
  - `scikit-learn`
  - `python-dotenv`
  - `torch`

---

## 🧩 Project Structure

```
StockSage/
│
├── app/
│   ├── retriever.py          # Handles chunking, embedding, and retrieval
│   ├── rag_qa.py             # Runs the end-to-end RAG pipeline
│   └── llm.py                # Wrapper to call the LLM
│
├── main.py                   # Entry point
├── .env                      # PostgreSQL connection string
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/StockSage.git
cd StockSage
```

### 2. Set up virtual environment

```bash
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Configure `.env`

Create a `.env` file with your PostgreSQL connection string:

```
DB_CONNECTION_STRING=postgresql://user:password@host:port/dbname
```

### 5. Run the app

```bash
python main.py
```

---

## ✨ Example Usage

```python
query = "How is Apple performing in the stock market?"
answer, sources = run_rag_pipeline(query)
print(answer)
```

---
---

## 📘 Future Improvements

- Add UI with Streamlit or React
- Replace CrossEncoder with ColBERT for faster reranking
- Deploy with Docker & FastAPI

---

## 📝 License

MIT License © 2025 [Akash Chekodu]
