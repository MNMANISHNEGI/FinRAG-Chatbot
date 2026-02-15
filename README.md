# FinRAG-Chatbot

A RAG (Retrieval-Augmented Generation) chatbot built with LangChain, FAISS vector store, and LLM integration for financial document analysis and conversational Q&A.

## Project Overview

This chatbot system enables intelligent interaction with financial documents by leveraging:
- **RAG Architecture**: Combines document retrieval with LLM generation
- **Vector Store**: FAISS for efficient semantic search
- **Memory Management**: Context-aware conversation memory
- **LangChain Integration**: Unified framework for LLM workflows

---

## Project Structure

```
RAG-Chatbot/
 main.py                          # Main entry point
 chatbot.py                       # Chatbot core logic
 create_memory_for_llm.py         # Memory initialization
 connect_memory_with_llm.py       # LLM memory integration
 requirements.txt                 # Python dependencies
 Pipfile                          # Pipenv configuration
 pyproject.toml                   # Project metadata
 .gitignore                       # Git ignore rules
 README.md                        # This file
 data/                            # Source documents (PDFs, etc.)
    NVIDIA-2025-Annual-Report.pdf
 vectorstore/                     # FAISS vector store (generated)
     db_faiss/
         index.faiss              # FAISS index
         index.pkl                # Index metadata
```

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip or Pipenv

### Option 1: Using requirements.txt (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Environment Setup

Create a .env file in the project root with necessary configuration:

```

GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
HF_TOKEN=your_key_here
# Add other API keys as needed
```



---

## Running the Application

```bash
# Run main chatbot
python main.py

# Or run specific module
python chatbot.py
```

---

## Dependencies

Key packages included:
- **langchain** - LLM orchestration framework
- **langchain_community** - Community integrations
- **langchain_huggingface** - HuggingFace integration
- **faiss-cpu** - Vector similarity search
- **pypdf** - PDF document processing
- **huggingface_hub** - HuggingFace model hub access
- **streamlit** - Web UI (optional)

See `requirements.txt` for complete list.

---

## Git Ignore Configuration

The `.gitignore` file excludes:
- Virtual environments: `.venv/`, `.python-version`
- Python caches: `__pycache__/`, `*.pyc`
- Environment files: `.env`
- Dependency locks: `Pipfile.lock`, `uv.lock`
- Generated data: `vectorstore/`, `data/` (large files)
- Temporary files: `*.log`, `*.db`, `*.bak`
- IDE files: `.idea/`

---

## Workflow

1. **Prepare Documents**: Place PDF files in `data/` folder
2. **Create Vector Store**: Run memory creation scripts to build FAISS index
3. **Initialize Chatbot**: Load memory and connect with LLM
4. **Query**: Interact with the chatbot via CLI or Streamlit UI

---

## Key Modules

| Module | Purpose |
|--------|---------|
| `main.py` | Application entry point |
| `chatbot.py` | Core chatbot implementation |
| `create_memory_for_llm.py` | Vector store and memory creation |
| `connect_memory_with_llm.py` | LLM-memory integration |

---

## Troubleshooting

**Missing `vectorstore/`**: Run `create_memory_for_llm.py` to generate FAISS indices.

**Dependency issues**: Ensure virtual environment is activated and run `pip install -r requirements.txt`.

**API Key errors**: Verify `.env` file contains required API keys.

---



## Author

MANISH NEGI
