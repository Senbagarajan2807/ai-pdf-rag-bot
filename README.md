# AI-Powered PDF Study Assistant (RAG)

This project is an AI-powered academic assistant that lets students **upload PDF notes and chat with them**.
It uses Retrieval-Augmented Generation (RAG) to answer questions from your own documents.

## Live Demo

'https://ai-pdf-rag-bot-dozzkpuhmijxp4lnmgtnvz.streamlit.app/'

## Features

- Upload one or more PDF lecture notes.
- Automatically extract and chunk text from PDFs.
- Build a vector store (FAISS) using **HuggingFace embeddings** (local, no API limits).
- Ask natural language questions and get answers grounded in your PDFs.
- Uses **Gemini 2.5 Flash** via LangChain for high-quality answers.

## Tech Stack

- Python, Streamlit
- LangChain (RAG pipeline)
- FAISS (vector store)
- HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (embeddings)
- Google Gemini (ChatGoogleGenerativeAI) for answer generation

## Project Structure

```text
.
├── app.py            # Streamlit app with RAG logic
├── requirements.txt  # Python dependencies
├── .env              # Local environment variables (NOT committed)
└── .gitignore
