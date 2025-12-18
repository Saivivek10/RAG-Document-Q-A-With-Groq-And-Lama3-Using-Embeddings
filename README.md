# RAG Chatbot with LangChain, Groq, and Embeddings

This project implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain, Groq, and OpenAI or HuggingFace embeddings. The chatbot allows users to query research papers and retrieve contextually relevant answers.

---

## Features

- **Document Ingestion**: Load and process research papers in PDF format.
- **Vector Database**: Create embeddings using OpenAI or HuggingFace models and store them in a FAISS vector database.
- **Question Answering**: Use Groq's Llama-3.1 model to generate answers based on retrieved document context.
- **Streamlit Interface**: A user-friendly web interface for interacting with the chatbot.

---

## Screenshots

<img width="1512" height="982" alt="Huggingfaceembedding" src="https://github.com/user-attachments/assets/a4de3781-10d4-46fc-9c92-97016f5d02b3" />

<img width="1512" height="982" alt="Openai Embedding" src="https://github.com/user-attachments/assets/8d34daeb-44e1-4131-99f3-a0a2b17d9ed3" />



## Requirements

- Python 3.8 or higher
- The following Python libraries:
  - `streamlit`
  - `langchain`
  - `langchain_groq`
  - `langchain_openai`
  - `langchain_community`
  - `faiss-cpu`
  - `python-dotenv`
  - `openai`
- OpenAI API Key (if using OpenAI embeddings)
- HuggingFace Token (if using HuggingFace embeddings)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rag-chatbot.git
   cd rag-chatbot
