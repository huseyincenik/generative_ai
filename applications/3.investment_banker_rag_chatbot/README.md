# Investment Banking RAG Chatbot
![image](https://github.com/user-attachments/assets/795f5050-176f-40b4-95e0-0f32263d7446)

This project explores the creation of an "Investment Banker RAG Chatbot" utilizing Intel's Neural Chat LLM. The chatbot leverages **RAG** (Retrieval Augmented Generation) to provide detailed and relevant responses in the domain of investment banking. The system is built with a robust architecture involving multiple cutting-edge technologies, frameworks, and models, ensuring both high performance and efficiency.

## Project Overview

In this project, I dive deep into the world of **Generative AI** and chatbots. The key components of the chatbot include:

- **Intel's Neural Chat LLM**: A 4-bit quantized model from The Bloke (Huggingface), providing remarkable performance with high efficiency.
- **Langchain**: The orchestration framework for managing the various components of the chatbot.
- **BGE Embeddings**: For generating embeddings that enrich the chatbot's responses.
- **Chroma DB**: Acts as the vector store to provide a robust database structure for handling complex queries.
- **CTransformers**: Used to load the LLM in GGUF format, ensuring seamless integration.
- **Flask and FastAPI**: Two distinct backend frameworks used to develop the backend services.
- **User-Friendly Interface**: A conversational frontend interface for intuitive interaction with the chatbot.

### Features:
- **Investment Banking Expertise**: The chatbot can answer queries related to investment banking, backed by data from PDFs.
- **Multi-Backend Architecture**: Showcases the use of both Flask and FastAPI.
- **Embeddings and Vector Store**: Uses BGE embeddings and Chroma DB for efficient handling of complex queries.
- **High Responsiveness**: Ensures seamless interaction with high-performance LLMs.

## Tech Stack

- **Intel Neural Chat LLM** (4-bit quantized) - Huggingface: [Intel Neural Chat LLM](https://huggingface.co/Intel/neural-chat-llm)
- **Langchain**: Framework for orchestrating LLMs and data sources.
- **BGE Embeddings**: [BGE Embeddings](https://huggingface.co/BAAI/bge-large-en)
- **Chroma DB**: A robust vector database for handling embeddings.
- **CTransformers**: [CTransformers GitHub](https://github.com/marella/ctransformers)
- **Flask** & **FastAPI**: Web frameworks for building the backend.
- **Python**: Programming language used for the backend and integration.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/AIAnytime/Investment-Banker-RAG-Chatbot.git
cd Investment-Banker-RAG-Chatbot

