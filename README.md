# Chat with PDFs using LangChain & AWS Bedrock (RAG)

This project is an end-to-end Retrieval Augmented Generation (RAG) application that allows users to chat with PDF documents using AWS Bedrock, Meta LLaMA 3, and FAISS.

Users can upload PDF files, generate embeddings using Amazon Titan Embeddings, store them in a FAISS vector database, and ask natural language questions to get context-aware, detailed answers directly from the documents.

---

## Overview

- End-to-end Retrieval Augmented Generation (RAG) application  
- Enables natural language Q&A over PDF documents  
- Built using AWS Bedrock, Meta LLaMA 3, and FAISS  
- Generates context-aware answers grounded in document content  
- Simple and interactive UI built with Streamlit  
- Easy to run locally  

---

## Features

- Load and process multiple PDF files  
- Intelligent text chunking using LangChain  
- Vector embeddings generated using Amazon Titan Embeddings  
- Fast similarity search with FAISS  
- Question answering using Meta LLaMA 3 (70B Instruct) via AWS Bedrock  
- Retrieval Augmented Generation for accurate, document-based responses  
- Clean and user-friendly Streamlit interface  

---

## Architecture Overview

### PDF Ingestion
- PDF files are loaded from a local directory  
- Documents are split into manageable text chunks  

### Embedding Generation
- Each text chunk is converted into vector embeddings using Amazon Titan  

### Vector Storage
- Embeddings are stored locally in a FAISS vector database  

### Query Handling (RAG)
- User question triggers similarity search in FAISS  
- Relevant document chunks are retrieved  
- Retrieved context is passed to Meta LLaMA 3  
- Model generates a detailed, context-aware response  

---

## Tech Stack

- AWS Bedrock  
- Amazon Titan Embeddings  
- Meta LLaMA 3 (70B Instruct)  
- LangChain  
- FAISS  
- Streamlit  
- Python  
- Boto3  
