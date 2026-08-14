# Zepto Support Assistant

## Overview

This module implements a policy-based customer support assistant for Zepto.

The application uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant policy information from the provided documents and answer customer questions using the retrieved context.

## Features

- Loads 8 Zepto policy documents.
- Creates and stores document embeddings in ChromaDB.
- Retrieves relevant policy documents for customer questions.
- Provides a FastAPI `/ask` endpoint.
- Returns an answer, source documents, and confidence score.
- Uses LangGraph-style workflow for the support assistant pipeline.

## Project Structure

```text
support_assistant/
├── app.py
├── Dockerfile
├── README.md
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
└── chroma_db/