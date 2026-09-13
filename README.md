# AI Study Assistant

An AI-powered study assistant designed to help students learn more effectively.

## Features

- Ask questions from uploaded PDFs
- Summarize study materials
- Generate flashcards
- Generate practice questions
- RAG-based document retrieval

## Project Structure

- app/        → Streamlit application
- rag/        → RAG core
- ingestion/ → PDF processing and text preprocessing
- cli/        → Command-line interface
- data/       → Local application data and ChromaDB storage

### Root Files

- .env.example      → Template for environment variables
- .gitignore        → Git ignore rules
- .python-version   → Pinned Python version (3.11)
- requirements.txt  → Project dependencies
- README.md         → Project documentation

## Requirements

- Python 3.11
- venv

## Setup

- Python: 3.11.x (see .python-version)
- Create venv: `python -m venv venv`
- Activate:
  - Linux/Mac: `source venv/bin/activate`
  - Windows: `venv\Scripts\activate`
- Install deps: `pip install -r requirements.txt`
- Verify: `python --version` should print 3.11.x

## Team Members
Arian
Fatemeh
Melika
Arshia 
Shahrayar
