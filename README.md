# Tech Jobs Analyzer

A two-stage RAG pipeline for semantic search over technology job postings, with an interactive Streamlit interface and AI-assisted career guidance.

## Project Overview

This project implements a two-stage Retrieval-Augmented Generation (RAG) workflow for analyzing technology job postings.

The system combines:

- Text preprocessing and normalization
- Text chunking
- Vector embeddings
- ChromaDB
- Semantic search
- Relevance filtering using cosine distance
- Grounded prompt construction
- Google Gemini
- Streamlit web interface

## Pipeline

```text
Job Postings CSV
       |
       v
Text Cleaning & Preprocessing
       |
       v
Chunking
       |
       v
ChromaDB + Embeddings
       |
       v
Semantic Retrieval
       |
       v
Relevant Job Context
       |
       v
Grounded RAG Prompt
       |
       v
Google Gemini
       |
       v
Streamlit Interface
       |
       v
Career / Job-Market Answer

Project Stages
Stage 1 — Build ChromaDB Index

Notebook:

notebooks/01_build_chroma_index.ipynb

The first stage:

1.Loads the job-posting dataset.
2.Cleans the job descriptions.
3.Splits descriptions into overlapping text chunks.
4.Generates embeddings.
5.Stores the chunks and metadata in ChromaDB.

Stage 2 — Semantic Search + RAG Career Advisor

Notebook:

notebooks/02_rag_job_advisor.ipynb

The second stage:

1.Loads the existing ChromaDB collection.
2.Performs semantic retrieval.
3.Filters low-relevance results using cosine distance.
4.Builds a grounded context prompt.
5.Sends the retrieved context to Google Gemini.
6.Generates a job-market / career-oriented answer.

Streamlit Application

The project also includes an interactive web interface:

app.py

The Streamlit application allows users to:

1.Enter a natural-language job query.
2.Search the ChromaDB vector database.
3.View the retrieved job-description chunks.
4.Inspect similarity / distance values.
5.Generate an AI-assisted answer using Google Gemini.

Repository Structure
tech-jobs-analyzer/
├── app.py
│
├── notebooks/
│   ├── 01_build_chroma_index.ipynb
│   └── 02_rag_job_advisor.ipynb
│
├── data/
│   └── .gitkeep
│
├── assets/
│   └── .gitkeep
│
├── README.md
├── requirements.txt
├── .gitignore
└── .env.example
Technologies
Python
Pandas
ChromaDB
Vector Embeddings
Semantic Search
RAG
Google Gemini API
Streamlit
Jupyter / Google Colab
Dataset

Place the job-posting dataset at:

data/postings.csv

The current implementation processes the first 10,000 rows by default.

The dataset itself is intentionally not included in the repository.

Make sure you have the right to redistribute any dataset before publishing it.

API Key

The Gemini API key must not be hard-coded in the application or notebook.

Set it as an environment variable:

Linux / macOS
export GEMINI_API_KEY="your-api-key"
Windows PowerShell
$env:GEMINI_API_KEY="your-api-key"

For Google Colab, use Colab Secrets or the runtime environment.

Installation

Install the project dependencies with:

pip install -r requirements.txt
Running the Project
Step 1 — Build the vector index

Open:

notebooks/01_build_chroma_index.ipynb

Set the dataset path for your environment and run the notebook.

This creates the local ChromaDB vector store.

Step 2 — Run the RAG notebook

Open:

notebooks/02_rag_job_advisor.ipynb

Set GEMINI_API_KEY, define your job-search query, and run the notebook.

Example query:

Python developer with machine learning experience

The retriever returns relevant job-description chunks, and Gemini generates the final answer using the retrieved context.

Step 3 — Run the Streamlit application

Make sure the ChromaDB database has already been created.

Then run:

streamlit run app.py

The Streamlit interface will open in your browser.

Enter a query such as:

Python developer with machine learning experience

The application will retrieve relevant job postings and generate a grounded response using the retrieved context.

Streamlit Features

The Streamlit interface provides:

Natural-language job search
Semantic retrieval from ChromaDB
Retrieved job metadata
Similarity / distance information
Grounded Gemini response
Interactive browser-based interface
Limitations

This repository is a prototype RAG workflow rather than a production job-search application.

Retrieval quality depends on:

Dataset quality
Chunking strategy
Embedding model
Relevance threshold

The current implementation also requires the ChromaDB vector store to be created before the Streamlit application can be used.

Security

Never commit:

API keys
.env files
Streamlit secrets
Private credentials
Large local datasets
Generated ChromaDB data
License

Add a license that matches your intended use and the license terms of any third-party dataset used by the project
