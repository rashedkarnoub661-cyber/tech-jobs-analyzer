# Tech Job Trends

A two-stage RAG pipeline for semantic search over technology job postings and AI-assisted career guidance.

## Project overview

The project is implemented as two notebooks:

1. **Stage 1 — Build ChromaDB Index**
   - Load `postings.csv`
   - Clean job descriptions
   - Split descriptions into overlapping chunks
   - Generate embeddings with ChromaDB's default embedding function
   - Store chunks and metadata in a persistent ChromaDB collection named `tech_jobs`

2. **Stage 2 — Semantic Search + RAG Career Advisor**
   - Load the existing ChromaDB collection
   - Perform semantic retrieval
   - Filter low-relevance results using cosine distance
   - Build a grounded context prompt
   - Generate an answer with Google Gemini

## Architecture

```text
postings.csv
    │
    ▼
[Stage 1: preprocessing + chunking]
    │
    ▼
[ChromaDB / embeddings]
    │
    ▼
[Stage 2: semantic retrieval]
    │
    ▼
[Grounded prompt]
    │
    ▼
[Gemini]
    │
    ▼
Career / job-market answer
```

## Repository structure

```text
tech-job-trends/
├── notebooks/
│   ├── 01_build_chroma_index.ipynb
│   └── 02_rag_job_advisor.ipynb
├── data/
│   └── .gitkeep
├── README.md
├── requirements.txt
├── .gitignore
└── .env.example
```

## Requirements

Python 3.10+ is recommended. Install the project dependencies with:

```bash
pip install -r requirements.txt
```

For Google Colab, you can also install them in a cell:

```python
!pip install -r requirements.txt
```

## Dataset

Place the job-posting dataset at:

```text
data/postings.csv
```

or update `DATA_PATH` in `01_build_chroma_index.ipynb` to match your environment.

The current implementation reads the first 10,000 rows by default.

> Do not commit a dataset unless you have the right to redistribute it and it is appropriate for a public repository.

## API key

Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key"
```

For Windows PowerShell:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
```

In Google Colab, prefer Colab Secrets or the runtime environment. **Do not hard-code API keys in the notebook.**

## Running the project

### Step 1 — Build the vector index

Open:

```text
notebooks/01_build_chroma_index.ipynb
```

Set `DATA_PATH`, run the notebook, and confirm that `chroma_db/` is created.

### Step 2 — Run the RAG advisor

Open:

```text
notebooks/02_rag_job_advisor.ipynb
```

Set `GEMINI_API_KEY`, update `USER_QUERY`, and run the notebook.

## Example query

```text
Python developer with machine learning experience
```

The retriever returns the most relevant job-description chunks, and Gemini generates the final answer using only that retrieved context.

## Notes

- `chroma_db/` is generated data and is excluded from Git.
- `.env` is excluded from Git.
- Notebook execution outputs are cleared in the clean versions so the repository stays lightweight and readable.
- The RAG prompt explicitly instructs the model not to invent information outside the retrieved job context.

## Limitations

This repository contains a prototype RAG workflow rather than a production job-search application. Retrieval quality depends on the source dataset, chunking strategy, embeddings, and relevance threshold.

## License

Add a license that matches your intended use and the license terms of any third-party dataset used by the project.
