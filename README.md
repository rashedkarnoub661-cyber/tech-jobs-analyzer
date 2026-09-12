# tech-jobs-analyzer
# Tech Jobs Analyzer & Semantic Search API

This project serves as a technical job market trend analyzer, aiming to build an advanced Retrieval-Augmented Generation (RAG) system for semantic searching across thousands of job postings. The system extracts and stores data as vector embeddings within a ChromaDB database, normalizes the text, and feeds it as contextual prompts to Large Language Models (LLMs) via an API endpoint.

## 1. Data Acquisition from Kaggle
The data lifecycle begins by retrieving the job postings dataset from Kaggle. The dataset is downloaded in `CSV` format, containing job details such as titles, company names, and full descriptions, serving as the foundation for the system.

## 2. Data Processing and Storage in ChromaDB
The data processing pipeline is managed via Google Colab and persistently stored on Google Drive:
* **Data Transfer:** Google Colab is mounted to Google Drive, transferring the downloaded `postings.csv` file to Drive for persistent storage.
* **Sample Loading:** Initial dataset loading is restricted to the first 10,000 job postings to optimize memory usage and accelerate indexing.
* **Chunking:** Long job description texts are split into smaller segments using `RecursiveCharacterTextSplitter` with a chunk size of 400 characters and an overlap of 40 characters to preserve context.
* **Indexing:** Segmented texts are indexed into the `tech_jobs` collection in `ChromaDB` using `cosine` distance similarity. The indexing process utilizes a Parent-Child architecture to link each chunk back to its parent job posting, inserting records in batches of 100.

## 3. Data Cleaning, RAG Implementation, and API Construction
This stage prepares the text for semantic querying and integrates it with generative AI:
* **Text Normalization:** Text passes through a `TextNormalizationPipeline` to strip HTML tags, URLs, special characters, and standardize whitespace for high-quality data ingestion.
* **Tech Role Filtering:** The system filters listings to focus exclusively on technical roles such as `developer`, `engineer`, `data scientist`, and `ai`.
* **Semantic Search Engine:** The pre-indexed database is initialized from Drive using `ChromaJobLoader` with text embeddings generated via `DefaultEmbeddingFunction`, retrieving the top 5 most relevant matching chunks per query.
* **RAG & LLM Integration:** Retrieved context chunks from `ChromaJobRetriever` are passed as contextual prompts to an LLM powered by the `google-genai` library.
* **API Endpoints:** The entire pipeline is wrapped into exposed endpoints, allowing users to send natural language queries and receive accurate, data-backed insights sourced directly from real job descriptions.

## Tech Stack & Dependencies
- `pandas`: For dataset manipulation and table operations.
- `chromadb`: For vector database creation and similarity search.
- `langchain-text-splitters`: For document chunking strategies.
- `sentence-transformers` & `google-genai`: For embeddings and LLM context processing.
