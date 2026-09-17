import os
import time

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types


# =========================
# Configuration
# =========================

DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    "/content/drive/MyDrive/chroma_db",
)

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "tech_jobs",
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]

TOP_K = int(os.getenv("TOP_K", "3"))
DISTANCE_THRESHOLD = float(
    os.getenv("DISTANCE_THRESHOLD", "0.8")
)


# =========================
# Streamlit configuration
# =========================

st.set_page_config(
    page_title="Tech Jobs Analyzer",
    page_icon="💼",
    layout="wide",
)

st.title("💼 Tech Jobs Analyzer")
st.write(
    "RAG-powered semantic search over technology job postings "
    "using ChromaDB and Google Gemini."
)


# =========================
# Configuration validation
# =========================

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY is not configured. "
        "Set it as an environment variable before using the app."
    )
    st.stop()


if not os.path.exists(DB_PATH):
    st.error(
        f"ChromaDB database was not found at: {DB_PATH}"
    )
    st.stop()


# =========================
# Load resources
# =========================

@st.cache_resource(show_spinner=False)
def load_resources():
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    chroma_client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    return collection, gemini_client


try:
    collection, gemini_client = load_resources()
except Exception as exc:
    st.error(
        f"Could not initialize the application: {exc}"
    )
    st.stop()


# =========================
# Gemini generation helper
# =========================

def generate_grounded_answer(prompt: str) -> str:
    """Generate an answer with retry/fallback handling for temporary 503 errors."""

    last_error = None

    for model in GEMINI_MODELS:
        for attempt in range(3):
            try:
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1
                    ),
                )

                if response.text:
                    return response.text

                last_error = RuntimeError(
                    f"Model {model} returned an empty response."
                )

            except Exception as exc:
                last_error = exc
                error_text = str(exc)

                temporary_error = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "high demand" in error_text.lower()
                )

                if not temporary_error:
                    raise

                if attempt < 2:
                    time.sleep(2 ** attempt)

        # Try the next configured model after retrying this one.

    raise RuntimeError(
        "All configured Gemini models failed. "
        f"Last error: {last_error}"
    )


# =========================
# User query
# =========================

query = st.text_input(
    "🔍 Enter your job query",
    placeholder=(
        "e.g. Python developer with machine learning experience"
    ),
)


# =========================
# Analyze jobs
# =========================

if st.button(
    "Analyze jobs",
    type="primary",
):
    if not query.strip():
        st.warning("Please enter a job query.")
        st.stop()

    try:
        # Semantic retrieval
        results = collection.query(
            query_texts=[query.strip()],
            n_results=TOP_K,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        context_blocks = []
        retrieval_results = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            job_title = metadata.get(
                "job_title",
                "Unknown title",
            )

            company = metadata.get(
                "company",
                "Unknown company",
            )

            distance = float(distance)

            retrieval_results.append(
                {
                    "job_title": job_title,
                    "company": company,
                    "distance": distance,
                    "chunk": document,
                }
            )

            if distance <= DISTANCE_THRESHOLD:
                context_blocks.append(
                    f"- Job title: {job_title}\n"
                    f"  Company: {company}\n"
                    f"  Details: {document}"
                )

        context_text = "\n\n".join(context_blocks)

        if not context_text:
            context_text = (
                "No sufficiently relevant jobs were found "
                "in the current database."
            )

        # Grounded RAG prompt
        prompt = f"""
You are an expert AI Career Advisor analyzing tech job market data.

Answer the user's question STRICTLY based on the provided Job Context below.

Do not invent jobs, companies, requirements, salaries, locations, or other facts.

If the context does not contain enough information, clearly say so.

### Job Context
{context_text}

### User Query
{query}

### Answer
"""

        # Gemini generation
        with st.spinner("Analyzing the retrieved jobs..."):
            llm_answer = generate_grounded_answer(prompt)

        # Results
        st.subheader("📡 Results")
        st.markdown(f"**Query:** `{query}`")

        st.subheader("🔎 Retrieval Results")

        if retrieval_results:
            for index, item in enumerate(
                retrieval_results,
                start=1,
            ):
                with st.expander(
                    f"{index}. {item['job_title']} — "
                    f"{item['company']} "
                    f"(cosine distance: {item['distance']:.4f})"
                ):
                    st.write(item["chunk"])
        else:
            st.info("No retrieval results were returned.")

        st.subheader("🤖 Gemini Career Analysis")
        st.markdown(llm_answer)

    except Exception as exc:
        st.error(
            f"An error occurred while processing the query: {exc}"
        )
