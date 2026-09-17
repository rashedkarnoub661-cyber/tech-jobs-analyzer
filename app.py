import os

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types


DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    "./chroma_db"
)

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "tech_jobs"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)


TOP_K = int(
    os.getenv("TOP_K", "3")
)

DISTANCE_THRESHOLD = float(
    os.getenv("DISTANCE_THRESHOLD", "0.8")
)


st.set_page_config(
    page_title="Tech Jobs Analyzer",
    page_icon="💼",
    layout="wide"
)


st.title("💼 Tech Jobs Analyzer")

st.write(
    "RAG-powered semantic search over technology job postings "
    "using ChromaDB and Google Gemini."
)


if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY is not configured."
    )
    st.stop()


if not os.path.exists(DB_PATH):
    st.error(
        f"ChromaDB database was not found at: {DB_PATH}"
    )
    st.stop()


@st.cache_resource
def load_resources():

    embedding_fn = (
        embedding_functions.DefaultEmbeddingFunction()
    )

    chroma_client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
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


query = st.text_input(
    "🔍 Enter your job query",
    placeholder=(
        "e.g. Python developer with "
        "machine learning experience"
    )
)


if st.button(
    "Analyze jobs",
    type="primary"
):

    if not query.strip():

        st.warning(
            "Please enter a job query."
        )

        st.stop()


    try:

        results = collection.query(
            query_texts=[query],
            n_results=TOP_K
        )


        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]


        context_blocks = []
        retrieval_results = []


        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            job_title = metadata.get(
                "job_title",
                "Unknown title"
            )

            company = metadata.get(
                "company",
                "Unknown company"
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
                    f"- Job Title: {job_title}\n"
                    f"  Company: {company}\n"
                    f"  Details: {document}\n"
                )


        context_text = "\n".join(
            context_blocks
        )


        if not context_text:

            context_text = (
                "No relevant job data found "
                "in the retrieved results."
            )


        prompt = f"""
You are an expert AI Career Advisor analyzing technology job market data.

Answer the user's question STRICTLY based on the provided Job Context below.

Do not invent jobs, companies, requirements, salaries, locations, or other facts.

If the context does not contain enough information, clearly say so.

### Job Context:
{context_text}

### User Query:
{query}

### Answer:
"""


        with st.spinner(
            "Analyzing the retrieved jobs..."
        ):

            response = (
                gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1
                    )
                )
            )


        llm_answer = (
            response.text
            if response.text
            else "No answer was returned by the model."
        )


        st.subheader(
            "📡 Results"
        )

        st.markdown(
            f"**Query:** `{query}`"
        )


        st.subheader(
            "🔎 Retrieval Results"
        )


        if retrieval_results:

            for index, item in enumerate(
                retrieval_results,
                start=1
            ):

                with st.expander(
                    f"{index}. "
                    f"{item['job_title']} — "
                    f"{item['company']} "
                    f"(cosine distance: "
                    f"{item['distance']:.4f})"
                ):

                    st.write(
                        item["chunk"]
                    )

        else:

            st.info(
                "No retrieval results were returned."
            )


        st.subheader(
            "🤖 Gemini Career Analysis"
        )

        st.markdown(
            llm_answer
        )


    except Exception as exc:

        st.error(
            f"An error occurred while processing the query: {exc}"
        )
