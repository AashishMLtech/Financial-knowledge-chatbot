import os
import re
import pickle
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from newspaper import Article
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ModuleNotFoundError as exc:
    st.error(
        "Missing dependency: `faiss-cpu`. Install requirements with `pip install -r requirements.txt` and restart Streamlit."
    )
    raise exc

load_dotenv()

APP_TITLE = "Financial Knowledge Chatbot"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHAT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
INDEX_PATH = "faiss_store_financial.index"
METADATA_PATH = "faiss_store_financial_meta.pkl"


@dataclass
class Chunk:
    text: str
    source: str


@st.cache_resource
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_groq_client() -> Optional[Groq]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_article(url: str) -> Optional[str]:
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        return text or None
    except Exception:
        return None


def load_fallback_text(url: str) -> Optional[str]:
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FinancialKnowledgeBot/1.0)"
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text or None
    except Exception:
        return None


def load_documents(urls: List[str]) -> List[Chunk]:
    chunks: List[Chunk] = []
    progress = st.progress(0)
    status = st.empty()

    for index, url in enumerate(urls, start=1):
        status.info(f"Loading {index}/{len(urls)}: {url}")
        text = load_article(url)
        if not text:
            text = load_fallback_text(url)
        if not text:
            st.warning(f"Could not extract content from {url}")
            progress.progress(index / len(urls))
            continue

        for chunk in split_text(text):
            chunks.append(Chunk(text=chunk, source=url))
        progress.progress(index / len(urls))

    status.empty()
    progress.empty()
    return chunks


def load_uploaded_urls(file) -> List[str]:
    if file is None:
        return []
    content = file.read().decode("utf-8", errors="ignore")
    urls = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            urls.append(line)
    return urls


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_index(chunks: List[Chunk]):
    embedder = get_embedder()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = normalize_embeddings(embeddings.astype(np.float32))
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, embeddings


def save_index(index, chunks: List[Chunk]):
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)


@st.cache_resource
def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return None, []
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def retrieve(query: str, chunks: List[Chunk], index, top_k: int = 5):
    embedder = get_embedder()
    query_embedding = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    query_embedding = normalize_embeddings(query_embedding)[0]
    scores, ranked_indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], ranked_indices[0]):
        if idx < 0:
            continue
        results.append(
            {
                "text": chunks[idx].text,
                "source": chunks[idx].source,
                "score": float(score),
            }
        )
    return results


def build_prompt(query: str, contexts: List[dict]) -> str:
    context_text = []
    for i, item in enumerate(contexts, start=1):
        context_text.append(
            f"[{i}] Source: {item['source']}\nExcerpt: {item['text']}"
        )

    return f"""
You are a precise financial research assistant.

Rules:
- Answer only using the provided excerpts.
- If the excerpts do not contain the answer, say "I could not find that in the provided sources."
- Do not guess, do not add background facts, and do not hallucinate.
- Cite the supporting source numbers in the answer.

Question:
{query}

Provided excerpts:
{chr(10).join(context_text)}

Return:
- A short direct answer
- Bullet points for key evidence if useful
- Source numbers in brackets, like [1] [2]
""".strip()


def answer_question(client: Groq, query: str, contexts: List[dict]) -> str:
    prompt = build_prompt(query, contexts)
    response = client.chat.completions.create(
        model=DEFAULT_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a careful assistant that only answers from provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=700,
    )
    return response.choices[0].message.content.strip()


def render_sources(contexts: List[dict]):
    for i, item in enumerate(contexts, start=1):
        with st.expander(f"Source {i} - {item['source']}", expanded=False):
            st.write(item["text"])


st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #10233f 0%, #08111d 45%, #050a12 100%);
            color: #f4f7fb;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .hero-card {
            padding: 1.5rem 1.75rem;
            border-radius: 1.25rem;
            background: rgba(10, 18, 32, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
        }
        .muted {
            color: #9fb3c8;
        }
        .top-metrics {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin-bottom:0.25rem;">Financial Knowledge Chatbot</h1>
        <p class="muted" style="margin-bottom:0;">Precise article retrieval for market and finance research, powered by local embeddings and a Groq-hosted open model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Ingestion")
    st.caption("Add article URLs directly or upload a text file with one URL per line.")
    uploaded_file = st.file_uploader("Upload URL list", type=["txt"])

    manual_urls = []
    for i in range(3):
        url = st.text_input(f"URL {i + 1}", key=f"url_{i}")
        if url.strip():
            manual_urls.append(url.strip())

    top_k = st.slider("Retrieval depth", min_value=3, max_value=8, value=5)
    process_clicked = st.button("Process sources", use_container_width=True)

api_key_present = bool(os.getenv("GROQ_API_KEY"))
if not api_key_present:
    st.warning("Set `GROQ_API_KEY` in your environment to enable answer generation.")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "sources_ready" not in st.session_state:
    st.session_state.sources_ready = False

all_urls = list(dict.fromkeys(manual_urls + load_uploaded_urls(uploaded_file)))

if process_clicked:
    if not all_urls:
        st.warning("Add at least one URL before processing.")
    else:
        with st.spinner("Loading and chunking articles..."):
            chunks = load_documents(all_urls)
        if not chunks:
            st.error("No usable article text could be extracted.")
        else:
            with st.spinner("Building retrieval index..."):
                index, _ = build_index(chunks)
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.sources_ready = True
            save_index(index, chunks)
            st.success(f"Indexed {len(chunks)} chunks from {len(all_urls)} source(s).")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Ask a question")
    query = st.text_area(
        "Question",
        placeholder="For example: What was the main market reaction mentioned in these articles?",
        height=120,
    )
    ask_clicked = st.button("Get answer", type="primary", use_container_width=True)

with col2:
    st.subheader("Status")
    st.metric("Indexed chunks", len(st.session_state.chunks))
    st.metric("Sources loaded", len(all_urls))
    st.caption(f"Chat model: `{DEFAULT_CHAT_MODEL}`")
    if st.session_state.sources_ready:
        if st.session_state.chunks:
            avg_len = sum(len(chunk.text) for chunk in st.session_state.chunks) / len(st.session_state.chunks)
            st.caption(f"Average chunk length: {avg_len:.0f} chars")

if ask_clicked:
    if not query.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.sources_ready or st.session_state.index is None:
        st.warning("Process the sources before asking a question.")
    elif not api_key_present:
        st.error("Missing GROQ_API_KEY.")
    else:
        client = get_groq_client()
        if client is None:
            st.error("Groq client could not be initialized.")
        else:
            with st.spinner("Retrieving the most relevant excerpts..."):
                contexts = retrieve(
                    query=query,
                    chunks=st.session_state.chunks,
                    index=st.session_state.index,
                    top_k=top_k,
                )
            with st.spinner("Generating grounded answer..."):
                answer = answer_question(client, query, contexts)

            st.subheader("Answer")
            st.write(answer)
            st.subheader("Top matches")
            render_sources(contexts)

if st.session_state.sources_ready and st.session_state.chunks:
    st.subheader("Loaded Sources")
    source_list = []
    seen = set()
    for chunk in st.session_state.chunks:
        if chunk.source not in seen:
            seen.add(chunk.source)
            source_list.append(chunk.source)
    for source in source_list:
        st.write(f"- {source}")


# Streamlit runs this file directly.
