"""Streamlit demo: streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402

from rag.config import get_settings  # noqa: E402
from rag.generate import answer  # noqa: E402
from rag.index import Index  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402

st.set_page_config(page_title="Bursa Report RAG", page_icon="📑", layout="wide")
settings = get_settings()


@st.cache_resource
def load_index():
    return Index(settings)


st.title("Bursa Annual Report Risk Assistant")
st.caption("Ask about risks, internal control and audit matters in Malaysian listed companies' annual reports. "
           "Every answer cites the report page it came from.")

try:
    index = load_index()
except (FileNotFoundError, ValueError) as e:
    st.error(f"{e}\n\nPut PDFs in data/raw/ and run `python -m rag.cli ingest`.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    company = st.selectbox("Company", ["All"] + index.companies)
    mode = st.radio("Retrieval", ["hybrid", "vector", "bm25"], index=["hybrid", "vector", "bm25"].index(settings.retrieval_mode))
    k = st.slider("Passages to retrieve (k)", 3, 10, settings.top_k)
    provider = st.radio("Answer with", ["ollama", "groq", "none"], index=["ollama", "groq", "none"].index(settings.llm_provider),
                        help="ollama = local model, groq = free cloud tier, none = show retrieved passages only")
    st.caption(f"Index: {index.meta['embedder']} · {index.meta['chunks']} chunks")

examples = [
    "What are the principal risks disclosed and how are they mitigated?",
    "What did the Audit Committee review during the year?",
    "How is the internal audit function structured and who does it report to?",
    "What key audit matters did the external auditor report?",
]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["q"] = ex

question = st.text_input("Your question", key="q")
if question:
    hits = retrieve(index, question, k=k, mode=mode, company=None if company == "All" else company)
    with st.spinner("Thinking..."):
        try:
            ans = answer(settings, question, hits, provider=provider)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
    st.markdown(ans.text)
    if ans.invalid_citations:
        st.warning(f"The model cited sources that do not exist: {ans.invalid_citations}")
    st.subheader("Sources")
    for n, h in enumerate(hits, start=1):
        cited = "✅ cited" if n in ans.citations else ""
        with st.expander(f"[S{n}] {h.chunk.company} {h.chunk.year} · page {h.chunk.page} · score {h.score:.3f} {cited}"):
            st.write(h.chunk.text)
