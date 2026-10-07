"""Streamlit demo: streamlit run app.py"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402

from rag.config import get_settings  # noqa: E402
from rag.ingest import parse_filename  # noqa: E402
from rag.pipeline import resolve_backends, run_all  # noqa: E402
from rag.generate import answer  # noqa: E402
from rag.index import Index  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402

st.set_page_config(page_title="Bursa Report RAG", page_icon="📑", layout="wide")


@st.cache_resource(ttl=60)
def app_settings():
    return resolve_backends(get_settings(), log=lambda *_: None)  # falls back if Ollama is not running


settings = app_settings()


@st.cache_resource
def load_index():
    return Index(settings)


st.title("Bursa Annual Report Risk Assistant")
st.caption("Ask about risks, internal control and audit matters in Malaysian listed companies' annual reports. "
           "Every answer cites the report page it came from.")

with st.sidebar:
    st.header("1. Upload reports")
    uploads = st.file_uploader("Annual report PDFs", type="pdf", accept_multiple_files=True,
                               help="Name them COMPANY_YEAR_anything.pdf, or fill in company and year below.")
    names = []
    for i, f in enumerate(uploads or []):
        company, year = parse_filename(Path(f.name))
        if year == "unknown":
            guess = re.search(r"20\d\d", f.name)
            c1, c2 = st.columns(2)
            words = []
            for w in re.split(r"[\s_\-.]+", company):
                if not w or re.fullmatch(r"(?i)(?:ar|ir|fy)\d*|annual|reports?|integrated|\d.*", w):
                    break
                words.append(w)
            company = c1.text_input("Company code", value="".join(words) or "Company", key=f"c{i}")
            year = c2.text_input("Year", value=guess.group(0) if guess else "2025", key=f"y{i}")
        company = re.sub(r"[^A-Za-z0-9-]", "", company) or "Company"
        year = year if re.fullmatch(r"\d{4}", year.strip()) else "2025"
        names.append(f"{company}_{year.strip()}_annual_report.pdf")
    if uploads:
        st.caption("Will save as: " + ", ".join(names))
    st.caption(f"Reports already in data/raw: {len(list(settings.raw_dir.glob('*.pdf')))}")
    if st.button("Build index and evaluate", type="primary", use_container_width=True,
                 help="Saves the uploads, builds the search index, generates test questions, scores the system "
                      "and updates README.md / eval/results.md. Same as `rag run`."):
        settings.raw_dir.mkdir(parents=True, exist_ok=True)
        for f, name in zip(uploads or [], names):
            (settings.raw_dir / name).write_bytes(f.getvalue())
        with st.status("Running the full pipeline...", expanded=True) as status:
            try:
                run_all(settings, log=st.write)
                status.update(label="Done. Results saved to eval/results.md", state="complete")
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                status.update(label=f"Failed: {e}", state="error")
        load_index.clear()

try:
    index = load_index()
except (FileNotFoundError, ValueError) as e:
    st.info("No search index yet. Upload annual report PDFs in the sidebar and click **Build index and evaluate**.")
    st.caption(str(e))
    st.stop()

with st.sidebar:
    st.header("2. Settings")
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
