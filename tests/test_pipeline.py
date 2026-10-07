"""End-to-end tests on the fictional sample report, using the TF-IDF backend (no downloads, no LLM)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_sample_report import main as make_sample  # noqa: E402
from rag.config import Settings  # noqa: E402
from rag.evaluate import load_questions, retrieval_metrics  # noqa: E402
from rag.generate import answer, parse_citations  # noqa: E402
from rag.index import Index, build_index  # noqa: E402
from rag.ingest import parse_filename, split_text  # noqa: E402
from rag.retrieve import retrieve, rrf  # noqa: E402


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    idx_dir = tmp_path_factory.mktemp("index")
    make_sample(str(raw))
    s = Settings(raw_dir=raw, index_dir=idx_dir, embed_backend="tfidf", llm_provider="none", chunk_size=400, chunk_overlap=60)
    build_index(s, log=lambda *_: None)
    return s, Index(s)


def test_parse_filename():
    assert parse_filename(Path("KLK_2025_annual_report.pdf")) == ("KLK", "2025")
    assert parse_filename(Path("report.pdf")) == ("report", "unknown")


def test_split_respects_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    parts = split_text(text, 300, 50)
    assert len(parts) > 1 and all(len(p) <= 300 for p in parts)


def test_rrf_prefers_items_in_both_lists():
    fused = rrf([[("a", 1), ("b", 1)], [("b", 1), ("c", 1)]], k=3)
    assert fused[0][0] == "b"


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_retrieves_correct_page(index, mode):
    s, idx = index
    hits = retrieve(idx, "Who does the internal audit department report to?", k=3, mode=mode)
    assert 5 in [h.chunk.page for h in hits]


def test_company_filter(index):
    s, idx = index
    assert retrieve(idx, "risk", k=3, mode="hybrid", company="NoSuchCo") == []


def test_eval_on_sample(index):
    s, idx = index
    qs = load_questions(ROOT / "eval" / "sample_questions.csv")
    m = retrieval_metrics(idx, qs, k=3, mode="hybrid")
    assert m["n"] == 8 and m["hit_at_k"] >= 0.75


def test_retrieval_only_answer(index):
    s, idx = index
    hits = retrieve(idx, "key audit matters", k=3)
    assert "Retrieval-only" in answer(s, "key audit matters", hits).text


def test_parse_citations():
    assert parse_citations("Risk A [S1]. Risk B [S2][S1]. Bad [S9].") == [1, 2, 9]


def test_ollama_answer_parsing(index, monkeypatch):
    """Generation path with a mocked Ollama response (no server needed)."""
    s, idx = index
    hits = retrieve(idx, "key audit matters", k=3)

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "Impairment of bearer plants [S1]. Biological assets [S1][S7]."}}

    monkeypatch.setattr("rag.generate.requests.post", lambda *a, **k: FakeResp())
    ans = answer(s, "key audit matters", hits, provider="ollama")
    assert ans.citations == [1, 7] and ans.invalid_citations == [7]
    assert ans.cited_pages[0][1] == hits[0].chunk.page
