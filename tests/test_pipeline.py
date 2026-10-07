"""End-to-end tests on the fictional sample report, using the TF-IDF backend (no downloads, no LLM)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_sample_report import main as make_sample  # noqa: E402
from rag.config import Settings  # noqa: E402
from rag.evaluate import load_questions, results_markdown, retrieval_metrics  # noqa: E402
from rag.generate import answer, format_sources, parse_citations  # noqa: E402
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


def test_results_markdown_lists_misses(index):
    s, idx = index
    qs = load_questions(ROOT / "eval" / "sample_questions.csv")
    qs[0].expected_pages = {999}  # force a miss
    md = results_markdown([retrieval_metrics(idx, qs, k=3, mode="bm25")], None, idx.meta)
    assert "## Missed questions" in md and "| 999 |" in md


def test_autolabel_matches_hand_labels(tmp_path):
    from rag.autolabel import build_questions
    make_sample(str(tmp_path))
    auto = {r["id"].split("_", 1)[1]: r["expected_pages"] for r in build_questions(tmp_path)}
    assert auto["risks"] == "3" and auto["ac_meetings"] == "4" and auto["ia_function"] == "5"
    assert auto["kam"] == "6" and auto["revenue"] == "2" and auto["none"] == ""
    assert "fx" not in auto  # the sample never mentions foreign exchange


def test_sample_set_aside_when_real_reports_present(tmp_path):
    from rag.pipeline import check_pdfs
    make_sample(str(tmp_path))
    real = tmp_path / "ACME_2024_annual_report.pdf"
    (tmp_path / "DemoPlantation_2025_annual_report.pdf").rename(real)  # stands in for a real report
    make_sample(str(tmp_path))
    info = check_pdfs(tmp_path, log=lambda *_: None)
    assert [r["company"] for r in info] == ["ACME"]
    assert (tmp_path / "_sample" / "DemoPlantation_2025_annual_report.pdf").exists()


def test_run_all_end_to_end(tmp_path, monkeypatch):
    """Full pipeline with an unreachable Ollama (falls back to TF-IDF) and a mocked Groq LLM."""
    from rag.pipeline import run_all
    raw, ev = tmp_path / "raw", tmp_path / "eval"
    make_sample(str(raw))

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "Answer [S1]."}}]}

    monkeypatch.setattr("rag.generate.requests.post", lambda *a, **k: FakeResp())
    s = Settings(raw_dir=raw, index_dir=tmp_path / "index", embed_backend="ollama", llm_provider="groq",
                 groq_api_key="test", ollama_url="http://127.0.0.1:9")
    out = run_all(s, update_docs=False, eval_dir=ev, log=lambda *_: None)
    assert out["settings"].embed_backend == "tfidf" and out["generation"]["citation_validity"] == 1.0
    assert [c["label"] for c in out["compare"]] == ["Chunk size 600", "Section prefix off"]
    assert not list(tmp_path.glob("index_*"))
    assert (ev / "questions_auto.csv").exists() and "## Missed questions" not in (ev / "results.md").read_text()


def test_rebuild_in_same_process(tmp_path):
    """The app rebuilds the index while an old one is loaded; Chroma's client cache must not break that."""
    make_sample(str(tmp_path / "raw"))
    s = Settings(raw_dir=tmp_path / "raw", index_dir=tmp_path / "index", embed_backend="tfidf", llm_provider="none")
    build_index(s, log=lambda *_: None)
    Index(s)
    build_index(s, log=lambda *_: None)
    assert len(Index(s).chunks) > 0


def test_section_detection():
    from rag.ingest import detect_section
    assert detect_section("KLK Annual Report 2025  87\nAUDIT AND RISK COMMITTEE\nREPORT\nThe Committee met") == "Audit Committee Report"
    assert detect_section("Statement on Risk Management and Internal\nControl\nThe Board") == \
        "Statement on Risk Management and Internal Control"
    assert detect_section("Independent Auditors’ Report\nto the members") == "Independent Auditors' Report"
    toc = "Contents\nChairman's Statement 4\nSustainability Statement 40\nAudit Committee Report 90\nIndependent Auditors' Report 120"
    assert detect_section(toc) == "Contents"
    body = "Revenue grew.\n" * 6 + "as set out in the Audit Committee Report on page 90"
    assert detect_section(body) is None  # a mention deep in the page is not a heading


def test_sections_carried_and_indexed(index):
    s, idx = index
    by_page = {c.page: c.section for c in idx.chunks}
    assert by_page[4] == "Audit Committee Report" and by_page[6] == "Independent Auditors' Report"
    hits = retrieve(idx, "How many times did the Audit Committee meet?", k=3)
    assert "Audit Committee Report" in format_sources(hits)
    assert idx.meta["context_prefix"] is True
