"""One-command pipeline: put PDFs in data/raw/, run `rag run`, everything else is automatic.

1. Pick backends that actually work on this machine (Ollama -> Groq -> none; Ollama embeddings -> TF-IDF).
2. Check the PDFs (pages, text, scanned pages) and set the fictional sample aside if real reports are present.
3. Build the index.
4. Use hand-labelled eval/questions.csv if it has rows, else generate eval/questions_auto.csv.
5. Evaluate retrieval (bm25 / vector / hybrid) and, if an LLM is available, citations and abstention.
6. Compare a second chunk size on hybrid retrieval.
7. Write eval/results.md and update the results blocks in README.md and docs/PROJECT_WRITEUP.md.
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
from datetime import date
from pathlib import Path

import requests
from pypdf import PdfReader

from .autolabel import build_questions, write_questions
from .config import ROOT
from .evaluate import generation_metrics, load_questions, results_markdown, retrieval_metrics
from .index import Index, build_index
from .ingest import parse_filename

SAMPLE_PREFIX = "DemoPlantation_"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _ollama_models(url: str) -> list[str] | None:
    try:
        return [m["name"] for m in requests.get(f"{url.rstrip('/')}/api/tags", timeout=3).json().get("models", [])]
    except (requests.RequestException, ValueError):
        return None


def resolve_backends(s, log=print):
    """Return settings with backends that work here, falling back instead of failing."""
    models = _ollama_models(s.ollama_url) if "ollama" in (s.embed_backend, s.llm_provider) else None
    has = lambda name: models is not None and any(m.startswith(name) for m in models)  # noqa: E731
    embed, llm = s.embed_backend, s.llm_provider
    if embed == "ollama" and not has(s.ollama_embed_model):
        embed = "tfidf"
        log(f"! Ollama embedding model '{s.ollama_embed_model}' not available -> using offline TF-IDF search")
    if embed == "hf":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            embed = "tfidf"
            log("! sentence-transformers not installed -> using offline TF-IDF search")
    if llm == "ollama" and not has(s.ollama_llm_model):
        llm = "groq" if s.groq_api_key else "none"
        log(f"! Ollama model '{s.ollama_llm_model}' not available -> LLM_PROVIDER={llm}")
    if llm == "groq" and not s.groq_api_key:
        llm = "none"
        log("! GROQ_API_KEY empty -> retrieval only (no LLM answers)")
    return dataclasses.replace(s, embed_backend=embed, llm_provider=llm)


def check_pdfs(raw_dir: Path, log=print) -> list[dict]:
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files in {raw_dir}. Upload the annual reports first.")
    real = [p for p in pdfs if not p.name.startswith(SAMPLE_PREFIX)]
    if real and len(real) < len(pdfs):  # keep the fictional sample out of real results
        aside = raw_dir / "_sample"
        aside.mkdir(exist_ok=True)
        for p in pdfs:
            if p.name.startswith(SAMPLE_PREFIX):
                shutil.move(str(p), aside / p.name)
                log(f"Moved fictional sample {p.name} to {aside} so it does not mix with real reports")
        pdfs = real
    info = []
    for p in pdfs:
        company, year = parse_filename(p)
        reader = PdfReader(str(p))
        n_text = sum(1 for pg in reader.pages if len((pg.extract_text() or "").strip()) >= 40)
        row = {"file": p.name, "company": company, "year": year, "pages": len(reader.pages), "text_pages": n_text}
        info.append(row)
        log(f"{p.name}: {company} {year}, {row['pages']} pages, {n_text} with text")
        if year == "unknown":
            log(f"  ! name it COMPANY_YEAR_anything.pdf (e.g. KLK_2025_annual_report.pdf) for company filters")
        if row["pages"] and n_text / row["pages"] < 0.5:
            log("  ! more than half the pages have no text (scanned?) - those pages cannot be searched")
    return info


def pick_questions(raw_dir: Path, eval_dir: Path, log=print) -> tuple[Path, str]:
    hand = eval_dir / "questions.csv"
    if hand.exists() and load_questions(hand):
        log(f"Using hand-labelled questions: {hand}")
        return hand, "hand-labelled"
    rows = build_questions(raw_dir)
    path = write_questions(rows, eval_dir / "questions_auto.csv")
    n_none = sum(1 for r in rows if not r["expected_pages"])
    log(f"Generated {len(rows) - n_none} answerable + {n_none} unanswerable questions -> {path}")
    return path, "auto-labelled (keyword rules)"


def _summary_md(reports, retrieval, generation, compare, meta, s, label_kind, k) -> str:
    hyb = next(r for r in retrieval if r["mode"] == "hybrid")
    best = max(retrieval, key=lambda r: (r["hit_at_k"], r["mrr"]))
    lines = [
        f"*Last run: {date.today().isoformat()} by `rag run`. Questions: {label_kind}. "
        f"Search: `{meta['embedder']}`. Answers: `{s.llm_provider}`"
        + (f" (`{s.ollama_llm_model}`)" if s.llm_provider == "ollama" else
           f" (`{s.groq_model}`)" if s.llm_provider == "groq" else "") + ".*", "",
        "| Report | Pages (with text) |", "|---|---|",
        *[f"| {r['company']} {r['year']} | {r['pages']} ({r['text_pages']}) |" for r in reports], "",
        f"| Search mode | Questions | Hit@{k} | Recall@{k} | MRR |", "|---|---|---|---|---|",
        *[f"| {r['mode']} | {r['n']} | {r['hit_at_k']:.2f} | {r['recall_at_k']:.2f} | {r['mrr']:.2f} |" for r in retrieval],
    ]
    if generation:
        ab = generation["abstention_accuracy"]
        lines += ["", "| Answer quality | Value |", "|---|---|",
                  f"| Citation validity | {generation['citation_validity']:.2f} |",
                  f"| Citation precision | {generation['citation_precision']:.2f} |",
                  f"| Abstention accuracy | {'n/a' if ab is None else f'{ab:.2f}'} |"]
    if compare:
        lines += ["", f"Chunk size {compare['other_size']} vs {meta['chunk_size']} (hybrid): "
                  f"Hit@{k} {compare['hit_at_k']:.2f} vs {hyb['hit_at_k']:.2f}, "
                  f"MRR {compare['mrr']:.2f} vs {hyb['mrr']:.2f}."]
    misses = [q["id"] for q in hyb["per_question"] if not q["hit"]]
    lines += ["", f"Best search mode: **{best['mode']}**. Hybrid missed {len(misses)} question(s)"
              + (f": {', '.join(misses)}." if misses else ".") + " Details in [eval/results.md](eval/results.md)."]
    if any(r["company"] == SAMPLE_PREFIX.rstrip("_") for r in reports):
        lines += ["", "> These numbers come from the fictional sample report and only show that the pipeline works. "
                  "Upload real annual reports and run `rag run` to replace them."]
    if label_kind.startswith("auto"):
        lines += ["", "> Correct pages were found by keyword rules (`src/rag/autolabel.py`), not checked by hand, "
                  "which favours keyword (BM25) search. Treat them as indicative."]
    return "\n".join(lines)


def update_block(path: Path, body: str, rel_fix: str = "") -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    if rel_fix:
        body = body.replace("](eval/", f"]({rel_fix}eval/")
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: f"{START}\n{body}\n{END}", text, flags=re.S)
    path.write_text(new, encoding="utf-8")
    return True


def run_all(s, compare_size: int | None = 600, with_llm: bool = True, update_docs: bool = True,
            eval_dir: Path | None = None, log=print) -> dict:
    eval_dir = eval_dir or ROOT / "eval"
    log("== 1/6 Checking setup")
    s = resolve_backends(s, log)
    log(f"Search: {s.embed_backend}, answers: {s.llm_provider}")

    log("== 2/6 Checking reports")
    reports = check_pdfs(s.raw_dir, log)

    log("== 3/6 Building index")
    build_index(s, log=log)
    idx = Index(s)

    log("== 4/6 Preparing questions")
    q_path, label_kind = pick_questions(s.raw_dir, eval_dir, log)
    qs = load_questions(q_path)
    k = s.top_k

    log("== 5/6 Evaluating")
    retrieval = [retrieval_metrics(idx, qs, k, m) for m in ("bm25", "vector", "hybrid")]
    for r in retrieval:
        log(f"{r['mode']:>7}  Hit@{k}={r['hit_at_k']:.2f}  Recall@{k}={r['recall_at_k']:.2f}  MRR={r['mrr']:.2f}")
    generation = None
    if with_llm and s.llm_provider != "none":
        log(f"Asking {s.llm_provider} all {len(qs)} questions (this can take a few minutes)...")
        try:
            generation = generation_metrics(s, idx, qs, k, s.retrieval_mode, s.llm_provider)
        except RuntimeError as e:
            log(f"! LLM evaluation skipped: {e}")

    compare = None
    if compare_size and compare_size != s.chunk_size:
        log(f"== 6/6 Comparing chunk size {compare_size}")
        alt_dir = s.index_dir.parent / f"{s.index_dir.name}_chunk{compare_size}"
        alt = dataclasses.replace(s, chunk_size=compare_size, chunk_overlap=min(s.chunk_overlap, compare_size // 4),
                                  index_dir=alt_dir)
        build_index(alt, log=lambda *_: None)
        r = retrieval_metrics(Index(alt), qs, k, "hybrid")
        compare = {"other_size": compare_size, "hit_at_k": r["hit_at_k"], "mrr": r["mrr"]}
        shutil.rmtree(alt_dir, ignore_errors=True)
        log(f"chunk {compare_size}: Hit@{k}={r['hit_at_k']:.2f}  MRR={r['mrr']:.2f}")

    out = eval_dir / "results.md"
    summary = _summary_md(reports, retrieval, generation, compare, idx.meta, s, label_kind, k)
    out.write_text(results_markdown(retrieval, generation, idx.meta).replace(
        "# Evaluation results\n", f"# Evaluation results\n\n{summary.replace('](eval/', '](')}\n", 1), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"reports": reports, "settings": {"embed": s.embed_backend, "llm": s.llm_provider, "chunk_size": s.chunk_size},
         "questions": str(q_path.name), "label_kind": label_kind, "retrieval": retrieval,
         "generation": generation, "chunk_compare": compare}, indent=2, default=str), encoding="utf-8")
    log(f"Saved {out}")
    if update_docs:
        for doc, fix in ((ROOT / "README.md", ""), (ROOT / "docs" / "PROJECT_WRITEUP.md", "../")):
            if update_block(doc, summary, fix):
                log(f"Updated results in {doc.relative_to(ROOT)}")
    return {"reports": reports, "retrieval": retrieval, "generation": generation, "compare": compare,
            "questions": str(q_path), "label_kind": label_kind, "settings": s}
