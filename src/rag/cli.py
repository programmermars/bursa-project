"""Command line: ingest, ask, eval, doctor.

    python -m rag.cli ingest
    python -m rag.cli ask "What are the key risks?" --company KLK
    python -m rag.cli eval --questions eval/questions.csv [--with-llm]
    python -m rag.cli doctor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

from .config import ROOT, get_settings
from .evaluate import generation_metrics, load_questions, results_markdown, retrieval_metrics
from .generate import answer
from .index import Index, build_index
from .retrieve import retrieve


def cmd_ingest(args, s):
    build_index(s, Path(args.raw) if args.raw else None)


def cmd_ask(args, s):
    idx = Index(s)
    hits = retrieve(idx, args.question, k=args.k or s.top_k, mode=args.mode or s.retrieval_mode, company=args.company)
    ans = answer(s, args.question, hits, provider=args.provider)
    print(ans.text, "\n\nSources:")
    for n, h in enumerate(hits, start=1):
        print(f"  [S{n}] {h.chunk.company} {h.chunk.year} p.{h.chunk.page}  score={h.score:.3f}")


def cmd_eval(args, s):
    idx = Index(s)
    qs = load_questions(Path(args.questions))
    k = args.k or s.top_k
    retrieval = [retrieval_metrics(idx, qs, k, m) for m in ("bm25", "vector", "hybrid")]
    for r in retrieval:
        print(f"{r['mode']:>7}  Hit@{k}={r['hit_at_k']:.2f}  Recall@{k}={r['recall_at_k']:.2f}  MRR={r['mrr']:.2f}")
    generation = None
    if args.with_llm:
        generation = generation_metrics(s, idx, qs, k, s.retrieval_mode, args.provider or s.llm_provider)
        print(json.dumps({k2: v for k2, v in generation.items() if k2 != "answers"}, indent=2))
    out = Path(args.out)
    out.write_text(results_markdown(retrieval, generation, idx.meta), encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps({"retrieval": retrieval, "generation": generation}, indent=2, default=str))
    print(f"Saved {out}")


def cmd_doctor(args, s):
    print(f"EMBED_BACKEND={s.embed_backend}  LLM_PROVIDER={s.llm_provider}  RETRIEVAL_MODE={s.retrieval_mode}")
    pdfs = list(s.raw_dir.glob("*.pdf"))
    print(f"PDFs in {s.raw_dir}: {len(pdfs)}")
    print(f"Index built: {(s.index_dir / 'meta.json').exists()}")
    if "ollama" in (s.embed_backend, s.llm_provider):
        try:
            tags = requests.get(f"{s.ollama_url}/api/tags", timeout=5).json()
            names = [m["name"] for m in tags.get("models", [])]
            print(f"Ollama OK, models: {names}")
            for need in {s.ollama_embed_model, s.ollama_llm_model}:
                if not any(n.startswith(need) for n in names):
                    print(f"  missing -> run: ollama pull {need}")
        except requests.RequestException:
            print("Ollama NOT reachable -> install from ollama.com and run `ollama serve`")
    if s.llm_provider == "groq":
        print(f"GROQ_API_KEY set: {bool(s.groq_api_key)}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="rag", description="Bursa annual report RAG")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest"); a.add_argument("--raw")
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("--company"); a.add_argument("--k", type=int)
    a.add_argument("--mode", choices=["vector", "bm25", "hybrid"]); a.add_argument("--provider", choices=["ollama", "groq", "none"])
    a = sub.add_parser("eval"); a.add_argument("--questions", default=str(ROOT / "eval" / "questions.csv"))
    a.add_argument("--k", type=int); a.add_argument("--with-llm", action="store_true")
    a.add_argument("--provider", choices=["ollama", "groq", "none"]); a.add_argument("--out", default=str(ROOT / "eval" / "results.md"))
    sub.add_parser("doctor")
    args = p.parse_args(argv)
    s = get_settings()
    try:
        {"ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval, "doctor": cmd_doctor}[args.cmd](args, s)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
