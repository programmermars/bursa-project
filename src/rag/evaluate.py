"""Evaluation.

Retrieval (no LLM, free):   Hit@k, Recall@k, MRR against hand-labelled pages.
Generation (optional LLM):  citation validity, citation precision (cited pages are correct pages),
                            abstention on questions the reports cannot answer.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .generate import answer
from .index import Index
from .retrieve import retrieve


@dataclass
class Question:
    qid: str
    company: str
    question: str
    expected_pages: set[int]  # empty set = unanswerable from the reports


def load_questions(path: Path) -> list[Question]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pages = {int(p) for p in str(row.get("expected_pages", "")).replace(",", ";").split(";") if p.strip()}
            out.append(Question(row["id"], row.get("company", "").strip(), row["question"].strip(), pages))
    return out


def retrieval_metrics(index: Index, questions: list[Question], k: int, mode: str, filter_company: bool = True) -> dict:
    answerable = [q for q in questions if q.expected_pages]
    hits = recall = rr = 0.0
    per_q = []
    for q in answerable:
        res = retrieve(index, q.question, k=k, mode=mode, company=q.company if filter_company and q.company else None)
        pages = [h.chunk.page for h in res if h.chunk.company == q.company or not q.company]
        found = [p for p in pages if p in q.expected_pages]
        first = next((i for i, p in enumerate(pages, start=1) if p in q.expected_pages), None)
        hits += 1 if found else 0
        recall += len(set(found)) / len(q.expected_pages)
        rr += 1 / first if first else 0
        per_q.append({"id": q.qid, "question": q.question, "hit": bool(found), "first_rank": first,
                      "retrieved_pages": pages, "expected_pages": sorted(q.expected_pages)})
    n = max(len(answerable), 1)
    return {"mode": mode, "k": k, "n": len(answerable),
            "hit_at_k": hits / n, "recall_at_k": recall / n, "mrr": rr / n, "per_question": per_q}


def generation_metrics(settings, index: Index, questions: list[Question], k: int, mode: str, provider: str) -> dict:
    valid = total_cites = correct_cites = 0
    abstain_ok = unanswerable = answered = 0
    rows = []
    for q in questions:
        hits = retrieve(index, q.question, k=k, mode=mode, company=q.company or None)
        ans = answer(settings, q.question, hits, provider=provider)
        said_not_found = "not found in the provided reports" in ans.text.lower()
        if not q.expected_pages:
            unanswerable += 1
            abstain_ok += said_not_found
        else:
            answered += 1
            total_cites += len(ans.citations)
            valid += len(ans.citations) - len(ans.invalid_citations)
            correct_cites += sum(1 for _, p in ans.cited_pages if p in q.expected_pages)
        rows.append({"id": q.qid, "answer": ans.text, "citations": ans.cited_pages})
    return {
        "provider": provider, "n_answerable": answered, "n_unanswerable": unanswerable,
        "citation_validity": valid / total_cites if total_cites else 0.0,
        "citation_precision": correct_cites / total_cites if total_cites else 0.0,
        "abstention_accuracy": abstain_ok / unanswerable if unanswerable else None,
        "answers": rows,
    }


def results_markdown(retrieval: list[dict], generation: dict | None, meta: dict) -> str:
    lines = [
        "# Evaluation results", "",
        f"Index: `{meta['embedder']}`, {meta['chunks']} chunks "
        f"(size {meta['chunk_size']}, overlap {meta['chunk_overlap']}).", "",
        "## Retrieval", "",
        "| Mode | k | Questions | Hit@k | Recall@k | MRR |", "|---|---|---|---|---|---|",
    ]
    for r in retrieval:
        lines.append(f"| {r['mode']} | {r['k']} | {r['n']} | {r['hit_at_k']:.2f} | {r['recall_at_k']:.2f} | {r['mrr']:.2f} |")
    misses = [(r["mode"], q) for r in retrieval for q in r["per_question"] if not q["hit"]]
    if misses:
        lines += ["", "## Missed questions", "",
                  "Questions where no correct page was in the top k. Read these first when improving the system.", "",
                  "| Mode | ID | Question | Retrieved pages | Expected pages |", "|---|---|---|---|---|"]
        for mode, q in misses:
            lines.append(f"| {mode} | {q['id']} | {q['question']} | {', '.join(map(str, q['retrieved_pages']))} "
                         f"| {', '.join(map(str, q['expected_pages']))} |")
    if generation:
        ab = generation["abstention_accuracy"]
        lines += [
            "", f"## Generation ({generation['provider']})", "",
            "| Metric | Value |", "|---|---|",
            f"| Citation validity (cited source exists) | {generation['citation_validity']:.2f} |",
            f"| Citation precision (cited page is a correct page) | {generation['citation_precision']:.2f} |",
            f"| Abstention accuracy (says 'not found' when it should) | {'n/a' if ab is None else f'{ab:.2f}'} |",
        ]
    lines += ["", "Hit@k: share of questions where at least one correct page is in the top k. "
              "Recall@k: share of correct pages retrieved. MRR: mean of 1/rank of the first correct page."]
    return "\n".join(lines) + "\n"
