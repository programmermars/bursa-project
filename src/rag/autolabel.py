"""Build an evaluation question set automatically from the PDFs, so no hand labelling is needed.

Each template is a question plus keyword rules. A page is a correct page ("expected page") for the question
when it matches every `must` group; candidate pages are ranked by how strongly they match and the top few kept.
Unanswerable questions are kept only if none of their topic words appear anywhere in that report.

These are *weak labels*: good enough to compare retrieval modes and settings, but not hand-checked. Hand-labelled
questions in eval/questions.csv always take priority (see pipeline.py).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .ingest import clean_text, parse_filename


@dataclass
class Template:
    key: str
    question: str
    must: list[list[str]]       # every group must match; a group matches if any regex in it matches
    prefer: list[str]           # extra regexes that raise a page's rank
    max_pages: int = 3


TEMPLATES = [
    Template("risks", "What are the principal risks disclosed and how are they mitigated?",
             [[r"principal risks?", r"key risks?", r"significant risks?", r"risk profile", r"key business risks?"],
              [r"mitigat", r"manag", r"control"]],
             [r"risk management", r"internal control", r"mitigat", r"principal risks?"]),
    Template("fx", "How does the company manage foreign exchange risk?",
             [[r"foreign (?:currency|exchange)", r"currency risk"], [r"\brisks?\b"]],
             [r"hedg", r"forward (?:foreign exchange )?contracts?", r"mitigat", r"exposure"]),
    Template("ac_meetings", "How many times did the Audit Committee meet during the financial year?",
             [[r"audit (?:and risk )?committee", r"\barc?\b"], [r"meetings?", r"\bmet\b"]],
             [r"attendance", r"meetings? (?:were )?held", r"number of meetings", r"\bmet\b", r"\d+/\d+"]),
    Template("ia_function", "Is the internal audit function in-house or outsourced, and to whom does it report?",
             [[r"internal audit (?:function|department|division)", r"group internal audit", r"internal auditors?"],
              [r"reports?(?:ing)? (?:directly |functionally )?to", r"in-house", r"outsourced"]],
             [r"in-house", r"outsourced", r"directly to the audit", r"functionally", r"independen"]),
    Template("ia_cost", "What was the total cost incurred for the internal audit function?",
             [[r"internal audit"], [r"\bcosts?\b"], [r"\brm\s?[\d.,]+"]],
             [r"total cost", r"incurred", r"internal audit function"]),
    Template("kam", "What key audit matters did the external auditor report?",
             [[r"key audit matters?"]],
             [r"how (?:our|the) audit addressed", r"audit addressed", r"we (?:performed|assessed|evaluated)"]),
    Template("climate", "What climate-related or sustainability risks does the company disclose?",
             [[r"climate"], [r"\brisks?\b"]],
             [r"tcfd", r"physical risks?", r"transition risks?", r"climate change", r"emissions?", r"carbon"]),
    Template("revenue", "What was the group's revenue for the financial year?",
             [[r"\brevenue\b"], [r"\brm\s?[\d.,]+", r"rm'000", r"rm million"]],
             [r"financial highlights", r"group revenue", r"revenue (?:of|rose|increased|decreased|fell)", r"five-year"]),
]

UNANSWERABLE = [
    ("What is the company's policy on cryptocurrency investments?", [r"crypto", r"bitcoin", r"blockchain"]),
    ("How many football clubs does the company sponsor?", [r"football", r"soccer"]),
    ("What is the company's plan for space exploration?", [r"space exploration", r"satellite", r"rocket"]),
]


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.I))


def read_pages(pdf: Path) -> list[str]:
    return [clean_text(p.extract_text() or "") for p in PdfReader(str(pdf)).pages]


def label_template(t: Template, pages: list[str]) -> list[int]:
    scored = []
    for page_no, text in enumerate(pages, start=1):
        if len(text) < 40:
            continue
        group_hits = [sum(_count(p, text) for p in group) for group in t.must]
        if not all(group_hits):
            continue
        score = sum(min(h, 5) for h in group_hits) + 2 * sum(min(_count(p, text), 3) for p in t.prefer)
        scored.append((score, page_no))
    scored.sort(reverse=True)
    if not scored:
        return []
    top = scored[0][0]
    return sorted(p for s, p in scored[:t.max_pages] if s >= top / 2)  # drop weak tail matches


def build_questions(raw_dir: Path) -> list[dict]:
    rows: list[dict] = []
    pdfs = sorted(raw_dir.glob("*.pdf"))
    for n, pdf in enumerate(pdfs):
        company, _ = parse_filename(pdf)
        pages = read_pages(pdf)
        full = "\n".join(pages)
        for t in TEMPLATES:
            expected = label_template(t, pages)
            if expected:
                rows.append({"id": f"{company}_{t.key}", "company": company, "question": t.question,
                             "expected_pages": ";".join(map(str, expected)), "source": "auto"})
        # one unanswerable question per report, rotating through the list, skipped if the topic does appear
        for k in range(len(UNANSWERABLE)):
            question, words = UNANSWERABLE[(n + k) % len(UNANSWERABLE)]
            if not any(_count(w, full) for w in words):
                rows.append({"id": f"{company}_none", "company": company, "question": question,
                             "expected_pages": "", "source": "auto"})
                break
    return rows


def write_questions(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "company", "question", "expected_pages", "source"])
        w.writeheader()
        w.writerows(rows)
    return path
