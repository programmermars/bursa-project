"""Find which PDF pages mention given keywords, to label `expected_pages` in eval/questions.csv.

    python scripts/find_pages.py "Audit Committee" "met" --company KLK
    python scripts/find_pages.py "key audit matter" --any

Prints only page numbers and a short snippet, so you never have to read whole reports.
Page numbers are PDF page indices starting at 1 (what the viewer shows), the same numbers used in citations.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pypdf import PdfReader  # noqa: E402

from rag.ingest import clean_text, parse_filename  # noqa: E402


def search(pdf: Path, keywords: list[str], match_any: bool, width: int) -> list[tuple[int, str]]:
    out = []
    for page_no, page in enumerate(PdfReader(str(pdf)).pages, start=1):
        text = clean_text(page.extract_text() or "")
        low = text.lower()
        found = [kw for kw in keywords if kw.lower() in low]
        if not found or (not match_any and len(found) < len(keywords)):
            continue
        pos = low.find(found[0].lower())
        start = max(0, pos - width // 3)
        snippet = re.sub(r"\s+", " ", text[start:start + width])
        out.append((page_no, snippet))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("keywords", nargs="+", help="case-insensitive phrases; by default a page must contain all of them")
    p.add_argument("--any", action="store_true", help="match pages containing any keyword instead of all")
    p.add_argument("--company", help="only PDFs whose filename starts with this company code")
    p.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    p.add_argument("--width", type=int, default=150, help="snippet length in characters")
    p.add_argument("--max", type=int, default=15, help="maximum pages shown per PDF")
    args = p.parse_args(argv)

    pdfs = sorted(Path(args.raw).glob("*.pdf"))
    if args.company:
        pdfs = [f for f in pdfs if parse_filename(f)[0].lower() == args.company.lower()]
    if not pdfs:
        sys.exit(f"No matching PDFs in {args.raw}")
    for pdf in pdfs:
        hits = search(pdf, args.keywords, args.any, args.width)
        print(f"{pdf.name}: {len(hits)} page(s) -> {';'.join(str(n) for n, _ in hits)}")
        for page_no, snippet in hits[:args.max]:
            print(f"  p.{page_no}: {snippet}")


if __name__ == "__main__":
    main()
