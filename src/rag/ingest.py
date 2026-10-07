"""Read annual-report PDFs, split them into page-aware chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    chunk_id: str
    company: str
    year: str
    source: str
    page: int  # 1-based page number as printed by the PDF viewer
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


_FILENAME = re.compile(r"^(?P<company>[A-Za-z0-9\-]+)_(?P<year>\d{4})", re.I)


def parse_filename(path: Path) -> tuple[str, str]:
    """`KLK_2025_annual_report.pdf` -> ("KLK", "2025"). Falls back to stem / "unknown"."""
    m = _FILENAME.match(path.stem)
    if m:
        return m.group("company"), m.group("year")
    return path.stem, "unknown"


def clean_text(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)      # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph/sentence boundaries into chunks of about `size` characters."""
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if len(text) <= size:
        return [text] if text else []
    pieces = re.split(r"(?<=[.!?])\s+|\n\n", text)
    chunks, current = [], ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        while len(piece) > size:  # very long sentence / table row dump
            head, piece = piece[:size], piece[size - overlap:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if len(current) + len(piece) + 1 <= size:
            current = f"{current} {piece}".strip()
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def load_pdf(path: Path, size: int, overlap: int) -> list[Chunk]:
    company, year = parse_filename(path)
    reader = PdfReader(str(path))
    out: list[Chunk] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if len(text) < 40:  # cover pages, photos, scanned pages without text
            continue
        for i, piece in enumerate(split_text(text, size, overlap)):
            out.append(Chunk(f"{company}_{year}_p{page_no}_c{i}", company, year, path.name, page_no, piece))
    return out


def load_folder(folder: Path, size: int, overlap: int) -> list[Chunk]:
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files in {folder}. See data/README.md for how to get the reports.")
    chunks: list[Chunk] = []
    for pdf in pdfs:
        chunks.extend(load_pdf(pdf, size, overlap))
    return chunks
