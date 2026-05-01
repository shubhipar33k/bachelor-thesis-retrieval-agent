from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF


CORPUS_DIR = Path("corpus_v1")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "chunks_v1.jsonl"

TARGET_WORDS = 300
OVERLAP_WORDS = 50
MIN_CHUNK_WORDS = 80


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text:
            pages.append(text)
    doc.close()
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Light cleaning for extracted text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*Seite\s+\d+\s*/\s*\d+\s*$", "", text)
    text = re.sub(r"(?m)^\s*\d+\s*\|\s*\d+\s*$", "", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split cleaned text into paragraphs.
    First tries double-newline splitting.
    If that produces very few or very large paragraphs,
    falls back to single-newline splitting.
    """
    # Try double-newline split first
    paragraphs = [p.strip() for p in text.split("\n\n")]
    paragraphs = [p for p in paragraphs if len(p.split()) >= 5]

    if not paragraphs:
        return []

    avg_words = sum(len(p.split()) for p in paragraphs) / len(paragraphs)

    # If average paragraph is very large (>300 words), fall back to single newline
    if avg_words > 300 or len(paragraphs) <= 2:
        paragraphs = [p.strip() for p in text.split("\n")]
        paragraphs = [p for p in paragraphs if len(p.split()) >= 5]

    return paragraphs


def chunk_paragraphs(
    paragraphs: list[str],
    target_words: int = TARGET_WORDS,
    overlap_words: int = OVERLAP_WORDS,
    min_chunk_words: int = MIN_CHUNK_WORDS,
) -> list[str]:
    """
    Chunk text by combining paragraphs up to a target size.
    Overlap is approximate and word-based.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If a single paragraph exceeds target, split it by sentence
        if para_words > target_words:
            # flush current buffer first
            if current:
                chunk_text = "\n\n".join(current).strip()
                if len(chunk_text.split()) >= min_chunk_words:
                    chunks.append(chunk_text)
                current = []
                current_words = 0
            # split large paragraph into sentence-level sub-chunks
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sub_current: list[str] = []
            sub_words = 0
            for sent in sentences:
                sw = len(sent.split())
                if sub_words + sw <= target_words or not sub_current:
                    sub_current.append(sent)
                    sub_words += sw
                else:
                    chunk_text = " ".join(sub_current).strip()
                    if len(chunk_text.split()) >= min_chunk_words:
                        chunks.append(chunk_text)
                    overlap_text = get_overlap_text(" ".join(sub_current), overlap_words)
                    sub_current = [overlap_text, sent] if overlap_text else [sent]
                    sub_words = sum(len(x.split()) for x in sub_current)
            if sub_current:
                chunk_text = " ".join(sub_current).strip()
                if len(chunk_text.split()) >= min_chunk_words:
                    chunks.append(chunk_text)
            continue

        if current_words + para_words <= target_words or not current:
            current.append(para)
            current_words += para_words
        else:
            chunk_text = "\n\n".join(current).strip()
            if len(chunk_text.split()) >= min_chunk_words:
                chunks.append(chunk_text)

            overlap_text = get_overlap_text(chunk_text, overlap_words)
            current = [overlap_text, para] if overlap_text else [para]
            current_words = sum(len(x.split()) for x in current)

    # Final chunk
    if current:
        chunk_text = "\n\n".join(current).strip()
        if len(chunk_text.split()) >= min_chunk_words:
            chunks.append(chunk_text)

    return chunks


def get_overlap_text(text: str, overlap_words: int) -> str:
    """Return the last overlap_words words from text."""
    words = text.split()
    if len(words) <= overlap_words:
        return text
    return " ".join(words[-overlap_words:])


def detect_language_very_rough(text: str) -> str:
    lower = text.lower()
    german_markers = [" und ", " der ", " die ", " das ", " mit ", " studium "]
    english_markers = [" the ", " and ", " with ", " study ", " student ", " use "]
    de_score = sum(marker in lower for marker in german_markers)
    en_score = sum(marker in lower for marker in english_markers)
    if de_score > en_score:
        return "de"
    if en_score > de_score:
        return "en"
    return "unknown"


def iter_pdf_files(folder: Path) -> Iterable[Path]:
    return sorted(folder.rglob("*.pdf"))


def main() -> None:
    all_records = []

    for pdf_path in iter_pdf_files(CORPUS_DIR):
        print(f"Processing: {pdf_path.name}")

        raw_text = extract_pdf_text(pdf_path)
        cleaned = clean_text(raw_text)
        paragraphs = split_into_paragraphs(cleaned)
        chunks = chunk_paragraphs(paragraphs)

        doc_id = pdf_path.stem

        for idx, chunk in enumerate(chunks, start=1):
            record = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{idx:03d}",
                "source_file": pdf_path.name,
                "text": chunk,
                "language": detect_language_very_rough(chunk),
                "word_count": len(chunk.split()),
            }
            all_records.append(record)

    print(f"\nTotal chunks: {len(all_records)}")
    print("\nPer-document summary:")
    from collections import Counter
    counts = Counter(r["doc_id"] for r in all_records)
    for doc_id, count in sorted(counts.items()):
        words = [r["word_count"] for r in all_records if r["doc_id"] == doc_id]
        print(f"  {doc_id[:60]}: {count} chunks, avg {sum(words)//len(words)} words")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_records)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()