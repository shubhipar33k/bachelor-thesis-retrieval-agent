"""
preprocess_and_chunk.py
=======================
 
PDF preprocessing and chunking for the thesis corpus.
 
This script extracts text from all PDF files in `corpus_v1/`, cleans the
text, splits it into paragraphs, and chunks paragraphs into overlapping
windows of approximately 300 words each.
 
Pipeline:
    1. Extract raw text from each PDF using PyMuPDF.
    2. Clean whitespace and remove page-number artefacts (e.g. "Seite 3/12").
    3. Split into paragraphs (double-newline first, fall back to single-newline
       for documents with poor paragraph structure).
    4. Combine paragraphs into chunks of ~300 words with 50-word overlap.
    5. Detect chunk language (German / English / unknown) with a simple
       word-frequency heuristic.
    6. Save all chunks to `data/chunks_v1.jsonl`.
 
Each output record contains: doc_id, chunk_id, source_file, text, language,
and word_count.
 
Run:
    python preprocess_and_chunk.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF


# Chunking parameters: 300-word target keeps chunks semantically coherent while staying within the embedding model's context window. 
# 50-word overlap ensures sentences spanning chunk boundaries are represented in both adjacent chunks.

CORPUS_DIR = Path("corpus_v1")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "chunks_v1.jsonl"

TARGET_WORDS = 300
OVERLAP_WORDS = 50
MIN_CHUNK_WORDS = 80 # Drop tiny trailing chunks below this threshold


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
    """
    Cleaning for extracted PDF text.
 
    Normalises whitespace and line endings, collapses multi-newlines, and
    removes common page-footer artefacts that PyMuPDF leaves behind:
        - "Seite 3 / 12"   (German page markers)
        - "3 | 12"          (Page X of Y patterns)
    Also fixes stray whitespace before punctuation.
    """
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
 
    Prefers double-newline splitting since most well-formatted PDFs preserve
    paragraph boundaries this way. Falls back to single-newline splitting
    when the document has very few or unusually large paragraphs, which
    happens with PDFs where extraction collapses spacing.
    """
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
    Combine paragraphs into chunks of approximately `target_words` size.
 
    Paragraphs are accumulated into the current chunk until adding the next
    one would exceed the target. The chunk is then closed and the next chunk
    begins with a `overlap_words`-word tail from the previous chunk for
    context continuity. Paragraphs that individually exceed the target are
    further split by sentence.
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
        # Normal case: keep accumulating paragraphs until the target is reached.
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
    """
    Heuristic language detection based on common function words.
 
    Counts occurrences of German vs. English stopword-like markers and
    returns the language with the higher count. Returns "unknown" if neither
    language clearly dominates. This is a rough heuristic, not a proper
    language identification system, but it suffices for tagging chunks in
    a small bilingual corpus.
    """
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
