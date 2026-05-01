from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


DATA_DIR = Path("data")
CHUNKS_FILE = DATA_DIR / "chunks_v1.jsonl"
BM25_INDEX_FILE = DATA_DIR / "bm25_index.pkl"


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Lowercases and removes punctuation.
    Works for both German and English.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return tokens


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    index = BM25Okapi(tokenized_corpus)
    return index


def save_index(index: BM25Okapi, chunks: list[dict], path: Path) -> None:
    payload = {
        "index": index,
        "chunks": chunks,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved BM25 index to {path}")


def test_query(index: BM25Okapi, chunks: list[dict], query: str, top_k: int = 3) -> None:
    tokens = tokenize(query)
    scores = index.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    print(f"\nQuery: '{query}'")
    print(f"Top {top_k} results:")
    for rank, idx in enumerate(top_indices, 1):
        chunk = chunks[idx]
        print(f"\n  Rank {rank} | score: {scores[idx]:.3f}")
        print(f"  doc_id: {chunk['doc_id']}")
        print(f"  chunk_id: {chunk['chunk_id']}")
        print(f"  words: {chunk['word_count']}")
        print(f"  text preview: {chunk['text'][:200]}...")


def main() -> None:
    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks.")

    print("Building BM25 index...")
    index = build_bm25_index(chunks)
    print("BM25 index built.")

    save_index(index, chunks, BM25_INDEX_FILE)

    # ── Manual spot checks against your benchmark tasks ──────────────────────
    print("\n" + "="*60)
    print("SPOT CHECK QUERIES")
    print("="*60)

    # T01 — should find Leistungsnachweise: Bachelorarbeit, 15 ECTS
    test_query(index, chunks, "Bachelorarbeit Kreditpunkte benotet")

    # T02 — should find deadline + Fehlversuch
    test_query(index, chunks, "Abgabefrist Bachelorarbeit nicht bestanden Fehlversuch")

    # T06 — should find Assessments: withdrawal illness 5 days
    test_query(index, chunks, "withdrawal illness five working days assessment")

    # T07 — should find: elective modules cannot be repeated
    test_query(index, chunks, "elective modules cannot be repeated Wahlmodul")

    # T08 — should find ASTO 120: Informatik 18 ECTS
    test_query(index, chunks, "Informatik ECTS Credits Wahlpflicht")


if __name__ == "__main__":
    main()