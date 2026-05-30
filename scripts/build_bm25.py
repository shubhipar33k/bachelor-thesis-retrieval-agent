"""
build_bm25.py
=============
 
Builds a BM25 (Okapi) index over the preprocessed corpus chunks.
 
Loads chunks from `data/chunks_v1.jsonl`, tokenises each chunk with a
simple lowercase-and-strip-punctuation tokeniser (which works for both
German and English without language-specific stemming), and builds a
BM25Okapi index from the `rank-bm25` library.
 
The index and the chunk metadata are pickled together to
`data/bm25_index.pkl` so the retrieval tool can load both at once.
 
After building, the script runs a handful of spot-check queries against
known benchmark tasks to verify the index returns reasonable results.
 
Run:
    python build_bm25.py
"""

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
    Simple whitespace + punctuation tokeniser.
 
    Lowercases the text and replaces punctuation with whitespace. No
    language-specific stemming is applied, which means German compound
    words and English plurals are treated as distinct tokens. This is
    acceptable for a corpus where exact terminology matters more than
    morphological matching.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return tokens


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """Builds a BM25Okapi index from a list of chunks."""
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    index = BM25Okapi(tokenized_corpus)
    return index


def save_index(index: BM25Okapi, chunks: list[dict], path: Path) -> None:
    """Pickle the index and chunks together for later loading."""
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

    # Spot-checks queries against known benchmark tasks. 
    # Each query targets a specific task to verify the index returns the expected document.
    print("\n" + "="*60)
    print("SPOT CHECK QUERIES")
    print("="*60)

    # T01: expected to retrieve Leistungsnachweise (Bachelorarbeit, 15 ECTS)
    test_query(index, chunks, "Bachelorarbeit Kreditpunkte benotet")
 
    # T02: expected to retrieve deadline + Fehlversuch passage
    test_query(index, chunks, "Abgabefrist Bachelorarbeit nicht bestanden Fehlversuch")
 
    # T06: expected to retrieve Assessments (withdrawal illness 5 days)
    test_query(index, chunks, "withdrawal illness five working days assessment")
 
    # T07: expected to retrieve rule that elective modules cannot be repeated
    test_query(index, chunks, "elective modules cannot be repeated Wahlmodul")
 
    # T08: expected to retrieve ASTO 120 Informatik 18 ECTS rule
    test_query(index, chunks, "Informatik ECTS Credits Wahlpflicht")
    

if __name__ == "__main__":
    main()
