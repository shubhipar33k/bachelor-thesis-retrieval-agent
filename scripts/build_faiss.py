"""
build_faiss.py
==============
 
Builds a FAISS dense retrieval index over the preprocessed corpus chunks.
 
Loads chunks from `data/chunks_v1.jsonl`, encodes each chunk using a
multilingual sentence-transformer model (`paraphrase-multilingual-
MiniLM-L12-v2`), and builds a FAISS inner-product index. Embeddings are
L2-normalised before indexing so inner-product search is equivalent to
cosine similarity.
 
The multilingual model is chosen because the corpus mixes German and
English documents; encoding both into a shared embedding space avoids
the need for language detection or separate indices.
 
The index, embeddings, chunks, and model name are pickled together to
`data/faiss_index.pkl` so the retrieval tool can reload everything in
one call.
 
Run:
    python build_faiss.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_DIR = Path("data")
CHUNKS_FILE = DATA_DIR / "chunks_v1.jsonl"
FAISS_INDEX_FILE = DATA_DIR / "faiss_index.pkl"

# Multilingual model supporting 50+ languages within a single shared embedding space. 
# Suitable for the bilingual (German/English) corpus.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_faiss_index(
    chunks: list[dict],
    model: SentenceTransformer,
) -> faiss.IndexFlatIP:
    """
    Encodes all chunks and builds a FAISS inner-product index.
 
    Embeddings are L2-normalised by the encoder, so inner product on the
    resulting vectors is equivalent to cosine similarity.
    """
    print("Encoding chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index, embeddings


def save_index(
    index: faiss.IndexFlatIP,
    embeddings: np.ndarray,
    chunks: list[dict],
    model_name: str,
    path: Path,) -> None:
    payload = {
        "index": index,
        "embeddings": embeddings,
        "chunks": chunks,
        "model_name": model_name,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved FAISS index to {path}")


def test_query(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    model: SentenceTransformer,
    query: str,
    top_k: int = 3,) -> None:
    """Prints the top-k FAISS results for a query (used for spot checks)."""
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, top_k)

    print(f"\nQuery: '{query}'")
    print(f"Top {top_k} results:")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        chunk = chunks[idx]
        print(f"\n  Rank {rank} | score: {score:.4f}")
        print(f"  doc_id: {chunk['doc_id'][:60]}")
        print(f"  chunk_id: {chunk['chunk_id'][:60]}")
        print(f"  words: {chunk['word_count']}")
        print(f"  text preview: {chunk['text'][:200]}...")


def main() -> None:
    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"\nLoading model: {MODEL_NAME}")
    print("(First run will download the model — ~120MB, one-time only)")
    model = SentenceTransformer(MODEL_NAME)

    index, embeddings = build_faiss_index(chunks, model)
    save_index(index, embeddings, chunks, MODEL_NAME, FAISS_INDEX_FILE)

    # Spot-check queries. These are paraphrased versions of the BM25 spot
    # checks, designed to test that FAISS retrieves correct chunks even
    # when query wording does not match the source document.
    print("\n" + "="*60)
    print("SPOT CHECK QUERIES — compare results to BM25")
    print("="*60)
    
    # T01: 15 ECTS, graded, paraphrased to test semantic matching
    test_query(index, chunks, model,
        "How many credits is the bachelor thesis and is it graded?")
 
    # T02: deadline consequence, paraphrased
    test_query(index, chunks, model,
        "What happens if a student submits the bachelor thesis too late?")
 
    # T06: withdrawal illness; BM25 struggled here due to lexical mismatch
    test_query(index, chunks, model,
        "A student is sick and cannot attend their exam. What should they do?")
 
    # T12: grammar correction with AI; ambiguous task
    test_query(index, chunks, model,
        "Does using ChatGPT to fix grammar in a thesis require a citation?")
 
    # T14: major to minor switch; multi-document ambiguous task
    test_query(index, chunks, model,
        "Can a student switch from the CL major to the CL minor without losing credits?")


if __name__ == "__main__":
    main()
