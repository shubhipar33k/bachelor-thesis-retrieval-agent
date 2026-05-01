from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from smolagents import Tool


DATA_DIR = Path("data")
BM25_INDEX_FILE = DATA_DIR / "bm25_index.pkl"
FAISS_INDEX_FILE = DATA_DIR / "faiss_index.pkl"


# ── Shared tokenizer (must match build_bm25.py) ───────────────────────────────

def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


# ── Index loaders (cached at module level so they load once) ──────────────────

_bm25_payload: dict | None = None
_faiss_payload: dict | None = None
_faiss_model: SentenceTransformer | None = None


def _load_bm25() -> tuple[BM25Okapi, list[dict]]:
    global _bm25_payload
    if _bm25_payload is None:
        with BM25_INDEX_FILE.open("rb") as f:
            _bm25_payload = pickle.load(f)
    return _bm25_payload["index"], _bm25_payload["chunks"]


def _load_faiss() -> tuple[faiss.IndexFlatIP, list[dict], SentenceTransformer]:
    global _faiss_payload, _faiss_model
    if _faiss_payload is None:
        with FAISS_INDEX_FILE.open("rb") as f:
            _faiss_payload = pickle.load(f)
    if _faiss_model is None:
        _faiss_model = SentenceTransformer(_faiss_payload["model_name"])
    return _faiss_payload["index"], _faiss_payload["chunks"], _faiss_model


def _format_results(results: list[dict]) -> str:
    """Format retrieved chunks into a string the agent can read."""
    if not results:
        return "No relevant documents found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[Result {i}]")
        lines.append(f"doc_id: {r['doc_id']}")
        lines.append(f"chunk_id: {r['chunk_id']}")
        lines.append(f"score: {r['score']:.4f}")
        lines.append(f"method: {r['method']}")
        lines.append(f"text:\n{r['text']}")
        lines.append("")
    return "\n".join(lines)


# ── BM25 Tool ─────────────────────────────────────────────────────────────────

class BM25SearchTool(Tool):
    name = "bm25_search"
    description = (
        "Search the university document corpus using BM25 keyword search. "
        "Use this tool when the question contains exact terms such as module names, "
        "deadlines, formal labels, ECTS values, or regulation-style wording. "
        "Returns the most relevant document chunks ranked by keyword match score."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query. Use exact terms from the question where possible.",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of results to return. Default is 3.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, query: str, top_k: int = 3) -> str:
        index, chunks = _load_bm25()
        tokens = tokenize(query)
        scores = index.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "doc_id": chunks[idx]["doc_id"],
                    "chunk_id": chunks[idx]["chunk_id"],
                    "score": float(scores[idx]),
                    "method": "bm25",
                    "text": chunks[idx]["text"],
                })

        return _format_results(results)


# ── FAISS Tool ────────────────────────────────────────────────────────────────

class FAISSSearchTool(Tool):
    name = "faiss_search"
    description = (
        "Search the university document corpus using semantic (dense) search. "
        "Use this tool when the question is paraphrased, uses informal wording, "
        "or describes a scenario rather than using exact regulation terminology. "
        "Returns the most semantically similar document chunks."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query. Can be a full question or scenario description.",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of results to return. Default is 3.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, query: str, top_k: int = 3) -> str:
        index, chunks, model = _load_faiss()
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append({
                    "doc_id": chunks[idx]["doc_id"],
                    "chunk_id": chunks[idx]["chunk_id"],
                    "score": float(score),
                    "method": "faiss",
                    "text": chunks[idx]["text"],
                })

        return _format_results(results)


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing BM25 tool...")
    bm25_tool = BM25SearchTool()
    print(bm25_tool.forward("Bachelorarbeit Abgabefrist Fehlversuch"))

    print("\n" + "="*60)
    print("Testing FAISS tool...")
    faiss_tool = FAISSSearchTool()
    print(faiss_tool.forward("What happens if I miss the thesis deadline?"))