"""Two-stage retrieval: dense search in FAISS, then cross-encoder reranking."""

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from . import index
from .ingest import Chunk

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"   # multilingual cross-encoder
CANDIDATES = 50                            # dense-search candidates passed to the reranker
MIN_RELEVANCE = 0.15                       # reranker score (0–1) below which a chunk is ignored


@dataclass
class Hit:
    chunk: Chunk
    score: float     # reranker relevance in [0, 1]


@lru_cache(maxsize=1)
def reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL, max_length=512)


def retrieve(question: str, top_k: int) -> list[Hit]:
    faiss_index, chunks = index.load()
    _, ids = faiss_index.search(index.embed_query(question), CANDIDATES)
    candidates = [chunks[i] for i in ids[0] if i != -1]
    if not candidates:
        return []

    scores = reranker().predict([(question, f"{c.title}\n{c.text}") for c in candidates])
    hits = sorted((Hit(c, float(s)) for c, s in zip(candidates, scores)),
                  key=lambda h: h.score, reverse=True)
    return [h for h in hits[:top_k] if h.score >= MIN_RELEVANCE]
