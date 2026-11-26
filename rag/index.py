"""Embedding the chunks and storing them in a FAISS index on disk."""

import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import Chunk, load_pdf

EMBED_MODEL = "intfloat/multilingual-e5-base"   # PL + EN in one vector space
INDEX_DIR = Path(__file__).resolve().parent.parent / "index"


@lru_cache(maxsize=1)
def embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def embed_passages(chunks: list[Chunk]) -> np.ndarray:
    # E5 models expect the "passage: " / "query: " prefixes.
    texts = [f"passage: {c.title}\n{c.text}" for c in chunks]
    return embedder().encode(texts, batch_size=32, normalize_embeddings=True,
                             show_progress_bar=True).astype("float32")


def embed_query(question: str) -> np.ndarray:
    return embedder().encode([f"query: {question}"], normalize_embeddings=True).astype("float32")


def _manifest(pdf_dir: Path) -> dict:
    return {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(pdf_dir.glob("*.pdf"))}


def is_current(pdf_dir: Path) -> bool:
    try:
        saved = json.loads((INDEX_DIR / "manifest.json").read_text())
    except FileNotFoundError:
        return False
    return saved == {"model": EMBED_MODEL, "files": _manifest(pdf_dir)}


def build(pdf_dir: Path) -> int:
    chunks = [c for path in sorted(pdf_dir.glob("*.pdf")) for c in load_pdf(path)]
    vectors = embed_passages(chunks)
    index = faiss.IndexFlatIP(vectors.shape[1])          # inner product = cosine on unit vectors
    index.add(vectors)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / "manifest.json").write_text(
        json.dumps({"model": EMBED_MODEL, "files": _manifest(pdf_dir)}))
    load.cache_clear()
    return len(chunks)


@lru_cache(maxsize=1)
def load() -> tuple[faiss.Index, list[Chunk]]:
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    raw = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return index, [Chunk(**c) for c in raw]
