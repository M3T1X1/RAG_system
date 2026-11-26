"""Interface between the Streamlit frontend and the RAG pipeline.

answer() and build_index() are placeholders until the pipeline is implemented.
"""

from dataclasses import dataclass
from pathlib import Path

import ollama

PDF_DIR = Path(__file__).resolve().parent / "pdf_files"
MODEL = "qwen2.5:14b"
TOP_K = 5


@dataclass
class Source:
    document: str
    page: int
    text: str
    score: float


def list_documents() -> list[str]:
    return sorted(p.name for p in PDF_DIR.glob("*.pdf"))


def model_available() -> bool:
    try:
        return any(m.model == MODEL for m in ollama.list().models)
    except Exception:
        return False


def build_index() -> int:
    """Index all PDFs; returns the number of chunks. TODO: implement."""
    raise NotImplementedError("Indeksowanie nie jest jeszcze zaimplementowane.")


def answer(question: str, top_k: int = TOP_K) -> tuple[str, list[Source]]:
    """Answer a question with cited sources. TODO: implement."""
    raise NotImplementedError("Pipeline RAG nie jest jeszcze zaimplementowany.")
