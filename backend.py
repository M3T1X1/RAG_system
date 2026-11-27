"""Interface between the Streamlit frontend and the RAG pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import ollama

from rag import generate, index, retrieve

PDF_DIR = Path(__file__).resolve().parent / "pdf_files"
MODEL = generate.MODEL
TOP_K = 5


@dataclass
class Source:
    number: int          # the [n] used in the answer text
    document: str
    page: int
    text: str
    score: float


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    found: bool = True


def list_documents() -> list[str]:
    return sorted(p.name for p in PDF_DIR.glob("*.pdf"))


def model_available() -> bool:
    try:
        return any(m.model == MODEL for m in ollama.list().models)
    except Exception:
        return False


def save_upload(name: str, data: bytes) -> bool:
    """Save an uploaded PDF to the collection; returns False if it is not a PDF."""
    if not data.startswith(b"%PDF"):
        return False
    PDF_DIR.mkdir(exist_ok=True)
    (PDF_DIR / Path(name).name).write_bytes(data)
    return True


def index_is_current() -> bool:
    return index.is_current(PDF_DIR)


def build_index() -> int:
    """Index all PDFs; returns the number of chunks."""
    return index.build(PDF_DIR)


def answer(question: str, top_k: int = TOP_K) -> Answer:
    if not index_is_current():
        build_index()

    refusal = generate.not_found(generate.detect_language(question))
    hits = retrieve.retrieve(question, top_k)
    if not hits:                                   # nothing relevant: refuse without calling the LLM
        return Answer(refusal, found=False)

    text = generate.generate(question, hits)
    if generate.is_not_found(text):
        return Answer(refusal, found=False)

    valid, invalid = generate.cited_numbers(text, len(hits))
    warnings = []
    if invalid:
        warnings.append(f"Odpowiedź powołuje się na nieistniejące źródła: {invalid}.")
    if not valid:
        warnings.append("Odpowiedź nie zawiera cytatów — traktuj ją ostrożnie.")
    unsupported = generate.verify(question, text, hits)
    if unsupported:
        warnings.append("Weryfikacja wykazała twierdzenia bez pokrycia w źródłach: "
                        + "; ".join(unsupported))

    sources = [Source(n, hits[n - 1].chunk.document, hits[n - 1].chunk.page,
                      hits[n - 1].chunk.text, hits[n - 1].score) for n in valid]
    return Answer(text, sources, warnings)
