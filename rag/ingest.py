"""PDF loading and chunking: page-level text extraction with overlapping chunks."""

import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

CHUNK_CHARS = 1000      # target chunk length
OVERLAP_CHARS = 200     # text carried over from the previous chunk
MARGIN_PT = 55          # running header/footer zone at the top and bottom of a page

FOOTER_RE = re.compile(r"^(Page|Strona)\s+\d+\s+(of|z)\s+\d+$")


@dataclass
class Chunk:
    document: str
    title: str
    page: int           # 1-based, as printed in the PDF footer
    text: str

    def to_dict(self):
        return asdict(self)


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)          # ligatures such as "ﬁ"
    text = re.sub(r"-\n(?=[a-ząćęłńóśźż])", "", text)   # words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


def _page_blocks(page: pymupdf.Page, title: str) -> list[str]:
    blocks = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks", sort=True):
        text = _clean(text)
        in_margin = y1 < MARGIN_PT or y0 > page.rect.height - MARGIN_PT
        if not text or (in_margin and (text == title or FOOTER_RE.match(text))):
            continue
        blocks.append(text)
    return blocks


def _split_long(text: str) -> list[str]:
    """Split an oversized block at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) > CHUNK_CHARS:
            parts.append(current)
            current = ""
        current = f"{current} {s}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_page(blocks: list[str]) -> list[str]:
    pieces = [p for b in blocks for p in (_split_long(b) if len(b) > CHUNK_CHARS else [b])]
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) > CHUNK_CHARS:
            chunks.append(current)
            tail = current[-OVERLAP_CHARS:]
            current = tail[tail.find(" ") + 1:] if " " in tail else ""
        current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def load_pdf(path: Path) -> list[Chunk]:
    doc = pymupdf.open(path)
    title = _clean(doc.metadata.get("title") or "") or path.stem
    chunks = []
    for page in doc:
        for text in chunk_page(_page_blocks(page, title)):
            chunks.append(Chunk(path.name, title, page.number + 1, text))
    return chunks
