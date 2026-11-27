"""PDF loading and chunking: page-level text extraction with overlapping chunks.

Tables are detected with PyMuPDF's find_tables() and linearised row by row as
"header: value; header: value", so that every value stays attached to its row and
column even after chunking. Plain text extraction would interleave multi-line cells
of neighbouring rows."""

import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

CHUNK_CHARS = 1000      # target chunk length
OVERLAP_CHARS = 200     # text carried over from the previous chunk
MARGIN_PT = 55          # running header/footer zone at the top and bottom of a page

FOOTER_RE = re.compile(r"^(Page|Strona)\s+\d+\s+(of|z)\s+\d+$")

# NFKC would turn "10²³" into "1023"; keep exponents readable as "10^23".
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ", "0123456789+-=()ni")
SUPERSCRIPT_RE = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ]+")

CONTINUATION_TOP_PT = 120     # a table starting this close to the top may continue the previous page's table
CONTINUATION_BOTTOM_PT = 150  # ... if that table ended this close to the bottom


@dataclass
class Chunk:
    document: str
    title: str
    page: int           # 1-based, as printed in the PDF footer
    text: str

    def to_dict(self):
        return asdict(self)


def _clean(text: str) -> str:
    text = SUPERSCRIPT_RE.sub(lambda m: "^" + m.group().translate(SUPERSCRIPTS), text)
    text = unicodedata.normalize("NFKC", text)          # ligatures such as "ﬁ"
    text = re.sub(r"-\n(?=[a-ząćęłńóśźż])", "", text)   # words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


def _clean_cell(text: str | None) -> str:
    # Narrow cells break lines at real hyphens ("Encoder-\nonly"), so keep the hyphen.
    return _clean(re.sub(r"(?<=\S)([-/])\n", r"\1", text or ""))


@dataclass
class _Table:
    rect: pymupdf.Rect
    header: list[str]
    rows: list[list[str]]


def _page_tables(page: pymupdf.Page, previous: _Table | None) -> list[_Table]:
    """Tables on the page. A headerless table at the top of the page that continues
    the previous page's table (same column count) inherits that table's header."""
    tables = []
    for t in page.find_tables().tables:
        rows = [[_clean_cell(c) for c in row] for row in t.extract()]
        rows = [r for r in rows if any(r)]
        if not rows:
            continue
        rect = pymupdf.Rect(t.bbox)
        continues = (not tables and previous is not None
                     and rect.y0 < CONTINUATION_TOP_PT
                     and len(previous.header) == len(rows[0])
                     and rows[0] != previous.header)
        if continues:
            tables.append(_Table(rect, previous.header, rows))
        else:
            tables.append(_Table(rect, rows[0], rows[1:]))
    return tables


def _table_text(table: _Table) -> list[str]:
    """One piece per row: "Model: Llama 2 7B (2023); Parameters (N): 7 B; ..."."""
    pieces = []
    for row in table.rows:
        cells = [f"{h}: {v}" if h else v for h, v in zip(table.header, row) if v]
        if cells:
            pieces.append("; ".join(cells) + ".")
    return pieces


def _page_blocks(page: pymupdf.Page, title: str, tables: list[_Table]) -> list[str]:
    """Text blocks and table rows in reading order (top to bottom)."""
    items = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks", sort=True):
        text = _clean(text)
        in_margin = y1 < MARGIN_PT or y0 > page.rect.height - MARGIN_PT
        if not text or (in_margin and (text == title or FOOTER_RE.match(text))):
            continue
        centre = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if any(centre in t.rect for t in tables):       # replaced by the table's rows
            continue
        items.append((y0, [text]))
    items += [(t.rect.y0, _table_text(t)) for t in tables]
    return [piece for _, pieces in sorted(items, key=lambda i: i[0]) for piece in pieces]


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
    chunks, previous = [], None
    for page in doc:
        tables = _page_tables(page, previous)
        for text in chunk_page(_page_blocks(page, title, tables)):
            chunks.append(Chunk(path.name, title, page.number + 1, text))
        # Only a table reaching the bottom of the page can continue on the next one.
        last = tables[-1] if tables else None
        previous = last if last and last.rect.y1 > page.rect.height - CONTINUATION_BOTTOM_PT else None
    return chunks
