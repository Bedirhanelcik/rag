import re
from dataclasses import dataclass

from rag_tr.ingestion.loaders import PageContent


@dataclass
class Chunk:
    text: str
    source_file: str
    page_number: int | None
    chunk_index: int


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(paragraph.strip()) if s]


def _tail_words(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    words = text.split()
    tail: list[str] = []
    total = 0
    for word in reversed(words):
        total += len(word) + 1
        if total > max_chars:
            break
        tail.insert(0, word)
    return " ".join(tail)


def chunk_text(
    pages: list[PageContent],
    source_file: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    units: list[tuple[str, int | None]] = []
    for page in pages:
        for paragraph in _split_paragraphs(page.text):
            if len(paragraph) <= chunk_size:
                units.append((paragraph, page.page_number))
            else:
                for sentence in _split_sentences(paragraph):
                    units.append((sentence, page.page_number))

    chunks: list[Chunk] = []
    current = ""
    current_page: int | None = None
    chunk_index = 0

    def flush() -> None:
        nonlocal current, chunk_index
        if current.strip():
            chunks.append(
                Chunk(
                    text=current.strip(),
                    source_file=source_file,
                    page_number=current_page,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    for unit_text, page_number in units:
        if current_page is None:
            current_page = page_number
        if page_number != current_page and current:
            flush()
            current = ""
            current_page = page_number

        candidate = f"{current} {unit_text}".strip() if current else unit_text
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            flush()
            overlap_text = _tail_words(current, overlap)
            current = f"{overlap_text} {unit_text}".strip()
            current_page = page_number

    flush()
    return chunks
