from dataclasses import dataclass
from pathlib import Path

import pypdf


@dataclass
class PageContent:
    text: str
    page_number: int | None


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document(path: str | Path) -> list[PageContent]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Desteklenmeyen dosya türü: {suffix}")
    if suffix == ".pdf":
        return _load_pdf(path)
    return [PageContent(text=path.read_text(encoding="utf-8"), page_number=None)]


def _load_pdf(path: Path) -> list[PageContent]:
    reader = pypdf.PdfReader(str(path))
    pages: list[PageContent] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(PageContent(text=text, page_number=i))
    return pages
