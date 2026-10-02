"""Agent tarafindan makine tarafindan okunabilir sekilde tuketilecek sozlesme.

Bir Agentic AI sistemi bu RAG'i tool olarak cagirdiginda "dokumanda bilgi yok"
durumunu bir hatadan, retrieval hatasini da generation (Claude) hatasindan
serbest metne bakmadan ayirt edebilmeli. Bu modul o ayrimi tek yerde tanimlar.
"""

from dataclasses import dataclass, field
from enum import Enum

NO_CONTEXT_MESSAGE = "Dokümanlarda bu bilgi yok."


class QueryStatus(str, Enum):
    """Basarili (HTTP 200) bir /query yanitinin makine-okunur durumu."""

    ANSWERED = "answered"
    NO_RELEVANT_CONTEXT = "no_relevant_context"


class RetrievalStatus(str, Enum):
    """Generation yapilmadan donen retrieval sonucunun durumu."""

    FOUND = "found"
    # "dokumanda yok" durumu generation yolundakiyle ayni string degeri tasir,
    # boylece agent tarafinda tek bir dal yeterli olur.
    NO_RELEVANT_CONTEXT = "no_relevant_context"


@dataclass(frozen=True)
class Passage:
    """Agent'a donen tek bir retrieval sonucu.

    VectorStore/BM25Index/EmbeddingModel detaylarini disa acmaz; `chunk_id`
    deterministiktir (`source_file::chunk_index`) ve atif/yeniden-getirme
    anahtari olarak kullanilabilir."""

    chunk_id: str
    source_file: str
    page_number: int | None
    text: str
    rank: int
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    status: RetrievalStatus
    passages: list[Passage] = field(default_factory=list)


class ErrorCode(str, Enum):
    """Hata yanitlarinda `detail.code` alaninda donen sabitler."""

    RETRIEVAL_ERROR = "retrieval_error"
    GENERATION_ERROR = "generation_error"
    INVALID_FILENAME = "invalid_filename"
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"


class RetrievalError(RuntimeError):
    """Embedding, vektor deposu veya keyword index tarafinda olusan hata."""


class GenerationError(RuntimeError):
    """Claude cagrisi sirasinda olusan hata (upstream)."""
