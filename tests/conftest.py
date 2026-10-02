"""Testler icin paylasilan fake bagimliliklar.

Hicbiri ag erisimi veya gercek bir Anthropic API key'i gerektirmez: embedding
modeli deterministik bir token-hash vektoru uretir, Claude client'i sabit bir
yanit dondurur. Vektor deposu gercek ChromaDB'dir ama tmp_path altinda calisir.
"""

import hashlib
import re

import numpy as np
import pytest

from rag_tr.config import Settings

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
EMBED_DIM = 48


class FakeEmbeddingModel:
    """Token'lari sabit bir boyuta hash'leyip normalize eder. Boylece sozcuk
    ortusmesi olan metinler birbirine yakin vektorler alir -- gercek modelin
    anlamsal yakinligini taklit etmez ama retrieval akisini uctan uca surer."""

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vec[digest[0] % EMBED_DIM] += 1.0
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class _FakeContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeContentBlock(text)]


class _FakeMessages:
    def __init__(self, parent: "FakeAnthropicClient") -> None:
        self._parent = parent

    def create(self, **kwargs):
        self._parent.calls.append(kwargs)
        if self._parent.error is not None:
            raise self._parent.error
        return _FakeMessage(self._parent.answer)


class FakeAnthropicClient:
    """Gercek API'yi cagirmaz. `error` verilirse create() onu firlatir."""

    def __init__(self, answer: str = "Cevap burada [1].", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[dict] = []
        self.messages = _FakeMessages(self)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Gercek .env'den bagimsiz, tamamen izole ayarlar."""
    return Settings(
        anthropic_api_key="test-key-not-real",
        anthropic_model="claude-test",
        chroma_persist_dir=str(tmp_path / "chroma"),
        chunk_size=1000,
        chunk_overlap=150,
    )


@pytest.fixture
def fake_embedding_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture
def fake_client() -> FakeAnthropicClient:
    return FakeAnthropicClient()
