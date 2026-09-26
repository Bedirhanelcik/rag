from rag_tr.generation.answerer import generate_answer
from rag_tr.ingestion.chunker import Chunk


class _FakeContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeContentBlock(text)]


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(self, **kwargs):
        return _FakeMessage(self._text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessages(text)


def test_generate_answer_includes_citation_and_sources():
    chunks = [
        Chunk(text="Ankara Türkiye'nin başkentidir.", source_file="ornek.md", page_number=None, chunk_index=0)
    ]
    client = _FakeClient("Ankara Türkiye'nin başkentidir [1].")

    result = generate_answer("Türkiye'nin başkenti nedir?", chunks, client, model="test-model")

    assert "[1]" in result.answer
    assert result.sources[0]["source_file"] == "ornek.md"
    assert result.used_chunk_ids == ["ornek.md::0"]


def test_generate_answer_no_chunks_returns_not_found_message():
    client = _FakeClient("kullanılmayacak")

    result = generate_answer("soru", [], client, model="test-model")

    assert result.answer == "Dokümanlarda bu bilgi yok."
    assert result.sources == []
