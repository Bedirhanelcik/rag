"""RAGService entegrasyon testleri: ingest -> retrieval -> query akisi.

Gercek ChromaDB ve gercek BM25 kullanilir; yalnizca embedding modeli ve Claude
client'i fake'lenir, boylece ag/API key gerekmez.
"""

import pytest

from conftest import FakeAnthropicClient
from rag_tr.contracts import (
    NO_CONTEXT_MESSAGE,
    GenerationError,
    QueryStatus,
    RetrievalError,
)
from rag_tr.retrieval.keyword_search import BM25Index
from rag_tr.retrieval.vector_store import VectorStore
from rag_tr.service import RAGService

DOC = """Türkiye'nin başkenti Ankara'dır. Ankara, İç Anadolu Bölgesi'nde yer alır.

Türkiye yedi coğrafi bölgeye ayrılır. En yüksek dağı Ağrı Dağı'dır.

İstanbul Boğaz ile ikiye bölünür ve iki kıtada toprağı olan tek büyük şehirdir.
"""


def _service(settings, embedding_model, client, bm25=None) -> RAGService:
    return RAGService(
        settings,
        embedding_model=embedding_model,
        vector_store=VectorStore(settings.chroma_persist_dir),
        bm25_index=bm25 or BM25Index(),
        client=client,
    )


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_service_can_be_constructed_without_network_or_api_key(
    settings, fake_embedding_model, fake_client
):
    service = _service(settings, fake_embedding_model, fake_client)
    assert service.vector_store.count() == 0
    assert service.bm25_index.size() == 0


def test_ingest_then_query_returns_cited_answer_and_sources(
    settings, fake_embedding_model, fake_client, tmp_path
):
    service = _service(settings, fake_embedding_model, fake_client)
    path = _write(tmp_path, "cografya.md", DOC)

    ingest = service.ingest_files([path])

    assert ingest.ingested_files == ["cografya.md"]
    assert ingest.failed_files == []
    assert ingest.chunk_count > 0
    assert service.vector_store.count() == ingest.chunk_count
    assert service.bm25_index.size() == ingest.chunk_count

    result = service.query("Türkiye'nin başkenti neresidir?", top_k=3)

    assert result.answer == "Cevap burada [1]."
    assert result.used_chunk_ids
    assert all(cid.startswith("cografya.md::") for cid in result.used_chunk_ids)
    assert result.sources
    assert result.sources[0]["source_file"] == "cografya.md"
    assert len(fake_client.calls) == 1, "Claude tam olarak bir kez cagrilmali"


def test_query_on_empty_corpus_short_circuits_without_calling_claude(
    settings, fake_embedding_model, fake_client
):
    service = _service(settings, fake_embedding_model, fake_client)

    result = service.query("Herhangi bir soru?", top_k=3)

    assert result.answer == "Dokümanlarda bu bilgi yok."
    assert result.sources == []
    assert result.used_chunk_ids == []
    assert fake_client.calls == [], "Bos corpus'ta Claude cagrilmamali"


def test_reingesting_same_filename_replaces_old_chunks(
    settings, fake_embedding_model, fake_client, tmp_path
):
    settings.chunk_size = 60
    settings.chunk_overlap = 10
    service = _service(settings, fake_embedding_model, fake_client)

    long_text = "Alfa birinci paragraf metni burada.\n\nAlfa ikinci paragraf metni burada.\n\nAlfa ucuncu paragraf metni burada."
    path = _write(tmp_path, "doc.txt", long_text)
    first = service.ingest_files([path])
    assert first.chunk_count > 1, "Bu test icin birden fazla chunk gerekiyor"

    path.write_text("Beta kisa metin.", encoding="utf-8")
    service.ingest_files([path])

    assert service.vector_store.count() == 1
    assert service.bm25_index.size() == 1
    remaining = service.vector_store.get_chunks(["doc.txt::0"])
    assert "Beta" in remaining[0].text
    assert service.bm25_index.query("alfa", 10) == []


def test_unreadable_file_lands_in_failed_files(
    settings, fake_embedding_model, fake_client, tmp_path
):
    bad_pdf = tmp_path / "bozuk.pdf"
    bad_pdf.write_text("bu bir PDF degil", encoding="utf-8")
    service = _service(settings, fake_embedding_model, fake_client)

    result = service.ingest_files([bad_pdf])

    assert result.failed_files == ["bozuk.pdf"]
    assert result.ingested_files == []
    assert result.chunk_count == 0


def test_hybrid_retrieval_uses_both_vector_and_keyword_paths(
    settings, fake_embedding_model, fake_client, tmp_path
):
    service = _service(settings, fake_embedding_model, fake_client)
    path = _write(tmp_path, "cografya.md", DOC)
    service.ingest_files([path])

    service.query("Ağrı Dağı kaç metre?", top_k=3)

    context = fake_client.calls[0]["messages"][0]["content"]
    assert "Bağlam:" in context
    assert "cografya.md" in context


def test_keyword_index_survives_a_service_restart(
    settings, fake_embedding_model, fake_client, tmp_path
):
    """ChromaDB kalici oldugu halde BM25 bellekte yasiyor; yeni bir servis
    ornegi ayni dizinden kalkarken keyword index'i yeniden kurmali."""
    path = _write(tmp_path, "cografya.md", DOC)
    first = _service(settings, fake_embedding_model, fake_client)
    ingest = first.ingest_files([path])
    assert ingest.chunk_count > 0
    del first

    # Ayni kalici dizin, tamamen yeni servis + bos bir BM25Index.
    restarted = _service(settings, fake_embedding_model, fake_client, bm25=BM25Index())

    assert restarted.vector_store.count() == ingest.chunk_count
    assert restarted.bm25_index.size() == ingest.chunk_count, "BM25 restart sonrasi bos kaldi"
    assert restarted.bm25_index.query("ankara", 10), "keyword aramasi restart sonrasi calismiyor"

    result = restarted.query("Türkiye'nin başkenti neresidir?", top_k=3)
    assert result.used_chunk_ids
    assert result.answer == "Cevap burada [1]."


def test_restart_rebuild_does_not_duplicate_or_diverge(
    settings, fake_embedding_model, fake_client, tmp_path
):
    path = _write(tmp_path, "cografya.md", DOC)
    service = _service(settings, fake_embedding_model, fake_client)
    ingest = service.ingest_files([path])

    # Ayni servis uzerinde iki kez yeniden kurmak corpus'u cogaltmamali.
    service.rebuild_keyword_index()
    service.rebuild_keyword_index()

    assert service.bm25_index.size() == ingest.chunk_count
    assert service.vector_store.count() == service.bm25_index.size()
    ids = set(service.bm25_index._chunk_ids)
    assert len(ids) == ingest.chunk_count, "tekrarlanan chunk id uretildi"


def test_empty_persistent_store_rebuilds_to_empty_index(
    settings, fake_embedding_model, fake_client
):
    service = _service(settings, fake_embedding_model, fake_client)

    assert service.rebuild_keyword_index() == 0
    assert service.bm25_index.size() == 0
    assert service.bm25_index.query("herhangi", 5) == []


def test_empty_corpus_reports_no_relevant_context_status(
    settings, fake_embedding_model, fake_client
):
    service = _service(settings, fake_embedding_model, fake_client)

    result = service.query("Herhangi bir soru?", top_k=3)

    assert result.status is QueryStatus.NO_RELEVANT_CONTEXT


def test_successful_answer_reports_answered_status(
    settings, fake_embedding_model, fake_client, tmp_path
):
    service = _service(settings, fake_embedding_model, fake_client)
    service.ingest_files([_write(tmp_path, "cografya.md", DOC)])

    result = service.query("Başkent neresi?", top_k=3)

    assert result.status is QueryStatus.ANSWERED


def test_claude_saying_not_in_documents_is_reported_as_no_relevant_context(
    settings, fake_embedding_model, tmp_path
):
    """Chunk bulundu ama Claude cevap olmadigina karar verdi: agent bunu serbest
    metni parse etmeden ayirt edebilmeli."""
    client = FakeAnthropicClient(answer=NO_CONTEXT_MESSAGE)
    service = _service(settings, fake_embedding_model, client)
    service.ingest_files([_write(tmp_path, "cografya.md", DOC)])

    result = service.query("Mars'ta hava nasil?", top_k=3)

    assert result.status is QueryStatus.NO_RELEVANT_CONTEXT
    assert result.answer == NO_CONTEXT_MESSAGE
    assert result.sources == []
    assert result.used_chunk_ids == []


def test_claude_failure_surfaces_as_generation_error(
    settings, fake_embedding_model, tmp_path
):
    client = FakeAnthropicClient(error=RuntimeError("upstream patladi"))
    service = _service(settings, fake_embedding_model, client)
    service.ingest_files([_write(tmp_path, "cografya.md", DOC)])

    with pytest.raises(GenerationError):
        service.query("Başkent neresi?", top_k=3)


def test_retrieval_failure_surfaces_as_retrieval_error(
    settings, fake_embedding_model, fake_client, tmp_path
):
    service = _service(settings, fake_embedding_model, fake_client)
    service.ingest_files([_write(tmp_path, "cografya.md", DOC)])

    def _boom(_text):
        raise RuntimeError("embedding modeli patladi")

    service.embedding_model.encode_query = _boom

    with pytest.raises(RetrievalError):
        service.query("Başkent neresi?", top_k=3)
