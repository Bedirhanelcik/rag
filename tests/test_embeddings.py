"""Gemini embedding katmani testleri.

Gercek bir API cagrisi yapilmaz: genai istemcisi enjekte edilir. Hicbir yerel
ML modeli yuklenmez.
"""

import numpy as np
import pytest

from rag_tr.retrieval.embeddings import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingModel,
)


class _Embedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _Response:
    def __init__(self, embeddings: list[_Embedding]) -> None:
        self.embeddings = embeddings


class _Models:
    """genai `client.models` yuzeyinin taklidi."""

    def __init__(self, dimensions: int, *, drop: bool = False) -> None:
        self.calls: list[dict] = []
        self._dimensions = dimensions
        self._drop = drop

    def embed_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        count = len(contents) - 1 if self._drop else len(contents)
        # Her metin icin normalize EDILMEMIS, farkli uzunlukta bir vektor uret:
        # normalizasyonun gercekten uygulandigini gorebilmek icin.
        return _Response(
            [
                _Embedding([float(index + 1)] * self._dimensions)
                for index in range(max(0, count))
            ]
        )


class FakeGenaiClient:
    def __init__(self, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS, *, drop: bool = False) -> None:
        self.models = _Models(dimensions, drop=drop)


def _model(dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS, **kwargs) -> tuple[EmbeddingModel, FakeGenaiClient]:
    client = FakeGenaiClient(dimensions)
    return (
        EmbeddingModel("gemini-embedding-001", dimensions=dimensions, client=client, **kwargs),
        client,
    )


# --- tembel kurulum: acilista ag/anahtar yok ---


def test_constructing_the_model_builds_no_client_and_needs_no_key():
    model = EmbeddingModel("gemini-embedding-001")

    assert model.model_name == "gemini-embedding-001"
    assert model.dimensions == DEFAULT_EMBEDDING_DIMENSIONS
    # Hicbir istemci kurulmadi; anahtar da istenmedi.
    assert model._client is None


def test_missing_key_raises_a_clear_error_only_on_first_use():
    model = EmbeddingModel("gemini-embedding-001", api_key=None)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        model.encode_query("soru")


def test_empty_passage_list_makes_no_api_call():
    model, client = _model()

    result = model.encode_passages([])

    assert result.shape == (0, DEFAULT_EMBEDDING_DIMENSIONS)
    assert client.models.calls == []


# --- asimetrik kodlama (e5 oneklerinin karsiligi) ---


def test_passages_use_the_document_task_type():
    model, client = _model()

    model.encode_passages(["birinci pasaj", "ikinci pasaj"])

    call = client.models.calls[0]
    assert call["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert call["contents"] == ["birinci pasaj", "ikinci pasaj"]


def test_queries_use_the_query_task_type():
    model, client = _model()

    model.encode_query("Türkiye'nin başkenti neresidir?")

    call = client.models.calls[0]
    assert call["config"].task_type == "RETRIEVAL_QUERY"
    assert call["contents"] == ["Türkiye'nin başkenti neresidir?"]


def test_model_name_and_dimensions_are_passed_through():
    client = FakeGenaiClient(256)
    model = EmbeddingModel("gemini-embedding-001", dimensions=256, client=client)

    model.encode_query("soru")

    call = client.models.calls[0]
    assert call["model"] == "gemini-embedding-001"
    assert call["config"].output_dimensionality == 256


# --- sekil ve normalizasyon ---


def test_encode_passages_returns_one_row_per_text():
    model, _ = _model()

    result = model.encode_passages(["a", "b", "c"])

    assert result.shape == (3, DEFAULT_EMBEDDING_DIMENSIONS)
    assert result.dtype == np.float32


def test_encode_query_returns_a_single_vector():
    model, _ = _model()

    result = model.encode_query("soru")

    assert result.shape == (DEFAULT_EMBEDDING_DIMENSIONS,)


def test_vectors_are_l2_normalised():
    """Chroma varsayilan L2 uzakligi kullaniyor; birim vektorlerde L2 siralamasi
    cosine siralamasiyla ayni oldugu icin normalizasyon mevcut siralama
    anlamini koruyor."""
    model, _ = _model()

    passages = model.encode_passages(["a", "b", "c"])
    query = model.encode_query("soru")

    for row in passages:
        assert np.isclose(np.linalg.norm(row), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(query), 1.0, atol=1e-5)


def test_zero_vectors_survive_normalisation_without_dividing_by_zero():
    class _ZeroModels(_Models):
        def embed_content(self, *, model, contents, config):
            self.calls.append({"model": model, "contents": list(contents), "config": config})
            return _Response([_Embedding([0.0] * self._dimensions) for _ in contents])

    client = FakeGenaiClient()
    client.models = _ZeroModels(DEFAULT_EMBEDDING_DIMENSIONS)
    model = EmbeddingModel("gemini-embedding-001", client=client)

    result = model.encode_passages(["a"])

    assert not np.isnan(result).any()
    assert np.allclose(result, 0.0)


# --- toplu istek bolme ---


def test_large_inputs_are_split_into_batches():
    model, client = _model(batch_size=2)

    result = model.encode_passages(["a", "b", "c", "d", "e"])

    assert result.shape == (5, DEFAULT_EMBEDDING_DIMENSIONS)
    assert [len(call["contents"]) for call in client.models.calls] == [2, 2, 1]


def test_single_batch_makes_one_call():
    model, client = _model()

    model.encode_passages(["a", "b", "c"])

    assert len(client.models.calls) == 1


# --- bozuk yanit ---


def test_short_response_is_rejected_rather_than_silently_misaligned():
    client = FakeGenaiClient(drop=True)
    model = EmbeddingModel("gemini-embedding-001", client=client)

    with pytest.raises(RuntimeError, match="beklenen sayıda"):
        model.encode_passages(["a", "b", "c"])


# --- varsayilanlar ---


def test_default_model_is_a_gemini_embedding_model():
    assert DEFAULT_EMBEDDING_MODEL.startswith("gemini-embedding")


def test_settings_default_to_the_gemini_embedding_model():
    from rag_tr.config import Settings

    settings = Settings(_env_file=None)

    assert settings.embedding_model_name == DEFAULT_EMBEDDING_MODEL
    assert settings.embedding_dimensions == DEFAULT_EMBEDDING_DIMENSIONS
    assert settings.gemini_api_key is None


def test_no_local_ml_runtime_is_importable_or_needed():
    """torch/transformers artik bagimlilik degil; embedding yolu bunlari
    import etmiyor."""
    import sys

    import rag_tr.retrieval.embeddings  # noqa: F401

    for forbidden in ("torch", "transformers", "sentence_transformers"):
        assert forbidden not in sys.modules, f"{forbidden} import edilmemeli"


def test_huggingface_style_model_name_gives_a_clear_error():
    """Eski EMBEDDING_MODEL_NAME degeri (HF model adi) birakilmissa anlamsiz
    bir API hatasi yerine ne yapilacagini soyleyen bir hata alinmali."""
    client = FakeGenaiClient()
    model = EmbeddingModel("intfloat/multilingual-e5-small", client=client)

    with pytest.raises(RuntimeError, match="Gemini embedding modeli"):
        model.encode_query("soru")

    assert client.models.calls == [], "gecersiz model adiyla istek atilmamali"
