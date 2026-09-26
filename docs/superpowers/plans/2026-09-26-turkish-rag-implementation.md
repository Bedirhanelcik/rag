# Türkçe RAG Soru-Cevap Uygulaması Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Türkçe dokümanlar üzerinde çalışan, hibrit arama (vektör + BM25 + RRF) kullanan ve Claude ile kaynak atıflı ([1], [2]) cevap üreten bir RAG soru-cevap uygulaması (FastAPI backend + Streamlit UI) inşa etmek.

**Architecture:** Katmanlı modüler yapı — `ingestion` (dosya okuma + chunking) → `retrieval` (embedding + ChromaDB + BM25 + RRF fusion) → `generation` (Claude çağrısı + citation formatlama). Bu üç katman `service.py` içinde `RAGService` ile birleştirilir; FastAPI (`api/`) bu servisi HTTP üzerinden, Streamlit (`ui/`) ise HTTP istemcisi olarak kullanır.

**Tech Stack:** Python 3.11+, FastAPI, Streamlit, sentence-transformers (`intfloat/multilingual-e5-small`), ChromaDB (persistent), rank-bm25, Anthropic SDK, pydantic-settings, pypdf, uv, Docker + docker-compose.

**Spec:** `docs/superpowers/specs/2026-09-26-turkish-rag-design.md`

## Global Constraints

- Python `>=3.11`.
- Bağımlılık yönetimi `uv` ile (`pyproject.toml` + `uv.lock`).
- Embedding modeli: `intfloat/multilingual-e5-small`, e5 konvansiyonuna göre `"query: "` / `"passage: "` prefix'leri uygulanır.
- Chunking varsayılanları: `chunk_size=1000`, `overlap=150` (karakter bazlı), asla kelime ortasından kesme, bir chunk birden fazla sayfaya yayılmaz.
- RRF: `score = Σ 1/(rank_i + k)`, varsayılan `k=60`.
- Retrieval varsayılanları: `top_k_vector=10`, `top_k_keyword=10`, `top_k_final=5`.
- Claude API key ve model adı `.env`'den okunur (`ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`), kod içine gömülmez.
- Bağlamda cevap yoksa: `"Dokümanlarda bu bilgi yok."` — LLM'e hiç gitmeden kısa yol alınabilir (boş chunk listesi durumunda).
- Testler gerçek Anthropic API key gerektirmeden çalışmalı (mock kullanılır).
- Her ana görev tamamlandığında `git commit` atılır.

---

### Task 1: Proje İskeleti ve Konfigürasyon

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/rag_tr/__init__.py`
- Create: `src/rag_tr/config.py`
- Create: `src/rag_tr/ingestion/__init__.py`
- Create: `src/rag_tr/retrieval/__init__.py`
- Create: `src/rag_tr/generation/__init__.py`
- Create: `src/rag_tr/api/__init__.py`

**Interfaces:**
- Produces: `rag_tr.config.Settings` — alanlar: `anthropic_api_key: str`, `anthropic_model: str = "claude-sonnet-5"`, `embedding_model_name: str = "intfloat/multilingual-e5-small"`, `chroma_persist_dir: str = "data/chroma"`, `chunk_size: int = 1000`, `chunk_overlap: int = 150`, `top_k_vector: int = 10`, `top_k_keyword: int = 10`, `top_k_final: int = 5`, `rrf_k: int = 60`. `.env` dosyasından `SettingsConfigDict(env_file=".env")` ile okunur (alan adları büyük harfle eşleşir, örn. `ANTHROPIC_API_KEY`).

- [ ] **Step 1: `pyproject.toml` oluştur**

```toml
[project]
name = "rag-tr"
version = "0.1.0"
description = "Turkce dokumanlar icin kaynak gosteren RAG soru-cevap uygulamasi"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "streamlit>=1.38",
    "sentence-transformers>=3.0",
    "chromadb>=0.5",
    "rank-bm25>=0.2.2",
    "anthropic>=0.34",
    "pydantic-settings>=2.4",
    "pypdf>=4.3",
    "python-multipart>=0.0.9",
    "requests>=2.32",
]

[dependency-groups]
dev = ["pytest>=8.3"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rag_tr"]
```

- [ ] **Step 2: `.gitignore` oluştur**

```
.venv/
__pycache__/
*.pyc
.env
data/uploads/
data/chroma/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 3: `.env.example` oluştur**

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-sonnet-5
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small
CHROMA_PERSIST_DIR=data/chroma
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K_VECTOR=10
TOP_K_KEYWORD=10
TOP_K_FINAL=5
RRF_K=60
```

- [ ] **Step 4: Paket iskeletini oluştur**

Boş `__init__.py` dosyaları: `src/rag_tr/__init__.py`, `src/rag_tr/ingestion/__init__.py`, `src/rag_tr/retrieval/__init__.py`, `src/rag_tr/generation/__init__.py`, `src/rag_tr/api/__init__.py`.

- [ ] **Step 5: `src/rag_tr/config.py` yaz**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-5"
    embedding_model_name: str = "intfloat/multilingual-e5-small"
    chroma_persist_dir: str = "data/chroma"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k_vector: int = 10
    top_k_keyword: int = 10
    top_k_final: int = 5
    rrf_k: int = 60
```

- [ ] **Step 6: `.env` oluştur ve bağımlılıkları kur**

```bash
cp .env.example .env
```

`.env` dosyasını açıp gerçek `ANTHROPIC_API_KEY` değerini gir (README'de kullanıcıya hatırlatılacak; bu adımda kendi key'inizi girin).

```bash
uv sync
```

Beklenen: `.venv/` oluşur, `uv.lock` oluşur, hata olmadan biter.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore .env.example src
git commit -m "chore: scaffold project structure and settings"
```

---

### Task 2: Örnek Türkçe Dokümanlar ve Eval Seti

**Files:**
- Create: `data/sample/turkiye_cografyasi.md`
- Create: `data/sample/yapay_zeka_temelleri.md`
- Create: `data/sample/osmanli_tarihi.md`
- Create: `eval/questions.json`

**Interfaces:**
- Produces: `eval/questions.json` formatı — `[{"question": str, "expected_answer": str, "source_file": str}]`, sonraki projede kullanılacak sabit bir sözleşme.

- [ ] **Step 1: `data/sample/turkiye_cografyasi.md` yaz**

```markdown
# Türkiye'nin Coğrafyası

Türkiye, Asya ve Avrupa kıtaları arasında köprü konumunda yer alan bir ülkedir. Topraklarının büyük bölümü Anadolu yarımadasında, küçük bir bölümü ise Balkan yarımadasının uzantısı olan Trakya'dadır. Türkiye'nin başkenti Ankara'dır ve ülkenin en büyük şehri İstanbul'dur.

Türkiye yedi coğrafi bölgeye ayrılır: Marmara, Ege, Akdeniz, İç Anadolu, Karadeniz, Doğu Anadolu ve Güneydoğu Anadolu. Bu bölgeler iklim, bitki örtüsü ve ekonomik faaliyetler açısından birbirinden farklılık gösterir.

Türkiye'nin en yüksek dağı, Doğu Anadolu Bölgesi'nde bulunan Ağrı Dağı'dır ve yüksekliği 5137 metredir. Ülkenin en uzun nehri Kızılırmak'tır. Türkiye'yi çevreleyen dört deniz vardır: Karadeniz, Ege Denizi, Akdeniz ve Marmara Denizi.

İstanbul, Boğaz ile ikiye bölünen ve hem Asya hem Avrupa kıtasında toprakları olan tek büyük şehirdir. Bu özelliğiyle dünyada kıtalar arası konuma sahip nadir şehirlerden biridir.
```

- [ ] **Step 2: `data/sample/yapay_zeka_temelleri.md` yaz**

```markdown
# Yapay Zeka Temelleri

Yapay zeka, bilgisayarların insan benzeri zihinsel süreçleri taklit etmesini sağlayan bilim ve mühendislik dalıdır. Makine öğrenmesi, yapay zekanın en yaygın kullanılan alt alanlarından biridir ve sistemlerin verilerden örüntüler öğrenerek tahmin yapmasını sağlar.

Derin öğrenme, çok katmanlı yapay sinir ağları kullanan bir makine öğrenmesi yöntemidir. Görüntü tanıma, doğal dil işleme ve konuşma tanıma gibi alanlarda büyük başarılar elde etmiştir.

Büyük dil modelleri (LLM), devasa miktarda metin verisiyle eğitilen ve insan diline benzer metinler üretebilen yapay zeka sistemleridir. RAG (Retrieval-Augmented Generation) yöntemi, bir dil modelinin cevap üretirken harici bir bilgi kaynağından ilgili bilgileri alarak (retrieval) bu bilgileri bağlam olarak kullanmasını sağlar. Bu yaklaşım, modelin güncel olmayan veya eksik bilgisine dayanarak yanlış bilgi üretmesini (halüsinasyon) azaltmaya yardımcı olur.

Vektör veritabanları, metinlerin sayısal temsillerini (embedding) saklayarak anlamsal benzerlik araması yapılmasına olanak tanır. ChromaDB bu amaçla kullanılan açık kaynaklı vektör veritabanlarından biridir.
```

- [ ] **Step 3: `data/sample/osmanli_tarihi.md` yaz**

```markdown
# Osmanlı Devleti'nin Kuruluşu

Osmanlı Devleti, 1299 yılında Osman Bey tarafından Anadolu'nun kuzeybatısında kurulmuştur. Devlet, kısa sürede Bizans topraklarına doğru genişlemiş ve bir beylikten imparatorluğa dönüşmüştür.

1453 yılında II. Mehmed (Fatih Sultan Mehmed) komutasındaki Osmanlı ordusu İstanbul'u fethetmiş ve Bizans İmparatorluğu'na son vermiştir. İstanbul'un fethi, Osmanlı Devleti'nin bir imparatorluğa dönüşmesinde dönüm noktası olmuştur.

Osmanlı Devleti, en geniş sınırlarına 16. yüzyılda Kanuni Sultan Süleyman döneminde ulaşmıştır. Bu dönemde devlet, Avrupa, Asya ve Afrika kıtalarında topraklara sahip olmuştur.

Osmanlı Devleti 1922 yılında saltanatın kaldırılmasıyla sona ermiş, yerine 1923 yılında Türkiye Cumhuriyeti kurulmuştur.
```

- [ ] **Step 4: `eval/questions.json` yaz**

```json
[
  {
    "question": "Türkiye'nin başkenti neresidir?",
    "expected_answer": "Ankara",
    "source_file": "turkiye_cografyasi.md"
  },
  {
    "question": "Türkiye kaç coğrafi bölgeye ayrılır?",
    "expected_answer": "Yedi coğrafi bölgeye ayrılır.",
    "source_file": "turkiye_cografyasi.md"
  },
  {
    "question": "Türkiye'nin en yüksek dağı nedir ve yüksekliği kaç metredir?",
    "expected_answer": "Ağrı Dağı, 5137 metre.",
    "source_file": "turkiye_cografyasi.md"
  },
  {
    "question": "İstanbul'un coğrafi açıdan önemi nedir?",
    "expected_answer": "Boğaz ile ikiye bölünen, hem Asya hem Avrupa kıtasında toprakları olan tek büyük şehirdir.",
    "source_file": "turkiye_cografyasi.md"
  },
  {
    "question": "RAG (Retrieval-Augmented Generation) yöntemi ne işe yarar?",
    "expected_answer": "Dil modelinin cevap üretirken harici bir bilgi kaynağından ilgili bilgileri alıp bağlam olarak kullanmasını sağlar, halüsinasyonu azaltır.",
    "source_file": "yapay_zeka_temelleri.md"
  },
  {
    "question": "Derin öğrenme nedir?",
    "expected_answer": "Çok katmanlı yapay sinir ağları kullanan bir makine öğrenmesi yöntemidir.",
    "source_file": "yapay_zeka_temelleri.md"
  },
  {
    "question": "ChromaDB nedir?",
    "expected_answer": "Metinlerin embedding'lerini saklayarak anlamsal benzerlik araması yapmaya olanak tanıyan açık kaynaklı bir vektör veritabanıdır.",
    "source_file": "yapay_zeka_temelleri.md"
  },
  {
    "question": "Osmanlı Devleti'ni kim ve hangi yıl kurmuştur?",
    "expected_answer": "Osman Bey, 1299 yılında.",
    "source_file": "osmanli_tarihi.md"
  },
  {
    "question": "İstanbul'u kim fethetmiştir ve hangi yılda?",
    "expected_answer": "II. Mehmed (Fatih Sultan Mehmed), 1453 yılında.",
    "source_file": "osmanli_tarihi.md"
  },
  {
    "question": "Osmanlı Devleti en geniş sınırlarına hangi padişah döneminde ulaşmıştır?",
    "expected_answer": "Kanuni Sultan Süleyman döneminde.",
    "source_file": "osmanli_tarihi.md"
  }
]
```

- [ ] **Step 5: Commit**

```bash
git add data/sample eval/questions.json
git commit -m "docs: add sample Turkish documents and eval question set"
```

---

### Task 3: Doküman Yükleyiciler (Ingestion — Loaders)

**Files:**
- Create: `src/rag_tr/ingestion/loaders.py`

**Interfaces:**
- Produces: `PageContent` dataclass (`text: str`, `page_number: int | None`), `load_document(path: str | Path) -> list[PageContent]` — desteklenmeyen uzantı için `ValueError` fırlatır.

- [ ] **Step 1: `src/rag_tr/ingestion/loaders.py` yaz**

```python
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
```

- [ ] **Step 2: Elle doğrula**

```bash
uv run python -c "
from rag_tr.ingestion.loaders import load_document
pages = load_document('data/sample/turkiye_cografyasi.md')
print(len(pages), pages[0].page_number, pages[0].text[:40])
"
```

Beklenen: `1 None Türkiye'nin Coğrafyası...` benzeri çıktı, hata yok.

- [ ] **Step 3: Commit**

```bash
git add src/rag_tr/ingestion/loaders.py
git commit -m "feat: add PDF/TXT/MD document loaders"
```

---

### Task 4: Chunking (TDD)

**Files:**
- Create: `src/rag_tr/ingestion/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `rag_tr.ingestion.loaders.PageContent`
- Produces: `Chunk` dataclass (`text: str`, `source_file: str`, `page_number: int | None`, `chunk_index: int`), `chunk_text(pages: list[PageContent], source_file: str, chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]`.

- [ ] **Step 1: Başarısız testleri yaz — `tests/test_chunker.py`**

```python
from rag_tr.ingestion.chunker import chunk_text
from rag_tr.ingestion.loaders import PageContent


def test_long_paragraph_splits_on_sentence_boundaries():
    paragraph = (
        "Bu birinci cümledir ve oldukça uzundur. "
        "Bu ikinci cümledir ve o da uzundur. "
        "Bu üçüncü cümledir ve sondur."
    )
    pages = [PageContent(text=paragraph, page_number=1)]
    chunks = chunk_text(pages, source_file="doc.txt", chunk_size=45, overlap=0)

    assert len(chunks) > 1
    rejoined = " ".join(c.text for c in chunks)
    for sentence in [
        "Bu birinci cümledir ve oldukça uzundur.",
        "Bu ikinci cümledir ve o da uzundur.",
        "Bu üçüncü cümledir ve sondur.",
    ]:
        assert sentence in rejoined


def test_overlap_carries_content_from_previous_chunk():
    paragraph = (
        "Cümle bir buradadır. Cümle iki buradadır. Cümle üç buradadır. "
        "Cümle dört buradadır. Cümle beş buradadır."
    )
    pages = [PageContent(text=paragraph, page_number=1)]
    chunks = chunk_text(pages, source_file="doc.txt", chunk_size=40, overlap=15)

    assert len(chunks) > 1
    first_tail_words = set(chunks[0].text.split()[-3:])
    second_head_words = set(chunks[1].text.split()[:3])
    assert first_tail_words & second_head_words


def test_chunk_metadata_across_pages():
    pages = [
        PageContent(text="Sayfa bir içeriği burada.", page_number=1),
        PageContent(text="Sayfa iki içeriği burada.", page_number=2),
    ]
    chunks = chunk_text(pages, source_file="ornek.pdf", chunk_size=1000, overlap=0)

    assert len(chunks) == 2
    assert chunks[0].source_file == "ornek.pdf"
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert chunks[1].page_number == 2
    assert chunks[1].chunk_index == 1
```

- [ ] **Step 2: Testlerin başarısız olduğunu doğrula**

```bash
uv run pytest tests/test_chunker.py -v
```

Beklenen: `ModuleNotFoundError` veya `ImportError` — `chunker.py` henüz yok.

- [ ] **Step 3: `src/rag_tr/ingestion/chunker.py` yaz**

```python
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
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

```bash
uv run pytest tests/test_chunker.py -v
```

Beklenen: 3/3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_tr/ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: add sentence-boundary-aware chunker with tests"
```

---

### Task 5: Embedding Modeli

**Files:**
- Create: `src/rag_tr/retrieval/embeddings.py`

**Interfaces:**
- Produces: `EmbeddingModel` sınıfı — `__init__(model_name: str)`, `encode_passages(texts: list[str]) -> np.ndarray`, `encode_query(text: str) -> np.ndarray`.

- [ ] **Step 1: `src/rag_tr/retrieval/embeddings.py` yaz**

```python
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"passage: {text}" for text in texts]
        return self._model.encode(prefixed, normalize_embeddings=True)

    def encode_query(self, text: str) -> np.ndarray:
        embeddings = self._model.encode([f"query: {text}"], normalize_embeddings=True)
        return embeddings[0]
```

- [ ] **Step 2: Elle doğrula (model indirme gerektirir, ilk çalıştırma internet ister)**

```bash
uv run python -c "
from rag_tr.retrieval.embeddings import EmbeddingModel
model = EmbeddingModel('intfloat/multilingual-e5-small')
vec = model.encode_query('Türkiye nin başkenti neresidir')
print(vec.shape)
"
```

Beklenen: `(384,)` şeklinde bir çıktı (model boyutuna bağlı), hata yok.

- [ ] **Step 3: Commit**

```bash
git add src/rag_tr/retrieval/embeddings.py
git commit -m "feat: add multilingual embedding model wrapper"
```

---

### Task 6: Vektör Deposu (ChromaDB)

**Files:**
- Create: `src/rag_tr/retrieval/vector_store.py`

**Interfaces:**
- Consumes: `rag_tr.ingestion.chunker.Chunk`
- Produces: `make_chunk_id(source_file: str, chunk_index: int) -> str`, `VectorStore` sınıfı — `__init__(persist_dir: str)`, `add(chunks: list[Chunk], embeddings: np.ndarray) -> None`, `query(query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]`, `get_chunks(chunk_ids: list[str]) -> list[Chunk]`, `count() -> int`.

- [ ] **Step 1: `src/rag_tr/retrieval/vector_store.py` yaz**

```python
import numpy as np
import chromadb

from rag_tr.ingestion.chunker import Chunk


def make_chunk_id(source_file: str, chunk_index: int) -> str:
    return f"{source_file}::{chunk_index}"


class VectorStore:
    def __init__(self, persist_dir: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection("chunks")

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        ids = [make_chunk_id(c.source_file, c.chunk_index) for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "page_number": c.page_number if c.page_number is not None else -1,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        documents = [c.text for c in chunks]
        self._collection.add(
            ids=ids,
            embeddings=np.asarray(embeddings).tolist(),
            metadatas=metadatas,
            documents=documents,
        )

    def query(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(query_embedding).tolist()],
            n_results=min(top_k, count),
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return list(zip(ids, distances))

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        result = self._collection.get(ids=chunk_ids)
        by_id = {
            id_: Chunk(
                text=doc,
                source_file=meta["source_file"],
                page_number=None if meta["page_number"] == -1 else meta["page_number"],
                chunk_index=meta["chunk_index"],
            )
            for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        }
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def count(self) -> int:
        return self._collection.count()
```

- [ ] **Step 2: Elle doğrula**

```bash
uv run python -c "
from rag_tr.retrieval.vector_store import VectorStore, make_chunk_id
from rag_tr.ingestion.chunker import Chunk
from rag_tr.retrieval.embeddings import EmbeddingModel
import shutil

shutil.rmtree('data/chroma_test', ignore_errors=True)
model = EmbeddingModel('intfloat/multilingual-e5-small')
store = VectorStore('data/chroma_test')
chunks = [Chunk(text='Ankara Türkiye nin başkentidir.', source_file='t.md', page_number=None, chunk_index=0)]
store.add(chunks, model.encode_passages([c.text for c in chunks]))
results = store.query(model.encode_query('Türkiye nin başkenti nedir'), top_k=1)
print(results)
print(store.get_chunks([results[0][0]]))
shutil.rmtree('data/chroma_test', ignore_errors=True)
"
```

Beklenen: bir `(chunk_id, distance)` çifti ve ardından ilgili `Chunk` nesnesi yazdırılır, hata yok.

- [ ] **Step 3: Commit**

```bash
git add src/rag_tr/retrieval/vector_store.py
git commit -m "feat: add ChromaDB-backed vector store"
```

---

### Task 7: Anahtar Kelime Arama (BM25)

**Files:**
- Create: `src/rag_tr/retrieval/keyword_search.py`

**Interfaces:**
- Consumes: `rag_tr.ingestion.chunker.Chunk`, `rag_tr.retrieval.vector_store.make_chunk_id`
- Produces: `BM25Index` sınıfı — `add(chunks: list[Chunk]) -> None`, `query(text: str, top_k: int) -> list[tuple[str, float]]`.

- [ ] **Step 1: `src/rag_tr/retrieval/keyword_search.py` yaz**

```python
import re

from rank_bm25 import BM25Okapi

from rag_tr.ingestion.chunker import Chunk
from rag_tr.retrieval.vector_store import make_chunk_id

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunk_ids.append(make_chunk_id(chunk.source_file, chunk.chunk_index))
            self._chunks.append(chunk)
        tokenized = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def query(self, text: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda item: item[1], reverse=True)
        return [(chunk_id, score) for chunk_id, score in ranked[:top_k] if score > 0]
```

- [ ] **Step 2: Elle doğrula**

```bash
uv run python -c "
from rag_tr.retrieval.keyword_search import BM25Index
from rag_tr.ingestion.chunker import Chunk

index = BM25Index()
index.add([
    Chunk(text='Ankara Türkiye nin başkentidir.', source_file='t.md', page_number=None, chunk_index=0),
    Chunk(text='İstanbul en büyük şehirdir.', source_file='t.md', page_number=None, chunk_index=1),
])
print(index.query('başkenti nedir', top_k=2))
"
```

Beklenen: skor sıralı `[(chunk_id, score), ...]` listesi, `t.md::0` en üstte.

- [ ] **Step 3: Commit**

```bash
git add src/rag_tr/retrieval/keyword_search.py
git commit -m "feat: add BM25 keyword search index"
```

---

### Task 8: Hibrit Arama — Reciprocal Rank Fusion (TDD)

**Files:**
- Create: `src/rag_tr/retrieval/hybrid.py`
- Test: `tests/test_hybrid.py`

**Interfaces:**
- Produces: `reciprocal_rank_fusion(vector_results: list[tuple[str, float]], keyword_results: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]` — chunk_id'leri füzyon skoruna göre azalan sırada döner.

- [ ] **Step 1: Başarısız testi yaz — `tests/test_hybrid.py`**

```python
from rag_tr.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_favors_items_ranked_high_in_both_lists():
    vector_results = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    keyword_results = [("b", 5.0), ("a", 4.0), ("d", 3.0)]

    fused = reciprocal_rank_fusion(vector_results, keyword_results, k=60)
    fused_ids = [chunk_id for chunk_id, _ in fused]

    assert fused_ids[0] in ("a", "b")
    assert set(fused_ids) == {"a", "b", "c", "d"}
    assert fused_ids.index("c") > fused_ids.index("a")


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([], [], k=60) == []
    assert reciprocal_rank_fusion([("a", 1.0)], [], k=60) == [("a", 1 / 61)]
```

- [ ] **Step 2: Testin başarısız olduğunu doğrula**

```bash
uv run pytest tests/test_hybrid.py -v
```

Beklenen: `ModuleNotFoundError` — `hybrid.py` henüz yok.

- [ ] **Step 3: `src/rag_tr/retrieval/hybrid.py` yaz**

```python
from collections import defaultdict


def reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    keyword_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for rank, (chunk_id, _score) in enumerate(vector_results, start=1):
        scores[chunk_id] += 1 / (rank + k)
    for rank, (chunk_id, _score) in enumerate(keyword_results, start=1):
        scores[chunk_id] += 1 / (rank + k)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

```bash
uv run pytest tests/test_hybrid.py -v
```

Beklenen: 2/2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_tr/retrieval/hybrid.py tests/test_hybrid.py
git commit -m "feat: add reciprocal rank fusion for hybrid search with tests"
```

---

### Task 9: Cevap Üretimi (Generation, mock ile TDD)

**Files:**
- Create: `src/rag_tr/generation/prompts.py`
- Create: `src/rag_tr/generation/answerer.py`
- Test: `tests/test_answerer.py`

**Interfaces:**
- Consumes: `rag_tr.ingestion.chunker.Chunk`, `rag_tr.retrieval.vector_store.make_chunk_id`
- Produces: `SYSTEM_PROMPT: str`, `format_context(chunks: list[Chunk]) -> str`, `AnswerResult` dataclass (`answer: str`, `sources: list[dict]`, `used_chunk_ids: list[str]`), `generate_answer(question: str, chunks: list[Chunk], client, model: str) -> AnswerResult`.

- [ ] **Step 1: `src/rag_tr/generation/prompts.py` yaz**

```python
from rag_tr.ingestion.chunker import Chunk

SYSTEM_PROMPT = (
    "Sen Türkçe dokümanlar üzerinde çalışan bir soru-cevap asistanısın. "
    "Sadece sana verilen bağlamdaki bilgileri kullanarak cevap ver. "
    "Her iddiana, bilgiyi aldığın kaynağı köşeli parantez içinde numarayla belirt, "
    "örneğin [1] veya [2]. Bağlamda soruya cevap verecek bilgi yoksa, kesinlikle "
    "bilgi uydurma; sadece 'Dokümanlarda bu bilgi yok.' şeklinde cevap ver."
)


def format_context(chunks: list[Chunk]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        page_info = f", sayfa {chunk.page_number}" if chunk.page_number is not None else ""
        lines.append(f"[{i}] ({chunk.source_file}{page_info}): {chunk.text}")
    return "\n\n".join(lines)
```

- [ ] **Step 2: Başarısız testleri yaz — `tests/test_answerer.py`**

```python
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
```

- [ ] **Step 3: Testlerin başarısız olduğunu doğrula**

```bash
uv run pytest tests/test_answerer.py -v
```

Beklenen: `ModuleNotFoundError` — `answerer.py` henüz yok.

- [ ] **Step 4: `src/rag_tr/generation/answerer.py` yaz**

```python
from dataclasses import dataclass

from rag_tr.generation.prompts import SYSTEM_PROMPT, format_context
from rag_tr.ingestion.chunker import Chunk
from rag_tr.retrieval.vector_store import make_chunk_id


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict]
    used_chunk_ids: list[str]


def generate_answer(question: str, chunks: list[Chunk], client, model: str) -> AnswerResult:
    if not chunks:
        return AnswerResult(answer="Dokümanlarda bu bilgi yok.", sources=[], used_chunk_ids=[])

    context = format_context(chunks)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {question}"}],
    )
    answer_text = message.content[0].text

    sources = [
        {
            "index": i,
            "source_file": chunk.source_file,
            "page_number": chunk.page_number,
            "text": chunk.text,
        }
        for i, chunk in enumerate(chunks, start=1)
    ]
    used_chunk_ids = [make_chunk_id(c.source_file, c.chunk_index) for c in chunks]
    return AnswerResult(answer=answer_text, sources=sources, used_chunk_ids=used_chunk_ids)
```

- [ ] **Step 5: Testlerin geçtiğini doğrula**

```bash
uv run pytest tests/test_answerer.py -v
```

Beklenen: 2/2 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rag_tr/generation tests/test_answerer.py
git commit -m "feat: add citation-aware answer generation with mocked tests"
```

---

### Task 10: RAGService — Katmanları Birleştirme

**Files:**
- Create: `src/rag_tr/service.py`

**Interfaces:**
- Consumes: `Settings`, `load_document`, `chunk_text`, `EmbeddingModel`, `VectorStore`, `BM25Index`, `reciprocal_rank_fusion`, `generate_answer`, `AnswerResult`.
- Produces: `IngestResult` dataclass (`ingested_files: list[str]`, `failed_files: list[str]`, `chunk_count: int`), `RAGService` sınıfı — `__init__(settings: Settings)`, `ingest_files(paths: list[Path]) -> IngestResult`, `query(question: str, top_k: int) -> AnswerResult`.

- [ ] **Step 1: `src/rag_tr/service.py` yaz**

```python
from dataclasses import dataclass
from pathlib import Path

import anthropic

from rag_tr.config import Settings
from rag_tr.generation.answerer import AnswerResult, generate_answer
from rag_tr.ingestion.chunker import chunk_text
from rag_tr.ingestion.loaders import load_document
from rag_tr.retrieval.embeddings import EmbeddingModel
from rag_tr.retrieval.hybrid import reciprocal_rank_fusion
from rag_tr.retrieval.keyword_search import BM25Index
from rag_tr.retrieval.vector_store import VectorStore


@dataclass
class IngestResult:
    ingested_files: list[str]
    failed_files: list[str]
    chunk_count: int


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding_model = EmbeddingModel(settings.embedding_model_name)
        self.vector_store = VectorStore(settings.chroma_persist_dir)
        self.bm25_index = BM25Index()
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def ingest_files(self, saved_paths: list[Path]) -> IngestResult:
        ingested: list[str] = []
        failed: list[str] = []
        total_chunks = 0

        for path in saved_paths:
            try:
                pages = load_document(path)
            except ValueError:
                failed.append(path.name)
                continue

            chunks = chunk_text(
                pages,
                source_file=path.name,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )
            if chunks:
                embeddings = self.embedding_model.encode_passages([c.text for c in chunks])
                self.vector_store.add(chunks, embeddings)
                self.bm25_index.add(chunks)
                total_chunks += len(chunks)
            ingested.append(path.name)

        return IngestResult(ingested_files=ingested, failed_files=failed, chunk_count=total_chunks)

    def query(self, question: str, top_k: int) -> AnswerResult:
        if self.vector_store.count() == 0:
            return AnswerResult(answer="Dokümanlarda bu bilgi yok.", sources=[], used_chunk_ids=[])

        query_embedding = self.embedding_model.encode_query(question)
        vector_results = self.vector_store.query(query_embedding, self.settings.top_k_vector)
        keyword_results = self.bm25_index.query(question, self.settings.top_k_keyword)
        fused = reciprocal_rank_fusion(vector_results, keyword_results, k=self.settings.rrf_k)
        top_ids = [chunk_id for chunk_id, _ in fused[:top_k]]

        if not top_ids:
            return AnswerResult(answer="Dokümanlarda bu bilgi yok.", sources=[], used_chunk_ids=[])

        chunks = self.vector_store.get_chunks(top_ids)
        return generate_answer(question, chunks, self.client, self.settings.anthropic_model)
```

- [ ] **Step 2: Elle doğrula (gerçek ANTHROPIC_API_KEY gerektirir, `.env` doldurulmuş olmalı)**

```bash
uv run python -c "
import shutil
from pathlib import Path
from rag_tr.config import Settings
from rag_tr.service import RAGService

shutil.rmtree('data/chroma_test', ignore_errors=True)
settings = Settings(chroma_persist_dir='data/chroma_test')
service = RAGService(settings)
result = service.ingest_files([Path('data/sample/turkiye_cografyasi.md')])
print(result)
answer = service.query('Türkiye nin başkenti neresidir?', top_k=3)
print(answer.answer)
print(answer.sources)
shutil.rmtree('data/chroma_test', ignore_errors=True)
"
```

Beklenen: ingest sonucu `chunk_count > 0`, cevap metninde `Ankara` geçer ve `[1]` gibi bir atıf içerir.

- [ ] **Step 3: Commit**

```bash
git add src/rag_tr/service.py
git commit -m "feat: add RAGService tying ingestion, retrieval and generation together"
```

---

### Task 11: FastAPI Uç Noktaları

**Files:**
- Create: `src/rag_tr/api/schemas.py`
- Create: `src/rag_tr/api/routes.py`
- Create: `src/rag_tr/api/main.py`

**Interfaces:**
- Consumes: `RAGService`, `Settings`, `IngestResult`, `AnswerResult`.
- Produces: `POST /ingest`, `POST /query`, `GET /health` uç noktaları; `app = create_app()` FastAPI örneği (`rag_tr.api.main:app`).

- [ ] **Step 1: `src/rag_tr/api/schemas.py` yaz**

```python
from pydantic import BaseModel


class IngestResponse(BaseModel):
    ingested_files: list[str]
    failed_files: list[str]
    chunk_count: int


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceItem(BaseModel):
    index: int
    source_file: str
    page_number: int | None
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    used_chunk_ids: list[str]


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    chunk_count: int
```

- [ ] **Step 2: `src/rag_tr/api/routes.py` yaz**

```python
from pathlib import Path

import anthropic
from fastapi import APIRouter, HTTPException, Request, UploadFile

from rag_tr.api.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
)

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, files: list[UploadFile]) -> IngestResponse:
    service = request.app.state.service
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for file in files:
        dest = UPLOAD_DIR / file.filename
        dest.write_bytes(await file.read())
        saved_paths.append(dest)

    result = service.ingest_files(saved_paths)
    return IngestResponse(
        ingested_files=result.ingested_files,
        failed_files=result.failed_files,
        chunk_count=result.chunk_count,
    )


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    service = request.app.state.service
    top_k = payload.top_k or service.settings.top_k_final
    try:
        result = service.query(payload.question, top_k)
    except anthropic.APIError as exc:
        raise HTTPException(status_code=502, detail=f"Claude API hatası: {exc}") from exc
    return QueryResponse(
        answer=result.answer,
        sources=[SourceItem(**source) for source in result.sources],
        used_chunk_ids=result.used_chunk_ids,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    service = request.app.state.service
    return HealthResponse(
        status="ok",
        embedding_model=service.settings.embedding_model_name,
        chunk_count=service.vector_store.count(),
    )
```

- [ ] **Step 3: `src/rag_tr/api/main.py` yaz**

```python
from pathlib import Path

from fastapi import FastAPI

from rag_tr.api.routes import router
from rag_tr.config import Settings
from rag_tr.service import RAGService


def create_app() -> FastAPI:
    app = FastAPI(title="Türkçe RAG API")
    settings = Settings()
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    Path("data/uploads").mkdir(parents=True, exist_ok=True)
    app.state.service = RAGService(settings)
    app.include_router(router)
    return app


app = create_app()
```

- [ ] **Step 4: Elle doğrula (gerçek `.env` ile sunucuyu başlat)**

```bash
uv run uvicorn rag_tr.api.main:app --port 8000 &
sleep 3
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ingest -F "files=@data/sample/turkiye_cografyasi.md"
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "Türkiye nin başkenti neresidir?"}'
kill %1
```

Beklenen: `/health` `{"status":"ok",...}`, `/ingest` chunk sayısı > 0, `/query` cevabında "Ankara" ve `[1]` atıfı.

- [ ] **Step 5: Commit**

```bash
git add src/rag_tr/api
git commit -m "feat: add FastAPI ingest/query/health endpoints"
```

---

### Task 12: Streamlit Arayüzü

**Files:**
- Create: `ui/streamlit_app.py`

**Interfaces:**
- Consumes: `/ingest`, `/query` HTTP uç noktaları (ortam değişkeni `API_URL`, varsayılan `http://localhost:8000`).

- [ ] **Step 1: `ui/streamlit_app.py` yaz**

```python
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Türkçe RAG Soru-Cevap")
st.title("Türkçe RAG Soru-Cevap")

with st.sidebar:
    st.header("Doküman Yükle")
    uploaded_files = st.file_uploader(
        "PDF, TXT veya MD dosyaları", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Yükle") and uploaded_files:
        files_payload = [("files", (f.name, f.getvalue())) for f in uploaded_files]
        response = requests.post(f"{API_URL}/ingest", files=files_payload)
        if response.ok:
            data = response.json()
            st.success(f"{len(data['ingested_files'])} dosya işlendi, {data['chunk_count']} chunk oluşturuldu.")
            if data["failed_files"]:
                st.warning(f"İşlenemeyen dosyalar: {', '.join(data['failed_files'])}")
        else:
            st.error(f"Yükleme başarısız: {response.text}")

question = st.text_input("Sorunuzu yazın")
if st.button("Sor") and question:
    response = requests.post(f"{API_URL}/query", json={"question": question})
    if response.ok:
        data = response.json()
        st.markdown(data["answer"])
        for source in data["sources"]:
            page_suffix = f" — sayfa {source['page_number']}" if source["page_number"] else ""
            with st.expander(f"[{source['index']}] {source['source_file']}{page_suffix}"):
                st.write(source["text"])
    else:
        st.error(f"Sorgu başarısız: {response.text}")
```

- [ ] **Step 2: Elle doğrula**

Backend'i ayrı bir terminalde çalıştır (`uv run uvicorn rag_tr.api.main:app --port 8000`), ardından:

```bash
uv run streamlit run ui/streamlit_app.py
```

Tarayıcıda `data/sample/turkiye_cografyasi.md` dosyasını yükle, "Türkiye'nin başkenti neresidir?" sorusunu sor. Beklenen: cevap altında `[1] turkiye_cografyasi.md` açılabilir kaynak kutusu görünür.

- [ ] **Step 3: Commit**

```bash
git add ui/streamlit_app.py
git commit -m "feat: add Streamlit UI for upload and Q&A"
```

---

### Task 13: Docker ve docker-compose

**Files:**
- Create: `Dockerfile.api`
- Create: `Dockerfile.ui`
- Create: `docker-compose.yml`

**Interfaces:**
- Produces: `api` servisi (port 8000), `ui` servisi (port 8501, `API_URL=http://api:8000`), `./data` bind mount ile kalıcılık.

- [ ] **Step 1: `Dockerfile.api` yaz**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "rag_tr.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: `Dockerfile.ui` yaz**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY ui ./ui
RUN uv sync --frozen
ENV PATH="/app/.venv/bin:$PATH"
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0"]
```

- [ ] **Step 3: `docker-compose.yml` yaz**

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./data:/app/data

  ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
```

- [ ] **Step 4: Elle doğrula**

```bash
docker compose up --build -d
sleep 15
curl http://localhost:8000/health
docker compose down
```

Beklenen: `{"status":"ok",...}` yanıtı; `docker compose down` ile temiz kapanış.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile.api Dockerfile.ui docker-compose.yml
git commit -m "feat: add Docker and docker-compose setup"
```

---

### Task 14: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: `README.md` yaz**

Aşağıdaki bölümleri içerecek şekilde yaz: Problem tanımı; Mermaid mimari diyagramı (ingestion → retrieval → generation → API → UI akışını gösteren); Kurulum (`uv sync`, `.env` doldurma, `uv run uvicorn ...`, `uv run streamlit run ...`, ve `docker compose up --build` alternatifi); Kullanım (dosya yükleme, soru sorma örneği, `curl` örnekleri); **Tasarım Kararları** (neden hibrit arama — salt vektör aramanın nadir terimlerde/özel adlarda zayıf kaldığı, BM25'in bunu tamamladığı; neden `chunk_size=1000`/`overlap=150` — Türkçe paragrafların ortalama uzunluğuna ve embedding modelinin bağlam penceresine uygunluk; neden yerel embedding modeli — API maliyeti/gecikmesi olmadan Türkçe çok dilli destek; neden RRF — farklı ölçeklerdeki (cosine similarity vs. BM25 skoru) sonuçları normalize etmeden, sıralamaya dayalı adil biçimde birleştirmesi); **Geliştirme Fikirleri** (reranker modeli eklenmesi, streaming yanıt, BM25 index'inin diske kalıcı serialize edilmesi, çoklu kullanıcı/oturum desteği, semantic caching, değerlendirme otomasyonu `eval/questions.json` üzerinden).

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with architecture diagram and design decisions"
```

---

### Task 15: Uçtan Uca Doğrulama

**Files:** (değişiklik yok — sadece doğrulama)

- [ ] **Step 1: Tüm testleri çalıştır**

```bash
uv run pytest -v
```

Beklenen: tüm testler PASS (chunker: 3, hybrid: 2, answerer: 2).

- [ ] **Step 2: docker-compose ile uçtan uca dene**

```bash
docker compose up --build -d
sleep 15
curl -X POST http://localhost:8000/ingest \
  -F "files=@data/sample/turkiye_cografyasi.md" \
  -F "files=@data/sample/yapay_zeka_temelleri.md" \
  -F "files=@data/sample/osmanli_tarihi.md"
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG yontemi ne ise yarar?"}'
```

Tarayıcıda `http://localhost:8501` açılıp UI üzerinden en az bir soru sorularak kaynak kutucuklarının doğru göründüğü teyit edilir.

```bash
docker compose down
```

- [ ] **Step 3: Son commit**

```bash
git add -A
git status
git commit -m "chore: final verification pass" --allow-empty
```

(Eğer Step 1-2 sırasında herhangi bir dosya değişmediyse bu commit boş geçilebilir; değişiklik varsa normal commit yapılır.)
