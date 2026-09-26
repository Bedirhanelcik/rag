# Türkçe RAG Soru-Cevap Uygulaması — Tasarım

## Amaç ve Kapsam

Türkçe dokümanlar üzerinde çalışan, kaynak gösteren (citation) bir RAG (Retrieval-Augmented Generation) soru-cevap uygulaması. Portfolyo projesi — 1-2 saatlik geliştirme kapsamında, gereksiz özellik eklenmeden, profesyonel kod kalitesiyle. Sonuçları `eval/questions.json` ile ikinci bir projede (değerlendirme/kıyaslama) kullanılacak, bu yüzden eval seti stabil ve tekrarlanabilir olmalı.

## Teknik Yığın

- Python 3.11+
- Backend: FastAPI
- UI: Streamlit
- Embedding: `sentence-transformers` ile `intfloat/multilingual-e5-small` (yerel, çok dilli, Türkçe destekli)
- Vektör DB: ChromaDB (`PersistentClient`, diske yazan)
- Anahtar kelime arama: `rank_bm25`
- LLM: Anthropic Claude API (`anthropic` SDK), API key + model adı `.env`'den
- Bağımlılık yönetimi: `uv` (pyproject.toml + uv.lock)
- Konteynerleştirme: Docker + docker-compose (2 servis: `api`, `ui`)

## Proje Yapısı

```
RAG/
├── src/rag_tr/
│   ├── __init__.py
│   ├── config.py                # pydantic-settings ile .env okuma
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py           # PDF/TXT/MD -> ham metin + sayfa bilgisi
│   │   └── chunker.py           # cümle/paragraf sınırlı chunking
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # sentence-transformers wrapper
│   │   ├── vector_store.py      # ChromaDB add/query
│   │   ├── keyword_search.py    # BM25 index + query
│   │   └── hybrid.py            # RRF fusion
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompts.py           # sistem promptu + bağlam formatlama
│   │   └── answerer.py          # Claude çağrısı, citation'lı cevap
│   └── api/
│       ├── __init__.py
│       ├── main.py              # FastAPI app, lifespan (model/db init)
│       ├── schemas.py           # Pydantic request/response modelleri
│       └── routes.py            # /ingest, /query, /health
├── ui/
│   └── streamlit_app.py
├── tests/
│   ├── conftest.py
│   ├── test_chunker.py
│   └── test_hybrid.py
├── data/
│   ├── sample/                  # 2-3 örnek Türkçe doküman (repo'ya dahil)
│   ├── uploads/                 # (gitignored) yüklenen kullanıcı dosyaları
│   └── chroma/                  # (gitignored) kalıcı vektör db
├── eval/
│   └── questions.json           # 10 soru + beklenen cevap
├── pyproject.toml
├── uv.lock
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

## Modül Detayları

### Ingestion

- `loaders.py`: `.pdf` (pypdf, sayfa bazlı metin çıkarma → her sayfa için `(text, page_number)`), `.txt` ve `.md` (tek parça metin, `page_number=None`). Tek bir `load_document(path) -> list[PageContent]` arayüzü.
- `chunker.py`: `chunk_text(pages: list[PageContent], chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]`.
  - Önce paragraflara (`\n\n`) böl.
  - Paragraf `chunk_size`'ı aşarsa cümle sınırından (Türkçe noktalama dahil regex: `. ! ? …` sonrası boşluk) böl.
  - Ardışık parçalar `chunk_size`'a ulaşana kadar birleştirilir; `overlap` kadar önceki chunk'ın sonundan metin eklenir (kelime ortasından kesmeden).
  - Her `Chunk`: `text`, `source_file`, `page_number`, `chunk_index`.

### Retrieval

- `embeddings.py`: `EmbeddingModel` sınıfı, `intfloat/multilingual-e5-small` yükler; e5 modelleri için gerekli `"query: "` / `"passage: "` prefix'lerini uygular. `encode(texts: list[str]) -> np.ndarray`.
- `vector_store.py`: `VectorStore` sınıfı, ChromaDB `PersistentClient(path="data/chroma")`. `add(chunks, embeddings)`, `query(query_embedding, top_k) -> list[(chunk_id, score)]`.
- `keyword_search.py`: `BM25Index` — ingest sırasında tüm chunk metinleri üzerinde in-memory `BM25Okapi` kurulur (basit Türkçe tokenization: küçük harfe çevirme + regex kelime ayırma; stemming yok — kapsam dışı). `query(text, top_k) -> list[(chunk_id, score)]`.
- `hybrid.py`: `reciprocal_rank_fusion(vector_results, keyword_results, k: int = 60) -> list[(chunk_id, fused_score)]` — `score = Σ 1/(rank_i + k)`, sıralı chunk_id listesi döner.

### Generation

- `prompts.py`: Sistem promptu — sadece verilen bağlamı kullan, her iddiaya `[n]` ile atıf yap, bağlamda yoksa "Dokümanlarda bu bilgi yok." de. Bağlam formatlama: `[1] ({source_file}, sayfa {page}): {text}`.
- `answerer.py`: `generate_answer(question, chunks) -> AnswerResult` — Anthropic `messages.create` çağrısı yapar, `answer` metni + `sources` (kullanılan chunk metadata) + `used_chunk_ids` döner.

### API (FastAPI)

- `POST /ingest`: multipart dosya listesi alır → her dosya için loaders → chunker → embed → vector_store.add + BM25 index güncelleme. Yanıt: `{ingested_files: [...], chunk_count: int}`.
- `POST /query`: `{question: str, top_k: int = 5}` → hybrid retrieval (vector top 10 + BM25 top 10 → RRF → ilk `top_k`) → `generate_answer`. Yanıt: `{answer: str, sources: [{source_file, page_number, text, score}], used_chunk_ids: [...]}`.
- `GET /health`: `{status: "ok", embedding_model: str, chunk_count: int}`.

### Streamlit UI

- Sol panel (`st.sidebar`): dosya yükleme (`st.file_uploader`, çoklu), "Yükle" butonu → `/ingest` çağrısı, sonuç mesajı.
- Ana alan: soru kutusu, "Sor" butonu → `/query` çağrısı → cevap metni; altında her kaynak için `st.expander("[n] {source_file} — sayfa {page}")` içinde chunk metni ve skor.

## Hata Yönetimi

- Desteklenmeyen dosya uzantısı → `/ingest` 400 döner, diğer dosyalar işlenmeye devam eder (kısmi başarı raporlanır).
- Boş/bulunamayan koleksiyon üzerinde `/query` → boş `sources` ve "Dokümanlarda bu bilgi yok." cevabı (LLM çağrısı yapılmadan, retrieval sonuç dönmezse kısa yol).
- Anthropic API hatası → 502 + hata mesajı, UI'da kullanıcıya gösterilir.

## Test Kapsamı

- `test_chunker.py`: paragraf/cümle sınırı saygısı, overlap doğruluğu, metadata (source_file, page_number, chunk_index) doğruluğu, kelime ortasından kesilmediğinin doğrulanması.
- `test_hybrid.py`: RRF fusion sıralama mantığı — sabit mock vektör/BM25 sonuçlarıyla beklenen sıralamanın doğrulanması (embedding modeli veya gerçek DB gerektirmez).
- Gerçek Claude API çağrısı testlerde mock'lanır; testler API key gerektirmeden çalışır.

## Docker

- `Dockerfile.api`: `uv sync` ile bağımlılıkları kurar, `uvicorn rag_tr.api.main:app` çalıştırır.
- `Dockerfile.ui`: aynı bağımlılıklar, `streamlit run ui/streamlit_app.py`.
- `docker-compose.yml`: `api` (port 8000) ve `ui` (port 8501, `API_URL=http://api:8000` ile bağlanır) servisleri, `./data` bind mount ile kalıcılık.

## Eval Seti

`eval/questions.json`: `data/sample/` altındaki örnek dokümanlara dayalı 10 soru:
```json
[{"question": "...", "expected_answer": "...", "source_file": "..."}]
```
İkinci projede (retrieval/generation kalitesi kıyaslaması) kullanılacağı için sorular ve beklenen cevaplar örnek dokümanların içeriğiyle net biçimde doğrulanabilir olmalı.

## README İçeriği

Problem tanımı, Mermaid mimari diyagramı, kurulum (uv + docker-compose), kullanım, **Tasarım Kararları** (neden hibrit arama, neden bu chunk boyutu/overlap, neden yerel embedding modeli, neden RRF), **Geliştirme Fikirleri** (reranker, streaming yanıt, çoklu dil desteği, semantic caching, vb.).

## Kapsam Dışı (Bilinçli Olarak)

- Kullanıcı kimlik doğrulama / çoklu kullanıcı izolasyonu.
- Streaming yanıt (geliştirme fikri olarak README'de belirtilecek, uygulanmayacak).
- Reranker modeli (geliştirme fikri).
- Dosya silme/güncelleme endpoint'i (yalnızca ekleme).
- BM25 index'in diske kalıcı serialize edilmesi (ingest sonrası bellekte kalır, API restart'ında ingestion tekrar tetiklenmeden BM25 boşalır — bu proje kapsamında kabul edilebilir bir sınırlama, README'de not edilecek).
