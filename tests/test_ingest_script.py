"""`scripts/ingest.py` davranis testleri.

Script, calisan servise HTTP uzerinden dosya gonderir. Burada gercek Gemini
cagrilmaz: yerine stdlib ile kurulan kucuk bir HTTP sunucusu `/ingest` ve
`/health` uclarini taklit eder. Boylece `requests` yolu, multipart gonderimi,
Authorization basligi, cikis kodlari ve ekrana basilan metin gercekten
calistirilarak dogrulanir -- hicbir kota harcanmadan.
"""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ingest.py"

TOKEN = "test-token-asla-basilmamali"


def _load_script():
    """Script bir paket icinde olmadigi icin dosya yolundan yuklenir."""
    spec = importlib.util.spec_from_file_location("ingest_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return _load_script()


class _FakeServer:
    """`/ingest` ve `/health` taklidi. Gelen istekleri kaydeder."""

    def __init__(self, status: int = 200, body: dict | None = None, chunk_count: int = 7):
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # sunucu logu testi kirletmesin
                pass

            def _respond(self, code: int, payload: dict) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                raw = self.rfile.read(length)
                outer.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": raw,
                    }
                )
                self._respond(
                    status,
                    body
                    if body is not None
                    else {
                        "ingested_files": ["a.md", "b.md"],
                        "failed_files": [],
                        "chunk_count": chunk_count,
                    },
                )

            def do_GET(self):
                self._respond(
                    200,
                    {
                        "status": "ok",
                        "chunk_count": chunk_count,
                        "keyword_index_size": chunk_count,
                    },
                )

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def docs(tmp_path):
    first = tmp_path / "a.md"
    first.write_text("Ankara baskenttir ve nufusu kalabaliktir.", encoding="utf-8")
    second = tmp_path / "b.md"
    second.write_text("Istanbul en kalabalik sehirdir.", encoding="utf-8")
    return [first, second]


# --- yapilandirma dogrulamasi ---


def test_missing_api_url_stops_before_any_request(script, monkeypatch, docs):
    monkeypatch.delenv("RAG_API_URL", raising=False)
    monkeypatch.setenv("INGEST_API_TOKEN", TOKEN)
    monkeypatch.setattr("sys.argv", ["ingest.py", str(docs[0])])

    with pytest.raises(SystemExit) as excinfo:
        script.main()

    assert "RAG_API_URL" in str(excinfo.value)


def test_missing_token_stops_before_any_request(script, monkeypatch, docs):
    monkeypatch.setenv("RAG_API_URL", "http://127.0.0.1:1")
    monkeypatch.delenv("INGEST_API_TOKEN", raising=False)
    monkeypatch.setattr("sys.argv", ["ingest.py", str(docs[0])])

    with pytest.raises(SystemExit) as excinfo:
        script.main()

    assert "INGEST_API_TOKEN" in str(excinfo.value)


def test_unsupported_extension_is_rejected_client_side(script, tmp_path):
    bad = tmp_path / "tablo.xlsx"
    bad.write_text("veri", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        script._collect([str(bad)])

    assert "Desteklenmeyen" in str(excinfo.value)


def test_a_missing_file_is_reported_by_name(script, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        script._collect([str(tmp_path / "yok.md")])

    assert "yok.md" in str(excinfo.value)


# --- basarili gonderim ---


def test_successful_ingest_sends_the_files_with_a_bearer_token(script, monkeypatch, docs, capsys):
    with _FakeServer() as server:
        monkeypatch.setenv("RAG_API_URL", server.url + "/")  # sondaki / temizlenmeli
        monkeypatch.setenv("INGEST_API_TOKEN", TOKEN)
        monkeypatch.setattr("sys.argv", ["ingest.py", *(str(p) for p in docs)])

        assert script.main() == 0

        sent = server.requests
        assert len(sent) == 1
        assert sent[0]["path"] == "/ingest"
        assert sent[0]["authorization"] == f"Bearer {TOKEN}"
        # Her iki dosya da multipart govdede olmali.
        assert b"a.md" in sent[0]["body"] and b"b.md" in sent[0]["body"]
        assert b"baskenttir" in sent[0]["body"]

    output = capsys.readouterr().out
    assert "chunk    : 7" in output
    assert "korpus   : 7 chunk" in output


def test_the_token_is_never_printed(script, monkeypatch, docs, capsys):
    with _FakeServer() as server:
        monkeypatch.setenv("RAG_API_URL", server.url)
        monkeypatch.setenv("INGEST_API_TOKEN", TOKEN)
        monkeypatch.setattr("sys.argv", ["ingest.py", str(docs[0])])
        script.main()

    captured = capsys.readouterr()
    assert TOKEN not in captured.out
    assert TOKEN not in captured.err


# --- basarisiz gonderim ---


def test_an_unauthorized_response_is_reported_with_its_error_code(
    script, monkeypatch, docs, capsys
):
    body = {"detail": {"code": "unauthorized", "message": "Gecersiz ingest token."}}
    with _FakeServer(status=401, body=body) as server:
        monkeypatch.setenv("RAG_API_URL", server.url)
        monkeypatch.setenv("INGEST_API_TOKEN", TOKEN)
        monkeypatch.setattr("sys.argv", ["ingest.py", str(docs[0])])

        assert script.main() == 1

    output = capsys.readouterr().out
    assert "HTTP 401" in output
    assert "kod=unauthorized" in output
    assert "Gecersiz ingest token." in output
    assert TOKEN not in output


def test_an_empty_corpus_after_ingest_is_treated_as_failure(script, monkeypatch, docs):
    """Hicbir chunk olusmadiysa islem basarili sayilmamali."""
    body = {"ingested_files": [], "failed_files": ["a.md"], "chunk_count": 0}
    with _FakeServer(status=200, body=body, chunk_count=0) as server:
        monkeypatch.setenv("RAG_API_URL", server.url)
        monkeypatch.setenv("INGEST_API_TOKEN", TOKEN)
        monkeypatch.setattr("sys.argv", ["ingest.py", str(docs[0])])

        assert script.main() == 1


def test_no_gemini_client_is_constructed_by_the_script(script, monkeypatch, docs):
    """Script yalnizca HTTP istemcisidir: kendi basina embedding uretmez."""
    import google.genai

    def _explode(*args, **kwargs):
        raise AssertionError("ingest script'i LLM istemcisi kurmamali")

    monkeypatch.setattr(google.genai, "Client", _explode)

    with _FakeServer() as server:
        monkeypatch.setenv("RAG_API_URL", server.url)
        monkeypatch.setenv("INGEST_API_TOKEN", TOKEN)
        monkeypatch.setattr("sys.argv", ["ingest.py", str(docs[0])])

        assert script.main() == 0
