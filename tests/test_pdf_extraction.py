"""PDF metin cikarma ve sayfa numarasi testleri.

Neden bu dosya var: PDF, README'de ilk sirada sayilan desteklenen tur ama test
takiminda yalnizca BOZUK bir PDF'in reddedildigi sinaniyordu -- gecerli bir
PDF'den metin cikarmanin, sayfa numarasinin dogru atanmasinin ve bos sayfalarin
atlanmasinin otomatik karsiligi yoktu. Bu yol canli dogrulamalarda elle
kontrol ediliyordu; burada kalici hale geliyor.

Hicbir API cagrisi yapilmaz ve depoya ikili (binary) fixture eklenmez: gecerli
PDF testin icinde, bayt bayt uretiliyor (`_build_pdf`). Boylece fixture'in neyi
icerdigi okunabilir kalıyor ve ek bir bagimlilik (reportlab vb.) gerekmiyor --
`pypdf` yalnizca okuma yapar, PDF uretemez.
"""

import pypdf
import pytest

from rag_tr.ingestion.chunker import chunk_text
from rag_tr.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document

#: Sayfa metinleri WinAnsi (cp1252) ile kodlanabilir Turkce harfler kullanir
#: (ü, ö, ç). Hicbir ek font gomulmedigi icin 'ş', 'ğ', 'ı' bu basit fixture'da
#: temsil edilemez; testin konusu karakter kapsamasi degil, cikarma yolunun
#: kendisi (metin + sayfa numarasi).
SAYFA_1 = "Türkiye'nin yüzölçümü 783.562 kilometrekaredir."
SAYFA_2 = "Van Gölü Türkiye'nin en büyük gölüdür."


def _escape(text: str) -> bytes:
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return escaped.encode("cp1252")


def _build_pdf(pages: list[str]) -> bytes:
    """Verilen her metin icin bir sayfa tasiyan, gecerli bir PDF uretir.

    Elle kurulan bir PDF: nesneler, byte offset'leriyle bir xref tablosu ve
    trailer. Amac minimal ama SARTNAMEYE UYGUN bir dosya -- pypdf'in bozuk bir
    xref'i onarma davranisina guvenmemek icin offset'ler gercekten hesaplanir.
    """
    objects: list[bytes] = [b"", b""]  # 1: catalog, 2: pages (sonra doldurulur)
    catalog_num, pages_num = 1, 2

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_num = add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>"
    )

    kids: list[int] = []
    for text in pages:
        stream = b"BT /F1 14 Tf 72 720 Td (" + _escape(text) + b") Tj ET\n"
        content_num = add(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
            + stream + b"endstream"
        )
        kids.append(
            add(
                b"<< /Type /Page /Parent " + str(pages_num).encode() + b" 0 R "
                b"/MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 " + str(font_num).encode() + b" 0 R >> >> "
                b"/Contents " + str(content_num).encode() + b" 0 R >>"
            )
        )

    objects[catalog_num - 1] = (
        b"<< /Type /Catalog /Pages " + str(pages_num).encode() + b" 0 R >>"
    )
    objects[pages_num - 1] = (
        b"<< /Type /Pages /Kids ["
        + b" ".join(str(k).encode() + b" 0 R" for k in kids)
        + b"] /Count " + str(len(kids)).encode() + b" >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode()
        + b" /Root " + str(catalog_num).encode() + b" 0 R >>\nstartxref\n"
        + str(xref_at).encode() + b"\n%%EOF\n"
    )
    return bytes(out)


@pytest.fixture
def iki_sayfali_pdf(tmp_path):
    path = tmp_path / "cografya.pdf"
    path.write_bytes(_build_pdf([SAYFA_1, SAYFA_2]))
    return path


# --- fixture'in kendisi gercekten gecerli mi -----------------------------------


def test_the_fixture_is_a_pdf_pypdf_can_open(iki_sayfali_pdf):
    """Fixture bozuksa asagidaki testler yanlis sebeple gecer/kalir."""
    reader = pypdf.PdfReader(str(iki_sayfali_pdf))

    assert len(reader.pages) == 2


# --- cikarma -------------------------------------------------------------------


def test_pdf_is_a_supported_extension():
    assert ".pdf" in SUPPORTED_EXTENSIONS


def test_text_is_extracted_from_every_page(iki_sayfali_pdf):
    pages = load_document(iki_sayfali_pdf)

    assert [page.text for page in pages] == [SAYFA_1, SAYFA_2]


def test_turkish_characters_survive_extraction(iki_sayfali_pdf):
    """Cikarilan metin Turkce harfleri bozmadan tasimali."""
    pages = load_document(iki_sayfali_pdf)

    assert "yüzölçümü" in pages[0].text
    assert "Gölü" in pages[1].text
    assert "?" not in pages[0].text, "karakterler mojibake'e dönüşmemeli"


def test_each_page_carries_its_one_based_page_number(iki_sayfali_pdf):
    """Atiflarda gosterilen sayfa numarasi buradan geliyor; 1'den baslamali."""
    pages = load_document(iki_sayfali_pdf)

    assert [page.page_number for page in pages] == [1, 2]


def test_a_single_page_pdf_yields_one_page(tmp_path):
    path = tmp_path / "tek.pdf"
    path.write_bytes(_build_pdf([SAYFA_1]))

    pages = load_document(path)

    assert len(pages) == 1
    assert pages[0].page_number == 1


def test_blank_pages_are_skipped_without_shifting_the_numbers(tmp_path):
    """Bos sayfa atlanir ama KALAN sayfalarin numarasi kaymaz.

    Numaralandirma dosyadaki gercek sayfa sirasina gore yapilmali: aksi halde
    bos bir on sayfa, atiflarin yanlis sayfaya isaret etmesine yol acardi."""
    path = tmp_path / "bosluklu.pdf"
    path.write_bytes(_build_pdf(["", SAYFA_1, "", SAYFA_2]))

    pages = load_document(path)

    assert [page.text for page in pages] == [SAYFA_1, SAYFA_2]
    assert [page.page_number for page in pages] == [2, 4], (
        "atıflar dosyadaki gerçek sayfa numarasını göstermeli"
    )


# --- metin dosyalariyla farki --------------------------------------------------


def test_text_files_have_no_page_number(tmp_path):
    """TXT/MD sayfasiz: atifta sayfa alani bos kalir."""
    path = tmp_path / "notlar.md"
    path.write_text(SAYFA_1, encoding="utf-8")

    pages = load_document(path)

    assert len(pages) == 1
    assert pages[0].page_number is None


# --- cikarma -> chunk'lama zinciri ---------------------------------------------


def test_extracted_pdf_text_flows_into_chunking(iki_sayfali_pdf):
    """Cikarilan metin chunk'layiciya verilebilir ve icerigi korunur.

    Ingestion zincirinin PDF ucu: load_document -> chunk_text. Embedding ve
    vektor deposu bu testin kapsaminda degil (API cagrisi yapilmaz)."""
    pages = load_document(iki_sayfali_pdf)

    chunks = chunk_text(pages, source_file="cografya.pdf")

    assert chunks, "chunk üretilmeli"
    assert any("yüzölçümü" in chunk.text for chunk in chunks)
    assert any("Gölü" in chunk.text for chunk in chunks)
    assert all(chunk.source_file == "cografya.pdf" for chunk in chunks)


def test_the_page_number_reaches_the_chunk_that_will_be_cited(iki_sayfali_pdf):
    """Atif zincirinin tamami: PDF sayfasi -> chunk.page_number.

    Kullaniciya gosterilen "hangi sayfadan geldi" bilgisi bu alandan geliyor;
    sayfa numarasi chunk'lama sirasinda dusse atif sessizce bozulurdu."""
    chunks = chunk_text(load_document(iki_sayfali_pdf), source_file="cografya.pdf")

    sayfalar = {
        chunk.page_number for chunk in chunks if "yüzölçümü" in chunk.text
    }
    assert sayfalar == {1}

    sayfalar = {chunk.page_number for chunk in chunks if "Gölü" in chunk.text}
    assert sayfalar == {2}


def test_chunk_ids_from_a_pdf_are_deterministic(iki_sayfali_pdf):
    """`source_file::chunk_index` sozlesmesi PDF icin de gecerli."""
    first = chunk_text(load_document(iki_sayfali_pdf), source_file="cografya.pdf")
    second = chunk_text(load_document(iki_sayfali_pdf), source_file="cografya.pdf")

    assert [c.chunk_index for c in first] == list(range(len(first)))
    assert [(c.chunk_index, c.text) for c in first] == [
        (c.chunk_index, c.text) for c in second
    ]


def test_a_corrupt_pdf_is_still_rejected(tmp_path):
    """Mevcut davranis korunuyor: PDF olmayan bir .pdf acilamaz."""
    path = tmp_path / "bozuk.pdf"
    path.write_text("bu bir PDF degil", encoding="utf-8")

    with pytest.raises(Exception):
        load_document(path)
