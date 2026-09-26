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
