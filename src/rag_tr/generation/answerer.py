from dataclasses import dataclass, field

from rag_tr.contracts import NO_CONTEXT_MESSAGE, QueryStatus
from rag_tr.generation.prompts import SYSTEM_PROMPT, format_context
from rag_tr.ingestion.chunker import Chunk
from rag_tr.retrieval.vector_store import make_chunk_id


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict]
    used_chunk_ids: list[str]
    status: QueryStatus = field(default=QueryStatus.ANSWERED)


def no_context_result() -> AnswerResult:
    return AnswerResult(
        answer=NO_CONTEXT_MESSAGE,
        sources=[],
        used_chunk_ids=[],
        status=QueryStatus.NO_RELEVANT_CONTEXT,
    )


def generate_answer(question: str, chunks: list[Chunk], client, model: str) -> AnswerResult:
    if not chunks:
        return no_context_result()

    context = format_context(chunks)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {question}"}],
    )
    answer_text = message.content[0].text

    # Chunk'lar bulundu ama Claude bunlarin soruyu cevaplamadigina karar verdi.
    # Agent'in bu durumu bir cevaptan ayirt etmesi icin serbest metne bakmasi
    # gerekmemeli, bu yuzden status alanina tasiyoruz.
    if answer_text.strip() == NO_CONTEXT_MESSAGE:
        return AnswerResult(
            answer=answer_text.strip(),
            sources=[],
            used_chunk_ids=[],
            status=QueryStatus.NO_RELEVANT_CONTEXT,
        )

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
    return AnswerResult(
        answer=answer_text,
        sources=sources,
        used_chunk_ids=used_chunk_ids,
        status=QueryStatus.ANSWERED,
    )
