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
