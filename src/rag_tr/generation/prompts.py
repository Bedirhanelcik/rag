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
