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
