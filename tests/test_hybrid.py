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
