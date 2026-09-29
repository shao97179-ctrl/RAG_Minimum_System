"""
检索流水线单元测试。

重写 / 向量检索 / 重排各环节全部用假对象替换，不发真实网络请求。
"""

from app.rag import retriever


class TestRetrieve:
    """retrieve 流水线测试。"""

    def test_pipeline_order(self, monkeypatch) -> None:
        """重写 → 粗检索 → 重排依次执行，结果来自重排后的 top_k。"""
        calls: list[tuple] = []

        def fake_rewrite(query: str, history=None) -> str:
            calls.append(("rewrite", query))
            return "改写后的查询"

        def fake_search(query: str, top_k: int) -> list[dict]:
            calls.append(("search", query, top_k))
            return [
                {"text": f"候选{i}", "filename": "a.txt", "distance": 0.1 * i}
                for i in range(top_k)
            ]

        def fake_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
            calls.append(("rerank", query, len(candidates), top_k))
            # 简单地把候选倒序当作重排结果
            return list(reversed(candidates))[:top_k]

        monkeypatch.setattr(retriever, "rewrite_query", fake_rewrite)
        monkeypatch.setattr(retriever, "vector_search", fake_search)
        monkeypatch.setattr(retriever, "rerank_documents", fake_rerank)

        results = retriever.retrieve(
            "原始问题", history=[{"role": "user", "content": "hi"}]
        )

        # 流水线顺序正确，且粗检索用了改写后的查询和候选数量配置
        assert calls[0] == ("rewrite", "原始问题")
        assert calls[1] == ("search", "改写后的查询", retriever.RAG_CANDIDATE_K)
        assert calls[2] == (
            "rerank", "改写后的查询", retriever.RAG_CANDIDATE_K, retriever.RAG_TOP_K,
        )
        assert results[0]["text"] == f"候选{retriever.RAG_CANDIDATE_K - 1}"

    def test_no_candidates_returns_empty(self, monkeypatch) -> None:
        """粗检索没有结果时直接返回空列表。"""
        monkeypatch.setattr(
            retriever, "rewrite_query", lambda query, history=None: query
        )
        monkeypatch.setattr(retriever, "vector_search", lambda query, top_k: [])

        assert retriever.retrieve("问题") == []

    def test_single_candidate_skips_rerank(self, monkeypatch) -> None:
        """只有一个候选时没有可比较对象，跳过重排。"""
        monkeypatch.setattr(
            retriever, "rewrite_query", lambda query, history=None: query
        )
        monkeypatch.setattr(
            retriever,
            "vector_search",
            lambda query, top_k: [{"text": "唯一", "filename": "a.txt", "distance": 0.1}],
        )

        called: list[tuple] = []

        def fake_rerank(*args, **kwargs) -> list[dict]:
            called.append(args)
            return []

        monkeypatch.setattr(retriever, "rerank_documents", fake_rerank)

        results = retriever.retrieve("问题")

        assert results[0]["text"] == "唯一"
        assert called == []

    def test_rewrite_disabled(self, monkeypatch) -> None:
        """重写开关关闭时，直接用原始问题检索。"""
        calls: list[tuple] = []

        def fake_rewrite(query: str, history=None) -> str:
            calls.append(("rewrite", query))
            return "不应该被调用"

        monkeypatch.setattr(retriever, "rewrite_query", fake_rewrite)
        monkeypatch.setattr(retriever, "RAG_REWRITE_ENABLED", False)
        monkeypatch.setattr(
            retriever,
            "vector_search",
            lambda query, top_k: calls.append(("search", query)) or [],
        )

        retriever.retrieve("原始问题")

        assert calls == [("search", "原始问题")]

    def test_rerank_disabled(self, monkeypatch) -> None:
        """重排开关关闭时，按向量检索顺序截取 top_k。"""
        monkeypatch.setattr(
            retriever, "rewrite_query", lambda query, history=None: query
        )
        monkeypatch.setattr(retriever, "RAG_RERANK_ENABLED", False)

        def fake_search(query: str, top_k: int) -> list[dict]:
            return [
                {"text": f"候选{i}", "filename": "a.txt", "distance": 0.1}
                for i in range(top_k)
            ]

        monkeypatch.setattr(retriever, "vector_search", fake_search)

        called: list[tuple] = []

        def fake_rerank(*args, **kwargs) -> list[dict]:
            called.append(args)
            return []

        monkeypatch.setattr(retriever, "rerank_documents", fake_rerank)

        results = retriever.retrieve("问题")

        assert called == []
        assert [r["text"] for r in results] == [
            f"候选{i}" for i in range(retriever.RAG_TOP_K)
        ]
