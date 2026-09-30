"""
重排模块单元测试。

所有 LLM 调用都用假对象替换，不发真实网络请求。
"""

import json
from types import SimpleNamespace

from app.rag import reranker

CANDIDATES = [
    {"text": "ChromaDB 是一个轻量级向量数据库", "filename": "a.txt", "distance": 0.2},
    {"text": "Milvus 是分布式向量数据库", "filename": "a.txt", "distance": 0.3},
    {"text": "今天天气不错", "filename": "b.txt", "distance": 0.4},
]


def _llm_output(scores: dict[int, float]) -> str:
    """把 {候选下标: 分数} 变成模型会输出的 JSON 文本（id 从 1 开始）。"""
    items = [{"id": i + 1, "score": s} for i, s in scores.items()]
    return json.dumps({"scores": items}, ensure_ascii=False)


def _fake_llm(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class TestRerankDocuments:
    """rerank_documents 测试。"""

    async def test_sorted_by_llm_score(self, monkeypatch) -> None:
        """按 LLM 分数从高到低排序。"""

        async def fake_chat(messages: list[dict[str, str]]):
            return _fake_llm(_llm_output({0: 3.0, 1: 9.0, 2: 1.0}))

        monkeypatch.setattr(reranker, "chat_without_tools", fake_chat)

        results = await reranker.rerank_documents("向量数据库", CANDIDATES, top_k=2)

        assert len(results) == 2
        assert results[0]["text"] == "Milvus 是分布式向量数据库"
        assert results[0]["rerank_score"] == 9.0
        assert results[1]["rerank_score"] == 3.0

    async def test_llm_failure_keeps_vector_order(self, monkeypatch) -> None:
        """LLM 调用失败时退回向量检索的原始顺序。"""

        async def boom(messages: list[dict[str, str]]):
            raise RuntimeError("API down")

        monkeypatch.setattr(reranker, "chat_without_tools", boom)

        results = await reranker.rerank_documents("查询", CANDIDATES, top_k=2)

        assert [r["text"] for r in results] == [c["text"] for c in CANDIDATES[:2]]
        assert "rerank_score" not in results[0]

    async def test_unparseable_response_keeps_vector_order(self, monkeypatch) -> None:
        """模型输出不是合法 JSON 时退回原始顺序。"""

        async def fake_chat(messages: list[dict[str, str]]):
            return _fake_llm("我觉得都还行")

        monkeypatch.setattr(reranker, "chat_without_tools", fake_chat)

        results = await reranker.rerank_documents("查询", CANDIDATES, top_k=3)

        assert [r["text"] for r in results] == [c["text"] for c in CANDIDATES]

    async def test_json_with_code_fence(self, monkeypatch) -> None:
        """模型输出带 ```json 围栏也能解析。"""
        text = "```json\n" + _llm_output({0: 1.0, 1: 5.0, 2: 8.0}) + "\n```"

        async def fake_chat(messages: list[dict[str, str]]):
            return _fake_llm(text)

        monkeypatch.setattr(reranker, "chat_without_tools", fake_chat)

        results = await reranker.rerank_documents("查询", CANDIDATES, top_k=1)

        assert results[0]["text"] == "今天天气不错"
        assert results[0]["rerank_score"] == 8.0

    async def test_missing_ids_score_zero(self, monkeypatch) -> None:
        """模型漏掉的候选按 0 分处理，排到最后。"""

        async def fake_chat(messages: list[dict[str, str]]):
            return _fake_llm(_llm_output({1: 5.0}))

        monkeypatch.setattr(reranker, "chat_without_tools", fake_chat)

        results = await reranker.rerank_documents("查询", CANDIDATES, top_k=3)

        assert results[0]["text"] == "Milvus 是分布式向量数据库"
        assert results[1]["text"] == "ChromaDB 是一个轻量级向量数据库"
        assert results[2]["text"] == "今天天气不错"

    async def test_score_clamped_to_range(self, monkeypatch) -> None:
        """模型输出超范围分数会被截断到 0~10。"""

        async def fake_chat(messages: list[dict[str, str]]):
            return _fake_llm(_llm_output({0: 99.0, 1: -5.0, 2: 5.0}))

        monkeypatch.setattr(reranker, "chat_without_tools", fake_chat)

        results = await reranker.rerank_documents("查询", CANDIDATES, top_k=3)

        assert results[0]["rerank_score"] == 10.0
        assert results[2]["rerank_score"] == 0.0

    async def test_empty_candidates(self) -> None:
        """空候选直接返回空列表。"""
        assert await reranker.rerank_documents("查询", [], top_k=3) == []

    async def test_does_not_mutate_input(self, monkeypatch) -> None:
        """重排不改动调用方传入的候选列表。"""

        async def fake_chat(messages: list[dict[str, str]]):
            return _fake_llm(_llm_output({0: 1.0, 1: 9.0, 2: 5.0}))

        monkeypatch.setattr(reranker, "chat_without_tools", fake_chat)

        await reranker.rerank_documents("查询", CANDIDATES, top_k=3)

        assert CANDIDATES[0]["text"] == "ChromaDB 是一个轻量级向量数据库"
        assert "rerank_score" not in CANDIDATES[0]
