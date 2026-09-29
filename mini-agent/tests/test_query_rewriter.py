"""
问题重写模块单元测试。

所有 LLM 调用都用假对象替换，不发真实网络请求。
"""

from types import SimpleNamespace

from app.rag import query_rewriter


def _fake_llm_response(text: str):
    """构造一个形似智谱响应的对象。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class TestRewriteQuery:
    """rewrite_query 测试。"""

    def test_rewrites_with_history(self, monkeypatch) -> None:
        """结合历史把指代替换成具体对象。"""
        captured: dict[str, str] = {}

        def fake_chat(messages: list[dict[str, str]]):
            captured["prompt"] = messages[0]["content"]
            return _fake_llm_response("ChromaDB 和 Milvus 有什么区别？")

        monkeypatch.setattr(query_rewriter, "chat_without_tools", fake_chat)

        history = [
            {"role": "user", "content": "我最近在学习 ChromaDB"},
            {"role": "assistant", "content": "ChromaDB 是一个向量数据库。"},
        ]
        result = query_rewriter.rewrite_query("它和 Milvus 有什么区别？", history=history)

        assert result == "ChromaDB 和 Milvus 有什么区别？"
        # 历史和原始问题都要出现在提示词里
        assert "我最近在学习 ChromaDB" in captured["prompt"]
        assert "它和 Milvus 有什么区别？" in captured["prompt"]

    def test_llm_failure_returns_original(self, monkeypatch) -> None:
        """LLM 调用失败时退回原始查询，不抛异常。"""

        def boom(messages: list[dict[str, str]]):
            raise RuntimeError("API down")

        monkeypatch.setattr(query_rewriter, "chat_without_tools", boom)

        assert query_rewriter.rewrite_query("什么是 RAG？") == "什么是 RAG？"

    def test_empty_response_returns_original(self, monkeypatch) -> None:
        """模型返回空内容时退回原始查询。"""
        monkeypatch.setattr(
            query_rewriter,
            "chat_without_tools",
            lambda messages: _fake_llm_response("   "),
        )
        assert query_rewriter.rewrite_query("什么是 RAG？") == "什么是 RAG？"

    def test_disabled_returns_original(self, monkeypatch) -> None:
        """开关关闭时不调用 LLM，原样返回。"""
        called: list[list[dict[str, str]]] = []

        def fake_chat(messages: list[dict[str, str]]):
            called.append(messages)
            return _fake_llm_response("x")

        monkeypatch.setattr(query_rewriter, "chat_without_tools", fake_chat)
        monkeypatch.setattr(query_rewriter, "RAG_REWRITE_ENABLED", False)

        assert query_rewriter.rewrite_query("什么是 RAG？") == "什么是 RAG？"
        assert called == []

    def test_empty_query_no_llm_call(self, monkeypatch) -> None:
        """空查询原样返回，不调用 LLM。"""
        called: list[list[dict[str, str]]] = []

        def fake_chat(messages: list[dict[str, str]]):
            called.append(messages)
            return _fake_llm_response("x")

        monkeypatch.setattr(query_rewriter, "chat_without_tools", fake_chat)

        assert query_rewriter.rewrite_query("   ") == "   "
        assert called == []

    def test_history_skips_tool_and_empty_messages(self, monkeypatch) -> None:
        """构造历史文本时跳过 tool 消息和空消息。"""
        captured: dict[str, str] = {}

        def fake_chat(messages: list[dict[str, str]]):
            captured["prompt"] = messages[0]["content"]
            return _fake_llm_response("查询")

        monkeypatch.setattr(query_rewriter, "chat_without_tools", fake_chat)

        history = [
            {"role": "user", "content": "问题一"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "type": "function",
                             "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "工具结果"},
            {"role": "assistant", "content": "回答一"},
        ]
        query_rewriter.rewrite_query("接着问", history=history)

        assert "问题一" in captured["prompt"]
        assert "回答一" in captured["prompt"]
        assert "工具结果" not in captured["prompt"]

    def test_history_truncated_to_configured_count(self, monkeypatch) -> None:
        """只参考最近 RAG_REWRITE_HISTORY_MSGS 条消息。"""
        captured: dict[str, str] = {}

        def fake_chat(messages: list[dict[str, str]]):
            captured["prompt"] = messages[0]["content"]
            return _fake_llm_response("查询")

        monkeypatch.setattr(query_rewriter, "chat_without_tools", fake_chat)

        history = [
            {"role": "user", "content": f"旧消息{i}"} for i in range(10)
        ]
        query_rewriter.rewrite_query("当前问题", history=history)

        # 最早的几条不应出现，最近 RAG_REWRITE_HISTORY_MSGS 条应出现
        assert "旧消息0" not in captured["prompt"]
        latest = query_rewriter.RAG_REWRITE_HISTORY_MSGS
        assert f"旧消息{10 - latest}" in captured["prompt"]
