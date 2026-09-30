"""
RAG 工具和文本切分单元测试。
"""

from app.rag.chunker import split_text, chunk_documents
from app.tools.rag import search_knowledge_base


class TestChunker:
    """文本切分测试。"""

    def test_short_text(self) -> None:
        """短文本不切分。"""
        result = split_text("这是一段短文本", chunk_size=300)
        assert len(result) == 1
        assert result[0] == "这是一段短文本"

    def test_long_text(self) -> None:
        """长文本被切分为多个 chunk。"""
        text = "这是一段很长的文本。" * 100
        result = split_text(text, chunk_size=300, chunk_overlap=50)
        assert len(result) > 1

    def test_empty_text(self) -> None:
        """空文本返回空列表。"""
        result = split_text("")
        assert result == []

    def test_chunk_documents(self) -> None:
        """文档切分测试。"""
        documents = [
            {"filename": "test.txt", "content": "测试内容"},
        ]
        chunks = chunk_documents(documents)
        assert len(chunks) >= 1
        assert chunks[0]["filename"] == "test.txt"


class TestSearchKnowledgeBase:
    """RAG 搜索测试。"""

    async def test_empty_query(self) -> None:
        """空查询返回错误。"""
        result = await search_knowledge_base("")
        assert "错误" in result

    async def test_not_initialized(self) -> None:
        """知识库未初始化时给出提示。"""
        # 如果索引已构建，这个测试会跳过
        from app.rag.vector_store import is_index_built
        if not await is_index_built():
            result = await search_knowledge_base("测试查询")
            assert "初始化" in result or "错误" in result

    async def test_history_passed_to_pipeline(self, monkeypatch) -> None:
        """对话历史会透传给检索流水线（供问题重写解决指代）。"""
        from app.tools import rag as rag_tool

        captured: dict[str, object] = {}

        async def fake_retrieve(query, history=None):
            captured["query"] = query
            captured["history"] = history
            return [
                {"text": "ChromaDB 是一个向量数据库", "filename": "a.txt", "distance": 0.1}
            ]

        async def fake_is_built() -> bool:
            return True

        monkeypatch.setattr(rag_tool, "retrieve", fake_retrieve)
        monkeypatch.setattr(rag_tool, "is_index_built", fake_is_built)

        history = [{"role": "user", "content": "我在学 ChromaDB"}]
        result = await rag_tool.search_knowledge_base("它是什么？", history=history)

        assert captured["history"] == history
        assert "ChromaDB 是一个向量数据库" in result
