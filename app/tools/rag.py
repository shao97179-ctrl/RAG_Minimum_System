"""
RAG 工具模块。

对 Agent 暴露的接口只有:
    search_knowledge_base(query: str, history: list | None = None) -> str

内部调用 app/rag/retriever.py 的完整检索流水线
（问题重写 → 向量粗检索 → 重排），细节全部封装。

history 参数不出现在工具 Schema 里（LLM 看不到它），
由 Tool Dispatcher 按函数签名自动注入，仅供问题重写
解决"它/这个"等指代使用。
"""

import logging
from typing import Any

from app.rag.retriever import retrieve
from app.rag.vector_store import is_index_built

logger = logging.getLogger(__name__)


async def search_knowledge_base(
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """
    在知识库中检索与 query 相关的内容。

    参数:
        query: 查询文本
        history: 对话历史（由 Tool Dispatcher 注入，不传也可用）

    返回:
        检索结果字符串，供 LLM 使用
    """
    if not query or not query.strip():
        return "错误：查询内容不能为空。"

    logger.info("RAG search: %s", query)

    # 检查索引是否已构建
    if not await is_index_built():
        logger.warning("Knowledge base index not built")
        return "知识库尚未初始化，请先运行知识库索引构建（python main.py --init-rag）。"

    # 完整检索流水线: 问题重写 → 向量粗检索 → 重排
    try:
        results = await retrieve(query, history=history)
    except Exception as e:
        logger.error("RAG search failed: %s", e)
        return f"知识库检索失败: {e}"

    if not results:
        logger.info("No relevant documents found for: %s", query)
        return "在知识库中未找到与该问题相关的内容。"

    # 格式化检索结果
    formatted_parts: list[str] = []

    for i, item in enumerate(results, 1):
        formatted_parts.append(
            f"【相关资料 {i}】(来源: {item['filename']})\n{item['text']}"
        )

    result_text = "\n\n".join(formatted_parts)
    logger.info("Retrieved %d documents for query: %s", len(results), query)

    return result_text
