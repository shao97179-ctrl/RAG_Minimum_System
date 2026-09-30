"""
检索流水线模块（Retriever）。

把 RAG 检索的完整链路串起来:

    用户问题（可能带指代、省略）
        ↓
    ① 问题重写（query_rewriter）→ 独立、明确的检索查询
        ↓
    ② 向量粗检索（vector_store）→ 多召回 RAG_CANDIDATE_K 个候选片段
        ↓
    ③ 重排（reranker）→ LLM 按相关性精排，取 RAG_TOP_K 个
        ↓
    最终检索结果

app/tools/rag.py 只需要调用这里的 retrieve()。
"""

import logging
from typing import Any

from app.config import (
    RAG_CANDIDATE_K,
    RAG_RERANK_ENABLED,
    RAG_REWRITE_ENABLED,
    RAG_TOP_K,
)
from app.rag.query_rewriter import rewrite_query
from app.rag.reranker import rerank_documents
from app.rag.vector_store import search as vector_search

logger = logging.getLogger(__name__)


async def retrieve(query: str, history: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """
    执行完整的检索流水线: 问题重写 → 向量粗检索 → 重排。

    三步之间有数据依赖（后一步用前一步的结果），因此顺序 await；
    一次检索会发起多次 LLM/Embedding 网络请求，全部为异步调用。

    参数:
        query: 用户的原始问题
        history: 对话历史（可选），供问题重写解决"它/这个"等指代

    返回:
        排好序的检索结果，每项
        {"text", "filename", "distance"(, "rerank_score")}
    """
    # ① 问题重写: 失败时 rewrite_query 内部会退回原始查询
    search_query = query
    if RAG_REWRITE_ENABLED:
        search_query = await rewrite_query(query, history)

    # ② 向量粗检索: 召回比 top_k 更多的候选，给重排留出挑选空间
    candidate_k = max(RAG_CANDIDATE_K, RAG_TOP_K)
    candidates = await vector_search(search_query, top_k=candidate_k)

    if not candidates:
        logger.info("No candidates retrieved for: %s", search_query)
        return []

    # ③ 重排: LLM 给候选按相关性打分排序，取 top_k；
    #    只有一个候选时没有可比较的对象，跳过重排
    if RAG_RERANK_ENABLED and len(candidates) > 1:
        results = await rerank_documents(search_query, candidates, top_k=RAG_TOP_K)
    else:
        results = candidates[:RAG_TOP_K]

    logger.info(
        "Retrieve done: query '%s' -> %d candidates -> %d results",
        search_query, len(candidates), len(results),
    )
    return results
