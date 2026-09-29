"""
问题重写模块（Query Rewriting）。

向量检索对"独立、完整"的查询最友好，但多轮对话里用户经常这样提问:

    用户: 我最近在学习 ChromaDB
    用户: 它和 Milvus 有什么区别？     ← "它"指 ChromaDB，直接拿去检索效果很差

问题重写就是检索前加的一步: 结合最近的对话历史，把用户问题改写成
一条"脱离上下文也能看懂"的独立查询:

    它和 Milvus 有什么区别？ → ChromaDB 和 Milvus 有什么区别？

实现方式: 调用一次 GLM（不带工具）。任何失败都退回原始查询，
绝不阻断后面的检索流程。
"""

import logging
from typing import Any

from app.config import RAG_REWRITE_ENABLED, RAG_REWRITE_HISTORY_MSGS
from app.llm import chat_without_tools

logger = logging.getLogger(__name__)

# 单条历史消息最多保留的字符数（控制重写请求的大小）
HISTORY_MSG_MAX_CHARS = 200

REWRITE_PROMPT = """你是搜索查询优化器。请把"用户最新问题"改写成一条适合在知识库中检索的独立查询。

要求:
1. 结合对话历史，把"它""这个""上面提到的"等指代替换成具体对象
2. 补全省略的主语和背景，让查询脱离上下文也能看懂
3. 保留用户问题的原意，不要扩大或歪曲问题
4. 如果问题本身已经完整清晰，原样输出即可
5. 只输出改写后的查询本身，不要解释、不要加引号

对话历史（可能为空）:
{history}

用户最新问题: {query}

改写后的查询:"""


def rewrite_query(query: str, history: list[dict[str, Any]] | None = None) -> str:
    """
    把用户问题重写为适合检索的独立查询。

    参数:
        query: 用户的原始问题
        history: 对话历史（可选），用于解决"它/这个"等指代

    返回:
        重写后的查询；任何失败都退回原始 query，保证检索不被阻断
    """
    if not query or not query.strip():
        return query

    if not RAG_REWRITE_ENABLED:
        return query

    try:
        prompt = REWRITE_PROMPT.format(
            history=_format_history(history) or "（无）",
            query=query,
        )
        response = chat_without_tools([{"role": "user", "content": prompt}])
        rewritten = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Query rewrite failed, using original query: %s", e)
        return query

    # 模型偶尔会把引号一起输出，这里做最基本的清理
    rewritten = rewritten.strip().strip("\"'“”‘’").strip()

    if not rewritten:
        logger.warning("Query rewrite returned empty, using original query")
        return query

    if rewritten != query:
        logger.info("Query rewritten: '%s' -> '%s'", query, rewritten)

    return rewritten


def _format_history(history: list[dict[str, Any]] | None) -> str:
    """
    把对话历史压成几行文本，供重写提示词使用。

    只取最近的 RAG_REWRITE_HISTORY_MSGS 条有内容的 user/assistant 消息，
    跳过 tool 消息和空消息，每条截断到 HISTORY_MSG_MAX_CHARS。
    """
    if not history:
        return ""

    picked: list[str] = []

    for msg in reversed(history):
        if len(picked) >= RAG_REWRITE_HISTORY_MSGS:
            break

        role = msg.get("role")
        content = (msg.get("content") or "").strip()

        if role not in ("user", "assistant") or not content:
            continue

        prefix = "用户" if role == "user" else "助手"
        picked.append(f"{prefix}: {content[:HISTORY_MSG_MAX_CHARS]}")

    # 从最新往前取的，再倒回来恢复时间顺序
    picked.reverse()
    return "\n".join(picked)
