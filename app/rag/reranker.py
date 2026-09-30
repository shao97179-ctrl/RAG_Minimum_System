"""
重排模块（Rerank）。

向量检索按"语义相似度"排序，但相似 ≠ 相关: 与问题"聊到同一话题"
的片段，不一定"回答了问题"。重排是检索之后的精排一步:

    向量粗检索（多召回几个候选）
        ↓
    LLM 给每个候选片段打 0~10 的相关性分数
        ↓
    按分数重新排序，只保留 top_k

实现方式: 一次 GLM 调用给所有候选打分（要求输出 JSON）。
调用失败或解析失败时退回向量检索的原始排序，绝不阻断检索。
"""

import json
import logging
from typing import Any

from app.config import RAG_TOP_K
from app.llm import chat_without_tools

logger = logging.getLogger(__name__)

# 送入重排提示词时，每个候选片段最多保留的字符数
CANDIDATE_MAX_CHARS = 300

RERANK_PROMPT = """你是搜索结果相关性评估器。给定一个查询和若干候选文档片段，请为每个片段打一个 0~10 的相关性分数:

- 9~10: 直接回答了查询
- 6~8: 高度相关，包含回答查询需要的关键信息
- 3~5: 部分相关，只提到了查询中的某些概念
- 0~2: 无关

只输出 JSON，不要解释，格式如下（id 对应候选片段的编号，从 1 开始；score 是 0~10 的数）:
{{"scores": [{{"id": 1, "score": 8}}, {{"id": 2, "score": 3}}]}}

查询: {query}

候选片段:
{candidates}

相关性分数 JSON:"""


async def rerank_documents(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = RAG_TOP_K,
) -> list[dict[str, Any]]:
    """
    用 LLM 给候选片段按相关性重新打分排序，返回前 top_k 个。

    参数:
        query: 查询文本
        candidates: 向量检索的候选列表
            [{"text": ..., "filename": ..., "distance": ...}]
        top_k: 重排后保留的数量

    返回:
        排序后的结果列表，每项额外带 rerank_score 字段；
        LLM 打分失败时退回候选的原始顺序（向量相似度序），不带 rerank_score
    """
    if not candidates:
        return []

    scores = await _llm_rerank_scores(query, candidates)

    if scores is None:
        logger.info("Rerank unavailable, keeping vector search order")
        return candidates[:top_k]

    # 按分数从高到低排序；同分时保持向量检索的原始顺序（稳定排序）
    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: scores[pair[0]],
        reverse=True,
    )

    results: list[dict[str, Any]] = []
    for idx, item in ranked[:top_k]:
        result = dict(item)  # 复制一份，不改动调用方的候选列表
        result["rerank_score"] = scores[idx]
        results.append(result)

    logger.info(
        "Reranked %d candidates, top score %.1f, kept %d",
        len(candidates), max(scores), len(results),
    )
    return results


async def _llm_rerank_scores(
    query: str,
    candidates: list[dict[str, Any]],
) -> list[float] | None:
    """
    调用 LLM 给每个候选打相关性分数。

    返回与 candidates 等长的分数列表（第 i 个对应第 i 个候选）；
    调用失败或解析失败时返回 None。
    """
    prompt = RERANK_PROMPT.format(
        query=query,
        candidates=_format_candidates(candidates),
    )

    try:
        response = await chat_without_tools([{"role": "user", "content": prompt}])
        content = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Rerank LLM call failed: %s", e)
        return None

    score_map = _parse_scores(content)
    if score_map is None:
        logger.warning("Rerank response not parseable, keeping vector order")
        return None

    # 模型可能漏掉个别 id，缺失的按 0 分处理（排到最后）
    return [score_map.get(i, 0.0) for i in range(len(candidates))]


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    """把候选列表格式化成带编号的文本。"""
    parts: list[str] = []

    for i, item in enumerate(candidates, 1):
        text = item["text"][:CANDIDATE_MAX_CHARS]
        parts.append(f"[{i}] (来源: {item.get('filename', 'unknown')})\n{text}")

    return "\n\n".join(parts)


def _parse_scores(content: str) -> dict[int, float] | None:
    """
    从模型回复中解析 {"id": .., "score": ..} 列表。

    模型输出偶尔会带 ```json 围栏或多余文字，这里做容错解析。
    返回 {候选下标: 分数}（模型给的 id 从 1 开始，这里转成 0 开始的下标）。
    """
    json_text = _extract_json(content)
    if json_text is None:
        return None

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    items = data.get("scores") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None

    score_map: dict[int, float] = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item["id"]) - 1   # 模型的 id 从 1 开始
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if index >= 0:
            score_map[index] = max(0.0, min(10.0, score))

    if not score_map:
        return None

    return score_map


def _extract_json(content: str) -> str | None:
    """从模型回复里截取第一段完整的 JSON 对象文本（容错 ```json 围栏）。"""
    start = content.find("{")
    if start == -1:
        return None

    # 从第一个 { 开始，找与之配对的 }
    depth = 0
    for i in range(start, len(content)):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return content[start:i + 1]

    return None
