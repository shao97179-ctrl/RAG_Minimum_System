"""
上下文管理模块：限制对话历史的最大长度，超限时自动压缩。

核心思路：
1. 用"估算 token 数"衡量历史大小
   （中文字符 ≈ 1 token，其他字符 ≈ 1/4 token，不依赖额外库）
2. 历史按"轮次"分组：一轮 = 一条用户消息 + 其后的 assistant / tool 消息。
   分组保证压缩只在轮次边界进行——
   assistant 的 tool_calls 消息和对应的 tool 结果消息绝不能被拆散，
   否则消息序列不符合 API 规范，LLM 会报错。
3. 历史超过 MAX_CONTEXT_TOKENS 时：
   - 较早的轮次 → 用 LLM 生成一段摘要（失败则直接丢弃，退化为截断）
   - 最近的轮次 → 原样保留
4. 摘要以 system 消息的形式放在历史开头，LLM 仍然能参考早期对话的关键信息。
"""

import logging
from typing import Any, Callable

from app.config import KEEP_RECENT_TOKENS, MAX_CONTEXT_TOKENS
from app.llm import chat_without_tools

logger = logging.getLogger(__name__)

# 摘要提示词中，单条消息内容最多保留的字符数
SUMMARY_MSG_MAX_CHARS = 200
# 摘要提示词的总字符数上限（避免摘要请求本身过大）
SUMMARY_TRANSCRIPT_MAX_CHARS = 6000


# ──────────────────────────────────────────────
# Token 估算
# ──────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    粗略估算文本的 token 数。

    规则：中文字符（含中文标点范围的汉字）按 1 个 token 计，
    其他字符（英文、数字、符号）按 4 个字符 ≈ 1 个 token 计。

    这是一个不依赖分词库的近似值，用于判断"是否需要压缩"足够了。
    """
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return chinese_chars + other_chars // 4


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """
    估算单条消息的 token 数。

    除了 content，assistant 消息里的 tool_calls
    （工具名 + 参数 JSON）也会占用 token，需要一并计算。
    """
    tokens = estimate_tokens(message.get("content") or "")

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function", {})
        tokens += estimate_tokens(function.get("name", ""))
        tokens += estimate_tokens(function.get("arguments", ""))

    return tokens


def estimate_history_tokens(history: list[dict[str, Any]]) -> int:
    """估算整个对话历史的 token 数。"""
    return sum(estimate_message_tokens(msg) for msg in history)


# ──────────────────────────────────────────────
# 历史分组
# ──────────────────────────────────────────────

def split_into_turns(history: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """
    将历史按用户消息切分成"轮次"。

    每轮从一条 user 消息开始，到下一条 user 消息之前结束。
    这样一轮内的 assistant(tool_calls) + tool 消息组永远在同一轮里，
    压缩时按轮丢弃/保留，就不会产生孤立的 tool 消息。
    """
    turns: list[list[dict[str, Any]]] = []
    current_turn: list[dict[str, Any]] = []

    for msg in history:
        if msg.get("role") == "user" and current_turn:
            turns.append(current_turn)
            current_turn = [msg]
        else:
            current_turn.append(msg)

    if current_turn:
        turns.append(current_turn)

    return turns


# ──────────────────────────────────────────────
# 摘要生成
# ──────────────────────────────────────────────

def summarize_turns(turns: list[list[dict[str, Any]]]) -> str | None:
    """
    用 LLM 为较早的对话轮次生成摘要。

    返回摘要文本；调用失败时返回 None（调用方退化为直接截断）。
    """
    lines: list[str] = []

    for turn in turns:
        for msg in turn:
            role = msg.get("role", "unknown")

            if role == "tool":
                content = "（工具结果）" + (msg.get("content") or "")[:SUMMARY_MSG_MAX_CHARS]
            else:
                content = msg.get("content") or ""
                # assistant 调用过工具的消息，content 可能为空，补充工具名
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    names = [tc["function"]["name"] for tc in tool_calls]
                    content = f"{content}（调用了工具: {', '.join(names)}）"
                content = content[:SUMMARY_MSG_MAX_CHARS]

            lines.append(f"{role}: {content}")

    transcript = "\n".join(lines)
    if len(transcript) > SUMMARY_TRANSCRIPT_MAX_CHARS:
        transcript = transcript[:SUMMARY_TRANSCRIPT_MAX_CHARS] + "\n...（过长已截断）"

    prompt = (
        "请用不超过300字总结以下对话的关键信息，"
        "包括：用户关注什么主题、得到的重要结论和数据。只输出摘要内容：\n\n"
        f"{transcript}"
    )

    try:
        response = chat_without_tools([{"role": "user", "content": prompt}])
        summary = response.choices[0].message.content or ""
        logger.info("Generated summary for %d old turns (%d chars)", len(turns), len(summary))
        return summary
    except Exception as e:
        logger.warning("Summarization failed, will truncate instead: %s", e)
        return None


# ──────────────────────────────────────────────
# 压缩入口
# ──────────────────────────────────────────────

def compress_history(
    history: list[dict[str, Any]],
    summarize_fn: Callable[[list[list[dict[str, Any]]]], str | None] | None = None,
) -> bool:
    """
    如果历史超过 token 预算，就地压缩它。

    参数:
        history: 对话历史（会被就地修改，列表对象本身保持不变）
        summarize_fn: 摘要函数（注入用于测试，默认用 LLM 摘要）

    返回:
        是否执行了压缩
    """
    if summarize_fn is None:
        summarize_fn = summarize_turns

    total_tokens = estimate_history_tokens(history)
    if total_tokens <= MAX_CONTEXT_TOKENS:
        return False

    turns = split_into_turns(history)

    # 从最新一轮往前累加，确定保留区的起始位置。
    # 即使最新一轮本身超预算，也至少保留它（i == len(turns) - 1 时不判断预算）。
    split_index = len(turns)
    kept_tokens = 0

    for i in range(len(turns) - 1, -1, -1):
        turn_tokens = sum(estimate_message_tokens(msg) for msg in turns[i])
        if i < len(turns) - 1 and kept_tokens + turn_tokens > KEEP_RECENT_TOKENS:
            break
        kept_tokens += turn_tokens
        split_index = i

    old_turns = turns[:split_index]
    recent_turns = turns[split_index:]

    if not old_turns:
        # 单轮就超预算且无法再分，不做处理（交给 MAX_ITERATIONS 层面兜底）
        logger.warning("Single turn exceeds budget, skip compression")
        return False

    logger.info(
        "Compressing history: %d tokens -> old %d turns summarized, "
        "recent %d turns kept (%d tokens)",
        total_tokens, len(old_turns), len(recent_turns), kept_tokens,
    )

    # 旧轮次生成摘要；失败时退化为直接丢弃
    summary = summarize_fn(old_turns)

    new_history: list[dict[str, Any]] = []
    if summary:
        new_history.append({
            "role": "system",
            "content": f"[历史对话摘要]\n{summary}",
        })
    else:
        logger.warning("Summary unavailable, old turns dropped entirely")

    for turn in recent_turns:
        new_history.extend(turn)

    # 就地替换，保证 main.py 持有的 history 引用仍然有效
    history[:] = new_history
    return True
