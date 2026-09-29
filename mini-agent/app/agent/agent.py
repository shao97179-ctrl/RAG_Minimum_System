"""
Agent 核心循环模块。

这是整个系统最核心的部分，实现了 Agent Loop：

    用户输入
        ↓
    智谱 GLM
        ↓
    LLM 判断是否需要工具
        ↓
    ┌─ 不需要 → 直接返回回答
    └─ 需要   → Function Calling
                ↓
            Tool Dispatcher
                ↓
            执行对应工具
                ↓
            Tool Result
                ↓
            返回给 GLM
                ↓
            GLM 最终回答

支持：
- 多轮对话（维护 messages 历史）
- 一次返回多个 Tool Call
- 工具执行完成后自动再次调用 LLM 生成最终回答
- 上下文压缩（每轮开始时检查，历史过大则压缩，见 context_manager.py）
"""

import logging
from typing import Any

from app.agent.context_manager import compress_history, estimate_history_tokens
from app.agent.prompts import SYSTEM_PROMPT, TOOLS
from app.agent.tool_dispatcher import (
    build_tool_message,
    dispatch_tool,
    parse_tool_args,
)
from app.llm import chat_with_tools

logger = logging.getLogger(__name__)

# 最大 Agent 循环轮次（防止无限循环）
MAX_ITERATIONS = 5


def process_tool_calls(
    tool_calls: list[Any],
    history: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """
    处理 LLM 返回的所有 tool_call。

    遍历每个 tool_call:
    1. 读取工具名称和参数
    2. 通过 Tool Dispatcher 执行
    3. 构造 assistant 消息（含 tool_call 信息）
    4. 构造 tool 结果消息

    参数:
        tool_calls: LLM 返回的 tool_call 列表
        history: 对话历史（可选），Dispatcher 会注入给声明了
            history 参数的工具（如 RAG 检索的问题重写）

    返回:
        (assistant_message, tool_messages)
        - assistant_message: 包含 tool_call 信息的 assistant 消息
        - tool_messages: 所有工具结果消息列表
    """
    tool_messages: list[dict[str, str]] = []
    assistant_tool_calls: list[dict[str, Any]] = []

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_args_str = tool_call.function.arguments
        tool_call_id = tool_call.id

        logger.info("Agent decided to call tool: %s", tool_name)

        # 解析参数
        tool_args = parse_tool_args(tool_args_str)

        # 记录 assistant 的 tool_call 信息（后续需要加入消息历史）
        assistant_tool_calls.append({
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": tool_args_str,
            },
        })

        # 执行工具（对话历史会注入给需要的工具）
        result = dispatch_tool(tool_name, tool_args, history=history)

        # 构造 tool 结果消息
        tool_msg = build_tool_message(tool_call_id, tool_name, result)
        tool_messages.append(tool_msg)

    # 构造 assistant 消息（包含 tool_calls）
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": assistant_tool_calls,
    }

    return assistant_message, tool_messages


def agent_loop(user_input: str, history: list[dict[str, Any]]) -> str:
    """
    Agent 主循环。

    流程:
    1. 将用户输入加入消息列表
    2. 检查上下文是否超限，超限则压缩（旧对话 → 摘要，新对话保留）
    3. 调用 LLM
    4. 如果 LLM 返回 tool_call:
       a. 执行工具
       b. 将结果返回给 LLM
       c. 再次调用 LLM 获取最终回答
    5. 如果 LLM 直接返回文本:
       a. 作为最终回答

    参数:
        user_input: 用户输入
        history: 对话历史（会被就地更新）

    返回:
        Agent 的最终回答文本
    """
    # 将用户消息加入历史
    history.append({"role": "user", "content": user_input})
    logger.info("User query: %s", user_input)

    # 上下文压缩：历史超过 token 预算时，较早的轮次被压缩成摘要
    if compress_history(history):
        logger.info(
            "Context compressed: %d messages, ~%d tokens now",
            len(history),
            estimate_history_tokens(history),
        )

    # 构建发给 LLM 的消息：系统提示 + 完整历史（可能已被压缩）
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # Agent 循环
    for iteration in range(MAX_ITERATIONS):
        logger.debug("Agent loop iteration %d", iteration + 1)

        # 调用 LLM
        response = chat_with_tools(messages, TOOLS)

        # 解析响应
        choice = response.choices[0]
        assistant_msg = choice.message

        # 检查是否有 tool_call
        if assistant_msg.tool_calls:
            logger.info(
                "LLM requested %d tool call(s)",
                len(assistant_msg.tool_calls),
            )

            # 处理所有 tool_call（历史会注入给声明了 history 参数的工具）
            assistant_message, tool_messages = process_tool_calls(
                assistant_msg.tool_calls, history=history
            )

            # 将 assistant 消息和 tool 消息加入历史
            history.append(assistant_message)
            history.extend(tool_messages)

            # 更新 messages，让 LLM 看到工具结果
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

            # 继续循环，让 LLM 基于工具结果生成回答
            continue

        # 没有工具调用，直接获取文本回答
        final_answer = assistant_msg.content or ""

        # 将 assistant 回答加入历史
        if final_answer:
            history.append({"role": "assistant", "content": final_answer})

        logger.info("Agent generating final answer")
        return final_answer

    # 超过最大循环次数
    logger.warning("Agent loop reached max iterations (%d)", MAX_ITERATIONS)
    return "抱歉，处理过程中出现了问题，请重试。"
