"""
Tool Dispatcher —— 工具分发器。

这是 Agent 系统的核心组件之一：

    LLM 返回 tool_call
          ↓
    读取 tool name
          ↓
    根据 name 找到 Python 函数
          ↓
    解析 arguments
          ↓
    执行函数（全部为异步调用）
          ↓
    得到结果
          ↓
    构造 tool message

这个模块只做"根据工具名称找到并执行对应函数"这一件事，
不关心 Agent 的循环逻辑，也不关心 LLM 的调用方式。
"""

import asyncio
import inspect
import json
import logging
from typing import Any, Callable

from app.tools.calculator import calculate
from app.tools.rag import search_knowledge_base
from app.tools.weather import get_weather
from app.tools.time_tool import get_time

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 工具映射表：工具名称 → 异步 Python 函数
# ──────────────────────────────────────────────
TOOLS_MAP: dict[str, Callable[..., Any]] = {
    "search_knowledge_base": search_knowledge_base,
    "get_weather": get_weather,
    "calculate": calculate,
    "get_time": get_time
}


async def dispatch_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> str:
    """
    根据工具名称和参数，异步执行对应的 Python 函数。

    这是 Tool Dispatcher 的核心函数。

    参数:
        tool_name: 工具名称，如 "get_weather"
        tool_args: 工具参数，如 {"city": "北京"}
        history: 对话历史（可选）。工具函数只要声明了 history 参数，
            就会自动拿到它（如 RAG 检索用历史做问题重写），
            该参数不出现在 LLM 看到的工具 Schema 里

    返回:
        工具执行结果字符串

    异常:
        ValueError: 工具名称不在映射表中
    """
    if tool_name not in TOOLS_MAP:
        available = ", ".join(TOOLS_MAP.keys())
        error_msg = f"未知工具 '{tool_name}'，可用工具: {available}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    func = TOOLS_MAP[tool_name]
    logger.info("Dispatching tool: %s(%s)", tool_name, _format_args(tool_args))

    # 按函数签名注入对话历史（工具 Schema 无需向 LLM 暴露该参数）
    kwargs: dict[str, Any] = dict(tool_args)
    if _accepts_history(func):
        kwargs["history"] = history

    try:
        result = await func(**kwargs)
        logger.info("Tool finished: %s", tool_name)
        return result
    except TypeError as e:
        error_msg = f"工具 '{tool_name}' 参数错误: {e}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"工具 '{tool_name}' 执行失败: {e}"
        logger.error(error_msg)
        return error_msg


def _accepts_history(func: Callable[..., Any]) -> bool:
    """检查工具函数是否声明了 history 参数。"""
    try:
        return "history" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def parse_tool_args(args_str: str) -> dict[str, Any]:
    """
    将 LLM 返回的工具参数 JSON 字符串解析为字典。

    参数:
        args_str: JSON 格式的参数字符串

    返回:
        参数字典
    """
    if not args_str:
        return {}

    try:
        return json.loads(args_str)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse tool args: %s - %s", args_str, e)
        return {}


def build_tool_message(tool_call_id: str, tool_name: str, result: str) -> dict[str, str]:
    """
    构造返回给 LLM 的 tool message。

    当工具执行完毕后，需要将结果告诉 LLM，
    格式遵循 OpenAI/智谱 的 tool message 规范。

    参数:
        tool_call_id: 工具调用的唯一标识（来自 LLM 的 tool_call.id）
        tool_name: 工具名称
        result: 工具执行结果

    返回:
        符合消息格式的字典
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": result,
    }


def _format_args(args: dict[str, Any]) -> str:
    """格式化参数字典用于日志（不打印敏感信息）。"""
    return json.dumps(args, ensure_ascii=False)
