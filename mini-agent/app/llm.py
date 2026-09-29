"""
大模型调用模块。

封装智谱 AI GLM 模型的调用逻辑，包括：
- 普通对话
- Function Calling（Tool Calling）

这里只做"与 LLM 通信"这一件事，
不关心工具怎么执行，不关心 Agent 循环逻辑。
"""

import logging
from typing import Any

from zhipuai import ZhipuAI

from app.config import ZHIPU_API_KEY, MODEL

logger = logging.getLogger(__name__)


def get_client() -> ZhipuAI:
    """获取智谱 AI 客户端实例。"""
    return ZhipuAI(api_key=ZHIPU_API_KEY)


def chat_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> Any:
    """
    调用智谱 GLM，带 Tool Calling 能力。

    参数:
        messages: 对话历史，格式同 OpenAI messages
        tools: 工具定义列表，格式同 OpenAI tools

    返回:
        模型的响应对象
    """
    client = get_client()
    logger.info("Calling LLM %s with %d tools", MODEL, len(tools))

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",   # 让模型自己决定是否调用工具
        )
    except Exception as e:
        logger.error("LLM API call failed: %s", e)
        raise RuntimeError(f"智谱 API 调用失败: {e}") from e

    return response


def chat_without_tools(
    messages: list[dict[str, Any]],
) -> Any:
    """
    调用智谱 GLM，不带工具（纯对话）。

    参数:
        messages: 对话历史

    返回:
        模型的响应对象
    """
    client = get_client()
    logger.info("Calling LLM %s without tools", MODEL)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
    except Exception as e:
        logger.error("LLM API call failed: %s", e)
        raise RuntimeError(f"智谱 API 调用失败: {e}") from e

    return response


def call_embedding(text: str) -> list[float]:
    """
    调用智谱 Embedding API，获取文本的向量表示。

    参数:
        text: 待嵌入的文本

    返回:
        浮点数向量
    """
    from app.config import EMBEDDING_MODEL

    client = get_client()
    logger.debug("Calling embedding API for text (len=%d)", len(text))

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error("Embedding API call failed: %s", e)
        raise RuntimeError(f"智谱 Embedding API 调用失败: {e}") from e
