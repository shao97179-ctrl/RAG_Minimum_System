"""
大模型调用模块。

封装智谱 AI GLM 模型的调用逻辑，包括：
- 普通对话
- Function Calling（Tool Calling）
- Embedding

关于异步: 智谱官方 SDK 是同步的（内部用 httpx 同步请求），
这里用 asyncio.to_thread 把同步调用丢进线程池执行，
让它不阻塞事件循环 —— 对外暴露的全部是 async 接口。

关于重试: 每次调用都经过 app/backoff.py 的指数退避，
超时 / 连接失败 / 限流(429) / 服务端错误(5xx) 会自动重试；
参数错误、鉴权失败等"重试也没用"的错误立即抛出。

这里只做"与 LLM 通信"这一件事，
不关心工具怎么执行，不关心 Agent 循环逻辑。
"""

import asyncio
import logging
from typing import Any

from zhipuai import (
    ZhipuAI,
    APIConnectionError,
    APIInternalError,
    APIReachLimitError,
    APIServerFlowExceedError,
    APITimeoutError,
)

from app.backoff import with_backoff
from app.config import ZHIPU_API_KEY, MODEL

logger = logging.getLogger(__name__)

# 值得重试的异常: 网络问题、超时、限流、服务端过载
RETRYABLE_LLM_ERRORS: tuple[type[Exception], ...] = (
    APIConnectionError,
    APITimeoutError,
    APIReachLimitError,        # 429 请求过快
    APIInternalError,          # 5xx 服务端错误
    APIServerFlowExceedError,  # 服务端流控
)

# 智谱客户端全局复用（每次调用都新建客户端没有必要）
_client: ZhipuAI | None = None


def get_client() -> ZhipuAI:
    """获取智谱 AI 客户端实例（全局单例）。"""
    global _client
    if _client is None:
        _client = ZhipuAI(api_key=ZHIPU_API_KEY)
    return _client


async def _call_api(desc: str, create: Any) -> Any:
    """
    在线程池里执行一次同步 SDK 调用，失败时按指数退避重试。

    参数:
        desc: 日志里显示的调用描述，如 "chat(completion)"
        create: 无参函数，内部调用同步 SDK

    返回:
        SDK 的响应对象
    """
    try:
        # asyncio.to_thread: 把同步 SDK 调用放进线程池，
        # 等待期间事件循环可以继续跑别的协程
        return await with_backoff(
            asyncio.to_thread, create,
            retry_on=RETRYABLE_LLM_ERRORS,
        )
    except Exception as e:
        logger.error("%s failed after retries: %s", desc, e)
        raise RuntimeError(f"智谱 API 调用失败: {e}") from e


async def chat_with_tools(
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

    def _create() -> Any:
        return client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",   # 让模型自己决定是否调用工具
        )

    return await _call_api("LLM chat(with tools)", _create)


async def chat_without_tools(
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

    def _create() -> Any:
        return client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )

    return await _call_api("LLM chat(without tools)", _create)


async def call_embedding(text: str) -> list[float]:
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

    def _create() -> Any:
        return client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )

    response = await _call_api("Embedding", _create)
    return response.data[0].embedding
