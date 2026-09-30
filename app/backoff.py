"""
指数退避（Exponential Backoff）重试模块。

网络请求（LLM API、天气 API 等）偶尔会因为限流、超时、服务端抖动而失败，
这类失败通常"等一会儿再试"就能成功。指数退避的思路是：

    第 1 次重试等 1 秒 → 第 2 次等 2 秒 → 第 3 次等 4 秒 → ...

每次失败后等待时间翻倍（delay = base_delay * 2^重试次数），
再叠加一点随机抖动（jitter），避免多个请求在同一时刻扎堆重试。

使用方式: 把一个异步函数交给 with_backoff 执行即可：

    result = await with_backoff(fetch_data, "参数", max_retries=3)

只重试 retry_on 里声明的"值得重试"的异常（超时、限流等）；
参数错误、鉴权失败这类"重试也没用"的异常会立刻抛出。
"""

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


async def with_backoff(
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """
    执行异步函数 func(*args, **kwargs)，失败时按指数退避自动重试。

    参数:
        func: 要执行的异步函数（async def 定义）
        *args, **kwargs: 透传给 func 的参数
        max_retries: 失败后最多重试几次（总尝试次数 = 1 + max_retries）
        base_delay: 首次重试的等待秒数，之后每次翻倍
        max_delay: 单次等待的秒数上限（指数增长会被截断到这里）
        retry_on: 需要重试的异常类型；不在其中的异常立即抛出

    返回:
        func 的返回值

    异常:
        重试次数用完后仍失败时，抛出最后一次的异常
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retry_on as e:
            last_error = e

            # 已经是最后一次机会，不再等待，直接抛出
            if attempt >= max_retries:
                break

            # 指数退避: 1s → 2s → 4s ...，封顶 max_delay
            delay = min(base_delay * (2 ** attempt), max_delay)
            # 抖动: 在 0.5~1 倍之间随机取值，避免请求扎堆重试
            delay *= random.uniform(0.5, 1.0)

            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt + 1, max_retries + 1, e, delay,
            )
            await asyncio.sleep(delay)

    assert last_error is not None
    logger.error(
        "All %d attempts failed, giving up: %s", max_retries + 1, last_error
    )
    raise last_error
