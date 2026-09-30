"""
指数退避模块单元测试。

通过替换 asyncio.sleep 验证重试次数和退避间隔，
测试本身不会真的等待。
"""

import asyncio

import pytest

from app.backoff import with_backoff


class TestWithBackoff:
    """with_backoff 测试。"""

    async def test_success_first_try(self) -> None:
        """首次成功直接返回，不重试。"""
        calls: list[int] = []

        async def fn(x: int) -> int:
            calls.append(x)
            return x * 2

        assert await with_backoff(fn, 3) == 6
        assert calls == [3]

    async def test_retries_then_succeeds(self, monkeypatch) -> None:
        """前几次失败后重试，最终成功；退避间隔逐次翻倍。"""
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        attempts: list[int] = []

        async def flaky() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("网络抖动")
            return "ok"

        result = await with_backoff(
            flaky, max_retries=3, base_delay=1.0, max_delay=20.0,
            retry_on=(ConnectionError,),
        )

        assert result == "ok"
        assert len(attempts) == 3          # 失败 2 次 + 成功 1 次
        assert len(sleeps) == 2            # 重试了 2 次，各等待一次
        # 第 1 次等待: 1.0 * 2^0 = 1s，乘 0.5~1.0 抖动
        assert 0.5 <= sleeps[0] <= 1.0
        # 第 2 次等待: 1.0 * 2^1 = 2s，乘 0.5~1.0 抖动
        assert 1.0 <= sleeps[1] <= 2.0

    async def test_gives_up_after_max_retries(self, monkeypatch) -> None:
        """超过最大重试次数后抛出最后一次的异常。"""

        async def fake_sleep(delay: float) -> None:
            pass

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        attempts: list[int] = []

        async def always_fail() -> None:
            attempts.append(1)
            raise ConnectionError("一直失败")

        with pytest.raises(ConnectionError):
            await with_backoff(always_fail, max_retries=2, retry_on=(ConnectionError,))

        # 总尝试次数 = 1 + max_retries
        assert len(attempts) == 3

    async def test_non_retryable_raises_immediately(self) -> None:
        """不在 retry_on 里的异常不重试，直接抛出。"""
        calls: list[int] = []

        async def bad_args() -> None:
            calls.append(1)
            raise ValueError("参数错误，重试也没用")

        with pytest.raises(ValueError):
            await with_backoff(bad_args, max_retries=5, retry_on=(ConnectionError,))

        assert calls == [1]

    async def test_delay_capped_at_max_delay(self, monkeypatch) -> None:
        """指数增长被 max_delay 封顶。"""
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        async def always_fail() -> None:
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError):
            await with_backoff(
                always_fail, max_retries=4, base_delay=1.0, max_delay=2.0,
                retry_on=(ConnectionError,),
            )

        # 等待序列名义上是 1s 2s 2s 2s（封顶），抖动后不超过各自的封顶值
        assert len(sleeps) == 4
        assert all(d <= 2.0 for d in sleeps)
