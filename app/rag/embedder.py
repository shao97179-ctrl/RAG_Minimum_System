"""
嵌入模块。

调用智谱 Embedding API 将文本转换为向量。
支持单条和批量嵌入；批量嵌入时用信号量限制并发数，
多段文本同时请求，比逐条串行快很多。
"""

import asyncio
import logging

from app.llm import call_embedding

logger = logging.getLogger(__name__)

# 批量嵌入的最大并发请求数（太高容易被限流，太低没提速效果）
EMBED_CONCURRENCY = 5


async def embed_text(text: str) -> list[float]:
    """
    对单条文本生成 Embedding 向量。

    参数:
        text: 待嵌入的文本

    返回:
        浮点数向量

    异常:
        RuntimeError: Embedding API 调用失败
    """
    if not text or not text.strip():
        raise ValueError("嵌入文本不能为空")

    try:
        vector = await call_embedding(text)
        logger.debug("Embedded text (len=%d) -> vector dim=%d", len(text), len(vector))
        return vector
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        raise


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    对多条文本生成 Embedding 向量（受限并发，同时最多 EMBED_CONCURRENCY 个请求）。

    参数:
        texts: 文本列表

    返回:
        向量列表，与输入顺序一一对应

    异常:
        任何一条嵌入失败就抛出异常（asyncio.gather 默认行为），
        由调用方决定跳过还是中止
    """
    # 信号量: 限制同时在飞的请求数，拿到许可才能发请求
    semaphore = asyncio.Semaphore(EMBED_CONCURRENCY)

    async def _embed_one(index: int, text: str) -> list[float]:
        async with semaphore:
            try:
                return await embed_text(text)
            except Exception as e:
                logger.error("Embedding text[%d] failed: %s", index, e)
                raise

    # gather 并发执行所有嵌入任务，结果顺序与输入顺序一致
    vectors = await asyncio.gather(*(
        _embed_one(i, text) for i, text in enumerate(texts)
    ))

    logger.info("Embedded %d texts (concurrency=%d)", len(vectors), EMBED_CONCURRENCY)
    return list(vectors)
