"""
嵌入模块。

调用智谱 Embedding API 将文本转换为向量。
支持单条和批量嵌入。
"""

import logging

from app.llm import call_embedding

logger = logging.getLogger(__name__)


def embed_text(text: str) -> list[float]:
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
        vector = call_embedding(text)
        logger.debug("Embedded text (len=%d) -> vector dim=%d", len(text), len(vector))
        return vector
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        raise


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    对多条文本生成 Embedding 向量（逐条调用）。

    参数:
        texts: 文本列表

    返回:
        向量列表，与输入顺序一一对应
    """
    vectors: list[list[float]] = []

    for i, text in enumerate(texts):
        try:
            vec = embed_text(text)
            vectors.append(vec)
        except Exception as e:
            logger.error("Embedding text[%d] failed: %s", i, e)
            raise

    logger.info("Embedded %d texts", len(vectors))
    return vectors
