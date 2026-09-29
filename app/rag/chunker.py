"""
文本切分器。

将长文本按固定字符数切分为多个 chunk，
相邻 chunk 之间有重叠，避免语义断裂。
"""

import logging

from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    将文本切分为多个 chunk。

    切分策略：按字符数滑动窗口，相邻 chunk 有重叠。

    参数:
        text: 待切分的文本
        chunk_size: 每个 chunk 的最大字符数
        chunk_overlap: 相邻 chunk 的重叠字符数

    返回:
        chunk 字符串列表
    """
    if not text or not text.strip():
        return []

    # 如果文本比 chunk_size 短，直接返回
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:  # 跳过空白 chunk
            chunks.append(chunk)

        # 滑动窗口：前进 (chunk_size - chunk_overlap) 个字符
        start += chunk_size - chunk_overlap

    logger.debug("Split text into %d chunks (chunk_size=%d, overlap=%d)",
                 len(chunks), chunk_size, chunk_overlap)
    return chunks


def chunk_documents(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    对多个文档进行切分。

    参数:
        documents: [{"filename": "xxx.txt", "content": "..."}]

    返回:
        [{"filename": "xxx.txt", "chunk_index": 0, "text": "..."}]
    """
    all_chunks: list[dict[str, str]] = []

    for doc in documents:
        chunks = split_text(doc["content"])
        for idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "filename": doc["filename"],
                "chunk_index": str(idx),
                "text": chunk_text,
            })

    logger.info("Chunked %d documents into %d chunks", len(documents), len(all_chunks))
    return all_chunks
