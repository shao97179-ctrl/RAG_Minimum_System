"""
向量存储模块。

使用 ChromaDB 存储文档的 Embedding 向量，
支持插入和相似度检索。

关于异步: Embedding 生成走网络请求，是真正的异步 I/O；
ChromaDB 本身是同步的本地库，这里用 asyncio.to_thread
把建索引、查询这类可能较慢的本地操作放进线程池，
避免阻塞事件循环。
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import chromadb

from app.config import CHROMA_DIR, RAG_TOP_K
from app.rag.embedder import embed_text, embed_texts

logger = logging.getLogger(__name__)

# ChromaDB 集合名称
COLLECTION_NAME = "knowledge_base"


def get_chroma_client(persist_dir: Path | None = None) -> chromadb.PersistentClient:
    """
    获取 ChromaDB 持久化客户端。

    参数:
        persist_dir: 持久化目录，默认使用 config 中的 CHROMA_DIR

    返回:
        ChromaDB PersistentClient
    """
    if persist_dir is None:
        persist_dir = CHROMA_DIR

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """获取或创建知识库集合。"""
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Agent 知识库"},
    )
    return collection


async def build_index(chunks: list[dict[str, str]]) -> None:
    """
    构建 ChromaDB 索引。

    将所有 chunk 的 Embedding 向量写入 ChromaDB。
    Embedding 通过 embed_texts 受限并发生成（同时最多 5 个请求）。

    参数:
        chunks: [{"filename": "xxx.txt", "chunk_index": "0", "text": "..."}]
    """
    if not chunks:
        logger.warning("No chunks to index, skipping.")
        return

    client = get_chroma_client()
    # 如果已有数据，先删除旧集合
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info("Deleted existing collection")
    except Exception:
        pass  # 集合不存在，忽略

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Agent 知识库"},
    )

    # 并发生成所有 chunk 的 Embedding（单个失败就跳过该 chunk）
    texts_all = [chunk["text"] for chunk in chunks]
    paired: list[tuple[dict[str, str], list[float]]] = []

    results = await asyncio.gather(
        *(embed_text(text) for text in texts_all),
        return_exceptions=True,  # 单个失败不影响其他，失败位置返回异常对象
    )
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Failed to embed chunk[%d], skipping: %s", i, result)
            continue
        paired.append((chunks[i], result))

    if not paired:
        logger.error("No vectors generated, index build failed.")
        return

    # chunk 与向量一一配对，保证 ids/metadatas/documents 对齐
    valid_chunks = [chunk for chunk, _ in paired]
    vectors = [vec for _, vec in paired]
    texts = [chunk["text"] for chunk in valid_chunks]

    # 生成唯一 ID
    ids = [f"{chunk['filename']}_chunk_{chunk['chunk_index']}" for chunk in valid_chunks]

    # 元数据
    metadatas = [
        {"filename": chunk["filename"], "chunk_index": chunk["chunk_index"]}
        for chunk in valid_chunks
    ]

    # 写入 ChromaDB（本地操作放线程池，不阻塞事件循环）
    await asyncio.to_thread(
        collection.add,
        ids=ids,
        embeddings=vectors,
        documents=texts[: len(vectors)],
        metadatas=metadatas,
    )

    logger.info("Indexed %d chunks into ChromaDB", len(vectors))


async def search(query: str, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
    """
    在 ChromaDB 中检索与 query 最相似的文档 chunk。

    参数:
        query: 查询文本
        top_k: 返回最相似的 K 个结果

    返回:
        [{"text": "chunk文本", "filename": "来源文件", "distance": 距离}]
    """
    # 生成 query 的 Embedding（网络请求，异步）
    try:
        query_vector = await embed_text(query)
    except Exception as e:
        logger.error("Failed to embed query: %s", e)
        raise RuntimeError(f"查询 Embedding 失败: {e}") from e

    # 查询 ChromaDB（本地操作放线程池）
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        logger.error("ChromaDB collection not found: %s", e)
        raise RuntimeError(
            "知识库集合不存在，请先运行 RAG 知识库初始化。"
        ) from e

    if collection.count() == 0:
        logger.warning("Knowledge base is empty")
        return []

    results = await asyncio.to_thread(
        collection.query,
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # 解析结果
    retrieved: list[dict[str, Any]] = []

    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0

            retrieved.append({
                "text": doc,
                "filename": metadata.get("filename", "unknown"),
                "distance": distance,
            })

    logger.info("RAG search: '%s' -> retrieved %d documents", query, len(retrieved))
    return retrieved


async def is_index_built() -> bool:
    """检查 ChromaDB 索引是否已构建（集合存在且有数据）。"""
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection.count() > 0
    except Exception:
        return False
