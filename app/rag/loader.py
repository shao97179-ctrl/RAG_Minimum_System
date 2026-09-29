"""
文档加载器。

从 data/documents/ 目录读取文本文件，
返回文件名和内容的列表。
"""

import logging
from pathlib import Path

from app.config import DOCUMENTS_DIR

logger = logging.getLogger(__name__)


def load_documents(directory: Path | None = None) -> list[dict[str, str]]:
    """
    从指定目录加载所有 .txt 文件。

    返回:
        列表，每项是 {"filename": "xxx.txt", "content": "文件内容"}
    """
    if directory is None:
        directory = DOCUMENTS_DIR

    if not directory.exists():
        logger.warning("Documents directory not found: %s", directory)
        raise FileNotFoundError(f"知识库文档目录不存在: {directory}")

    documents: list[dict[str, str]] = []

    for filepath in sorted(directory.glob("*.txt")):
        try:
            content = filepath.read_text(encoding="utf-8")
            documents.append({
                "filename": filepath.name,
                "content": content,
            })
            logger.info("Loaded document: %s (%d chars)", filepath.name, len(content))
        except Exception as e:
            logger.error("Failed to load %s: %s", filepath.name, e)

    if not documents:
        logger.warning("No documents found in %s", directory)

    return documents
