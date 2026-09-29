"""
项目配置模块。

所有配置从环境变量读取，不硬编码任何 API Key 或敏感信息。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ──────────────────────────────────────────────
# 项目路径
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma_db"

# ──────────────────────────────────────────────
# 智谱 AI 配置
# ──────────────────────────────────────────────
ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "")

# 智谱 API 官方地址
ZHIPU_API_BASE: str = "https://open.bigmodel.cn/api/paas/v4"

# 使用 glm-4-flash：支持 Function Calling，速度快，成本低
MODEL: str = "glm-4.5-air"

# ──────────────────────────────────────────────
# OpenWeather 配置
# ──────────────────────────────────────────────
OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_API_URL: str = "https://api.openweathermap.org/data/2.5/weather"

# ──────────────────────────────────────────────
# RAG 配置
# ──────────────────────────────────────────────
# 智谱 Embedding 模型
EMBEDDING_MODEL: str = "embedding-3"

# 文本切分参数
CHUNK_SIZE: int = 300       # 每个chunk最多多少字符
CHUNK_OVERLAP: int = 50     # chunk之间的重叠字符数

# 检索参数
RAG_TOP_K: int = 4          # 重排后最终返回最相关的K个chunk

# ──────────────────────────────────────────────
# RAG 检索增强配置（问题重写 + 重排）
# ──────────────────────────────────────────────
# 是否启用"问题重写"：检索前先用 LLM 把问题改写成独立、明确的查询
RAG_REWRITE_ENABLED: bool = True
# 问题重写时参考最近几条对话消息（用于解决"它/这个"等指代）
RAG_REWRITE_HISTORY_MSGS: int = 4
# 是否启用"重排"：向量粗检索后，用 LLM 给候选片段按相关性打分精排
RAG_RERANK_ENABLED: bool = True
# 向量粗检索召回的候选数量（先多召回一些，重排后再取 RAG_TOP_K 个）
RAG_CANDIDATE_K: int = 8

# ──────────────────────────────────────────────
# 上下文管理配置
# ──────────────────────────────────────────────
# 对话历史的最大 token 预算（估算值），超过则触发压缩
MAX_CONTEXT_TOKENS: int = 6000
# 压缩时为"最近对话"保留的 token 预算，更早的对话会被压缩成摘要
KEEP_RECENT_TOKENS: int = 3000

# ──────────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────────
# 日志文件目录（日志只写文件，不在终端显示）
LOGS_DIR = PROJECT_ROOT / "logs"
# 日志级别: DEBUG / INFO / WARNING / ERROR
LOG_LEVEL: str = "INFO"
# 日志保留天数：每次启动时自动删除比它更早的日志文件
LOG_RETENTION_DAYS: int = 7


def validate_config() -> None:
    """启动时校验关键配置项，缺失则抛出明确错误。"""
    errors: list[str] = []

    if not ZHIPU_API_KEY:
        errors.append(
            "缺少 ZHIPU_API_KEY。"
            "请在 .env 文件中设置，或通过环境变量传入。"
        )

    if not OPENWEATHER_API_KEY:
        errors.append(
            "缺少 OPENWEATHER_API_KEY。"
            "请在 .env 文件中设置，或通过环境变量传入。"
        )

    if errors:
        raise RuntimeError("\n".join(errors))
