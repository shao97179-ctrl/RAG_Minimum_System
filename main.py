"""
Mini Agent —— 最小但完整的 Agent 系统

启动方式:
    python main.py              # 启动交互式对话
    python main.py --init-rag   # 初始化 RAG 知识库索引

日志:
    写入 logs/ 目录下的日志文件，不在终端显示。
    文件名带日期（一天一个文件），超过 LOG_RETENTION_DAYS 天的
    旧日志在启动时自动清理（配置见 app/config.py）。

Agent 核心流程:
    用户输入 → 智谱 GLM → Function Calling → Tool Dispatcher → 执行工具 → 返回结果 → GLM 最终回答

全流程异步: LLM 调用、RAG 检索、工具执行都是 async/await，
网络请求失败自动按指数退避重试（见 app/backoff.py）。
"""

import argparse
import asyncio
import logging
import sys

from app.logging_config import setup_logging

logger = logging.getLogger(__name__)


def fail_fast(message: str) -> None:
    """致命启动错误：写入日志文件，并在终端给出提示后退出。"""
    logger.error(message)
    print(f"\n[启动失败] {message}", file=sys.stderr)
    print(f"详情请查看 logs/ 目录下今天的日志文件。\n", file=sys.stderr)
    sys.exit(1)


async def init_rag() -> None:
    """
    初始化 RAG 知识库（异步）。

    流程: 读取文档 → 文本切分 → 并发生成 Embedding → 写入 ChromaDB
    """
    from app.config import validate_config
    from app.rag.loader import load_documents
    from app.rag.chunker import chunk_documents
    from app.rag.vector_store import build_index, is_index_built

    try:
        validate_config()
    except RuntimeError as e:
        fail_fast(f"配置校验失败: {e}")

    logger.info("开始初始化 RAG 知识库...")

    # 1. 加载文档
    try:
        documents = load_documents()
    except FileNotFoundError as e:
        fail_fast(f"文档加载失败: {e}")

    if not documents:
        fail_fast("没有找到任何文档，请检查 data/documents/ 目录")

    logger.info("加载了 %d 个文档", len(documents))

    # 2. 文本切分
    chunks = chunk_documents(documents)
    if not chunks:
        fail_fast("文档切分结果为空")

    logger.info("切分为 %d 个 chunk", len(chunks))

    # 3. 生成 Embedding 并写入 ChromaDB
    logger.info("正在并发生成 Embedding 并构建索引（这可能需要几分钟）...")
    try:
        await build_index(chunks)
    except Exception as e:
        fail_fast(f"索引构建失败: {e}")

    # 4. 验证
    if await is_index_built():
        logger.info("RAG 知识库初始化完成！")
        print("RAG 知识库初始化完成！")
    else:
        fail_fast("索引构建后验证失败")


async def run_agent() -> None:
    """
    启动 Agent 交互式对话（异步）。

    用户可以输入问题，Agent 根据问题自动判断是否需要调用工具。
    对话内命令:
        /clear  清空上下文，开始全新对话
        /status 查看当前上下文用量
    输入 'quit' 或 'exit' 退出。
    """
    from app.agent.agent import agent_loop
    from app.agent.context_manager import estimate_history_tokens
    from app.config import validate_config, MAX_CONTEXT_TOKENS
    from app.rag.vector_store import is_index_built

    try:
        validate_config()
    except RuntimeError as e:
        fail_fast(f"配置校验失败: {e}")

    # 检查 RAG 知识库是否已初始化
    if not await is_index_built():
        logger.warning(
            "RAG 知识库尚未初始化。"
            "请先运行: python main.py --init-rag"
        )

    # 对话历史（内存中维护）
    history: list[dict[str, str]] = []

    print("=" * 60)
    print("  Mini Agent - 最小 Agent 系统")
    print("  模型: 智谱 AI GLM-4-Flash")
    print("  工具: search_knowledge_base / get_weather / calculate")
    print("  命令: /clear 清空上下文 | /status 查看上下文用量")
    print("  日志: 写入 logs/ 目录，不在终端显示")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    print()

    while True:
        try:
            # input() 是阻塞的，但单用户 CLI 场景下没有其他协程需要运行，
            # 等待用户输入时阻塞事件循环没有影响
            user_input = input("用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 清空上下文，开始新对话
        if user_input in ("/clear", "/new"):
            history.clear()
            logger.info("Context cleared by user command /clear")
            print("\n[系统] 上下文已清空，开始新对话。\n")
            continue

        # 查看上下文用量
        if user_input == "/status":
            used = estimate_history_tokens(history)
            print(f"\n[系统] 上下文用量: {len(history)} 条消息, "
                  f"约 {used} / {MAX_CONTEXT_TOKENS} tokens\n")
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        # 调用 Agent（异步执行整轮对话）
        try:
            answer = await agent_loop(user_input, history)
            print(f"\nAgent: {answer}\n")
        except Exception as e:
            logger.error("Agent error: %s", e, exc_info=True)
            print(f"\n[错误] {e}\n")


def main() -> None:
    """程序入口。"""
    # 初始化日志：只写文件，不在终端显示
    log_file = setup_logging()
    logger.info("Mini Agent 启动，日志文件: %s", log_file)

    parser = argparse.ArgumentParser(description="Mini Agent - 最小 Agent 系统")
    parser.add_argument(
        "--init-rag",
        action="store_true",
        help="初始化 RAG 知识库索引",
    )
    args = parser.parse_args()

    if args.init_rag:
        # asyncio.run: 创建事件循环并运行协程直到完成
        asyncio.run(init_rag())
    else:
        asyncio.run(run_agent())


if __name__ == "__main__":
    main()
