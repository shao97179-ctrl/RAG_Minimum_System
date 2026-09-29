"""
日志配置模块。

职责:
1. 把日志写入 logs/ 目录下的文件，不在终端输出（保持对话界面干净）
2. 文件名带明确日期，一天一个文件，便于按日期查找
3. 每次启动自动清理超过保留期（LOG_RETENTION_DAYS 天）的旧日志

日志文件命名: mini-agent_2026-09-27.log
同一天多次运行共用同一个文件（追加写入），每次启动写入一条分隔线。

使用方式（只在程序入口 main.py 调用一次）:

    from app.logging_config import setup_logging
    log_file = setup_logging()
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

from app.config import LOG_LEVEL, LOG_RETENTION_DAYS, LOGS_DIR

# 日志文件名前缀和日期格式
FILENAME_PREFIX = "mini-agent"
FILENAME_DATE_FORMAT = "%Y-%m-%d"

# 用于从文件名解析日期的正则（只匹配本项目命名的日志文件）
_FILENAME_PATTERN = re.compile(
    rf"^{re.escape(FILENAME_PREFIX)}_(\d{{4}}-\d{{2}}-\d{{2}})\.log$"
)


def setup_logging() -> Path:
    """
    初始化日志系统（只写文件，不在终端输出），返回本次日志文件路径。

    做了三件事:
    1. 创建 logs/ 目录
    2. 清理超过保留期的旧日志
    3. 给根 logger 挂上文件 handler
    """
    logs_dir = LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 先清理过期日志，再把清理结果写进本次日志
    removed_count = clean_old_logs()

    log_file = logs_dir / _make_log_filename()

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 只挂 FileHandler，不挂 StreamHandler → 终端不再显示日志
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # 同一天多次运行时，用分隔线区分每次运行
    separator = (
        f"\n{'=' * 60}\n"
        f"运行开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 60}"
    )
    logging.getLogger(__name__).info(separator)

    logger = logging.getLogger(__name__)
    logger.info("日志文件: %s（保留 %d 天）", log_file, LOG_RETENTION_DAYS)
    if removed_count:
        logger.info("已清理 %d 个过期日志文件", removed_count)

    return log_file


def clean_old_logs(
    retention_days: int = LOG_RETENTION_DAYS,
    logs_dir: Path | None = None,
) -> int:
    """
    删除超过保留期的日志文件，返回删除的文件数。

    文件年龄根据文件名中的日期判断（而不是文件修改时间），
    这样即使文件被复制或移动过，判断依然可靠。

    删除失败的单个文件会被跳过，不影响其他文件和主流程。
    """
    if logs_dir is None:
        logs_dir = LOGS_DIR

    if not logs_dir.exists():
        return 0

    # 按"天"比较（解析出的文件日期不含时分秒），
    # 因此 retention_days=0 表示"只保留今天的日志"
    cutoff_date = (datetime.now() - timedelta(days=retention_days)).date()
    removed = 0

    for path in sorted(logs_dir.glob(f"{FILENAME_PREFIX}_*.log")):
        file_date = _parse_date_from_filename(path.name)
        if file_date is None or file_date.date() >= cutoff_date:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue  # 单个文件删除失败（如被占用），跳过

    return removed


def _make_log_filename() -> str:
    """生成今天的日志文件名: mini-agent_YYYY-MM-DD.log"""
    return f"{FILENAME_PREFIX}_{datetime.now().strftime(FILENAME_DATE_FORMAT)}.log"


def _parse_date_from_filename(filename: str) -> datetime | None:
    """从日志文件名解析日期，命名不符合规则时返回 None。"""
    match = _FILENAME_PATTERN.match(filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), FILENAME_DATE_FORMAT)
    except ValueError:
        return None
