from datetime import datetime
import logging


logger = logging.getLogger(__name__)
async def get_time(format_type: str | None = None) -> str:
    """
    获取当前日期/时间。

    Args:
        format: 可选。
            - "date"：只返回日期，格式 YYYY-MM-DD
            - "time"：只返回时间，格式 HH:MM:SS
            - "datetime" 或 None：返回完整日期时间，格式 YYYY-MM-DD HH:MM:SS
    """
    now = datetime.now()
    fmt = format_type or "datetime"

    logger.info("Getting current time, format: %s", fmt)

    if fmt == "date":
        return now.strftime("%Y-%m-%d")
    if fmt == "time":
        return now.strftime("%H:%M:%S")
    if fmt == "datetime":
        return now.strftime("%Y-%m-%d %H:%M:%S")

    # 对未知格式回退到完整时间，避免工具调用因模型给错参数而直接失败
    logger.warning("时间格式未知")
    return now.strftime("%Y-%m-%d %H:%M:%S")
    