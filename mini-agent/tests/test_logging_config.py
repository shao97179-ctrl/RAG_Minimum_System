"""
日志配置模块单元测试。

测试清理逻辑和文件名解析，不测试 setup_logging 的 handler 挂载
（那会污染整个测试进程的全局 logger 配置）。
"""

from datetime import datetime
from pathlib import Path

from app.logging_config import (
    FILENAME_PREFIX,
    _make_log_filename,
    _parse_date_from_filename,
    clean_old_logs,
)


def make_log_file(directory: Path, filename: str) -> Path:
    """在目录下创建一个假的日志文件。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text("dummy log content", encoding="utf-8")
    return path


class TestMakeLogFilename:
    """文件名生成测试。"""

    def test_contains_date(self) -> None:
        filename = _make_log_filename()
        today = datetime.now().strftime("%Y-%m-%d")
        assert filename == f"{FILENAME_PREFIX}_{today}.log"


class TestParseDate:
    """文件名日期解析测试。"""

    def test_valid_filename(self) -> None:
        result = _parse_date_from_filename("mini-agent_2026-09-27.log")
        assert result == datetime(2026, 9, 27)

    def test_other_filename(self) -> None:
        assert _parse_date_from_filename("notes.txt") is None

    def test_wrong_prefix(self) -> None:
        assert _parse_date_from_filename("other_2026-09-27.log") is None

    def test_illegal_date(self) -> None:
        # 月份 13 不存在，解析应返回 None 而不是抛异常
        assert _parse_date_from_filename("mini-agent_2026-13-99.log") is None


class TestCleanOldLogs:
    """过期日志清理测试。"""

    def test_removes_expired_keeps_fresh(self, tmp_path: Path) -> None:
        """过期的删除，未过期的保留。"""
        old_file = make_log_file(tmp_path, "mini-agent_2020-01-01.log")
        fresh_file = make_log_file(
            tmp_path, f"mini-agent_{datetime.now().strftime('%Y-%m-%d')}.log"
        )

        removed = clean_old_logs(retention_days=7, logs_dir=tmp_path)

        assert removed == 1
        assert not old_file.exists()
        assert fresh_file.exists()

    def test_keeps_non_log_files(self, tmp_path: Path) -> None:
        """非日志命名的文件不动。"""
        other = make_log_file(tmp_path, "important.txt")
        removed = clean_old_logs(retention_days=7, logs_dir=tmp_path)
        assert removed == 0
        assert other.exists()

    def test_retention_zero_deletes_old_days(self, tmp_path: Path) -> None:
        """保留 0 天 = 今天之前的都删，今天的保留。"""
        past = make_log_file(tmp_path, "mini-agent_2020-01-01.log")
        today = make_log_file(
            tmp_path, f"mini-agent_{datetime.now().strftime('%Y-%m-%d')}.log"
        )

        removed = clean_old_logs(retention_days=0, logs_dir=tmp_path)

        assert removed == 1
        assert not past.exists()
        assert today.exists()

    def test_missing_directory(self, tmp_path: Path) -> None:
        """目录不存在时返回 0，不报错。"""
        removed = clean_old_logs(retention_days=7, logs_dir=tmp_path / "no_such_dir")
        assert removed == 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        tmp_path.mkdir(exist_ok=True)
        assert clean_old_logs(retention_days=7, logs_dir=tmp_path) == 0
