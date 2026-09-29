"""
上下文管理模块单元测试。

通过注入假的 summarize_fn，测试压缩逻辑本身，不需要调用真实 LLM API。
"""

from typing import Any

from app.agent.context_manager import (
    compress_history,
    estimate_history_tokens,
    estimate_message_tokens,
    estimate_tokens,
    split_into_turns,
)


def make_turn(user_text: str, assistant_text: str) -> list[dict[str, Any]]:
    """构造一轮简单对话（user + assistant）。"""
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def make_long_history(n_turns: int) -> list[dict[str, Any]]:
    """构造 n_turns 轮较长的对话，用于触发压缩。"""
    history: list[dict[str, Any]] = []
    for i in range(n_turns):
        history.extend(make_turn(f"这是一个比较长的问题编号{i}" * 30,
                                 f"这是一个比较长的回答编号{i}" * 30))
    return history


class TestEstimateTokens:
    """Token 估算测试。"""

    def test_chinese_chars(self) -> None:
        # 每个汉字按 1 token 计
        assert estimate_tokens("你好世界") == 4

    def test_other_chars(self) -> None:
        # 其他字符 4 个 ≈ 1 token
        assert estimate_tokens("abcd") == 1

    def test_mixed(self) -> None:
        # 4 个汉字 + 4 个英文字符 = 4 + 1
        assert estimate_tokens("你好世界abcd") == 5

    def test_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_message_with_tool_calls(self) -> None:
        """assistant 的 tool_calls 也应计入 token。"""
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "北京"}',
                    },
                }
            ],
        }
        tokens = estimate_message_tokens(msg)
        assert tokens > 0


class TestSplitIntoTurns:
    """历史分组测试。"""

    def test_basic(self) -> None:
        history = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
            {"role": "user", "content": "问题2"},
            {"role": "assistant", "content": "回答2"},
        ]
        turns = split_into_turns(history)
        assert len(turns) == 2
        assert turns[0][0]["content"] == "问题1"
        assert turns[1][0]["content"] == "问题2"

    def test_tool_group_not_split(self) -> None:
        """assistant(tool_calls) + tool 消息组必须留在同一轮。"""
        history = [
            {"role": "user", "content": "北京天气"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "晴"},
            {"role": "assistant", "content": "北京晴"},
            {"role": "user", "content": "谢谢"},
            {"role": "assistant", "content": "不客气"},
        ]
        turns = split_into_turns(history)
        assert len(turns) == 2
        assert len(turns[0]) == 4  # 工具消息组完整保留在第一轮

    def test_empty(self) -> None:
        assert split_into_turns([]) == []


class TestCompressHistory:
    """压缩逻辑测试（注入假摘要函数，不调用真实 API）。"""

    def test_under_limit_no_change(self) -> None:
        """未超限时不应压缩。"""
        history = make_turn("你好", "你好！")

        def should_not_be_called(turns: list) -> str:
            raise AssertionError("未超限时不应调用摘要函数")

        assert compress_history(history, summarize_fn=should_not_be_called) is False
        assert len(history) == 2

    def test_over_limit_compresses_with_summary(self) -> None:
        """超限时压缩：旧轮次变摘要，最近轮次保留。"""
        history = make_long_history(15)  # 约 9000 tokens > 6000

        def fake_summarize(turns: list) -> str:
            return "这是假摘要"

        assert compress_history(history, summarize_fn=fake_summarize) is True

        # 摘要消息应在开头
        assert history[0]["role"] == "system"
        assert "假摘要" in history[0]["content"]
        # 压缩后消息数应明显减少（30 条 -> 摘要1条 + 最近几轮）
        assert len(history) < 30
        # 最近的对话必须保留（最后一轮内容还在）
        assert any("编号14" in str(m.get("content")) for m in history)

    def test_summary_failure_falls_back_to_truncation(self) -> None:
        """摘要失败（返回 None）时，退化为丢弃旧轮次。"""
        history = make_long_history(15)

        result = compress_history(history, summarize_fn=lambda turns: None)

        assert result is True
        # 没有摘要消息，只有最近轮次
        assert all(m["role"] != "system" for m in history)
        assert len(history) < 30
        assert any("编号14" in str(m.get("content")) for m in history)

    def test_tool_message_not_orphaned(self) -> None:
        """压缩后 tool 消息前面必须紧跟带 tool_calls 的 assistant 消息。"""
        history = make_long_history(3)
        # 在中间插入一轮带工具调用的对话
        tool_turn = [
            {"role": "user", "content": "北京天气" * 50},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "北京"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "晴 " * 100},
            {"role": "assistant", "content": "北京晴天 " * 50},
        ]
        history.extend(tool_turn)
        history.extend(make_long_history(12))

        assert compress_history(history, summarize_fn=lambda t: "摘要") is True

        # 校验：每条 tool 消息的前一条必须是带 tool_calls 的 assistant
        for i, msg in enumerate(history):
            if msg["role"] == "tool":
                prev = history[i - 1]
                assert prev["role"] == "assistant"
                assert prev.get("tool_calls"), "tool 消息被孤立了！"

    def test_history_object_identity_preserved(self) -> None:
        """压缩必须就地修改，外部持有的引用仍然有效。"""
        history = make_long_history(15)
        outer_ref = history

        compress_history(history, summarize_fn=lambda t: "摘要")

        assert outer_ref is history  # 同一个列表对象
        assert len(outer_ref) < 30
