"""
Tool Dispatcher 单元测试。
"""

from app.agent.tool_dispatcher import dispatch_tool, parse_tool_args, build_tool_message


class TestParseToolArgs:
    """参数解析测试。"""

    def test_valid_json(self) -> None:
        result = parse_tool_args('{"city": "北京"}')
        assert result == {"city": "北京"}

    def test_empty_string(self) -> None:
        result = parse_tool_args("")
        assert result == {}

    def test_invalid_json(self) -> None:
        result = parse_tool_args("not json")
        assert result == {}


class TestBuildToolMessage:
    """工具消息构造测试。"""

    def test_basic(self) -> None:
        msg = build_tool_message("call_123", "get_weather", "晴天")
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"
        assert msg["name"] == "get_weather"
        assert msg["content"] == "晴天"


class TestDispatchTool:
    """工具分发测试。"""

    def test_calculate(self) -> None:
        result = dispatch_tool("calculate", {"expression": "2 + 3"})
        assert "5" in result

    def test_unknown_tool(self) -> None:
        try:
            dispatch_tool("nonexistent_tool", {})
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "未知工具" in str(e)


class TestHistoryInjection:
    """history 参数自动注入测试。"""

    def test_injects_history_when_declared(self, monkeypatch) -> None:
        """工具函数声明了 history 参数时，自动注入对话历史。"""
        from app.agent import tool_dispatcher

        captured: dict[str, object] = {}

        def fake_tool(query: str, history=None) -> str:
            captured["query"] = query
            captured["history"] = history
            return "ok"

        monkeypatch.setitem(tool_dispatcher.TOOLS_MAP, "fake_tool", fake_tool)

        history = [{"role": "user", "content": "我在学 ChromaDB"}]
        result = dispatch_tool("fake_tool", {"query": "它是什么"}, history=history)

        assert result == "ok"
        assert captured["query"] == "它是什么"
        assert captured["history"] == history

    def test_no_injection_when_not_declared(self) -> None:
        """工具函数没有声明 history 参数时，不注入多余参数。"""
        # calculate 没有 history 参数；如果误注入会走 TypeError 分支，
        # 返回"参数错误"而不是计算结果
        result = dispatch_tool(
            "calculate", {"expression": "1 + 1"}, history=[{"role": "user"}]
        )
        assert "2" in result

    def test_history_none_by_default(self, monkeypatch) -> None:
        """不传 history 时，声明了该参数的工具收到 None。"""
        from app.agent import tool_dispatcher

        captured: dict[str, object] = {}

        def fake_tool(query: str, history=None) -> str:
            captured["history"] = history
            return "ok"

        monkeypatch.setitem(tool_dispatcher.TOOLS_MAP, "fake_tool", fake_tool)

        dispatch_tool("fake_tool", {"query": "q"})

        assert captured["history"] is None
