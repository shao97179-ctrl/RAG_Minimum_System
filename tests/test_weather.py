"""
Weather 工具单元测试。

注意：真实 API 测试需要 OPENWEATHER_API_KEY。
这里测试基本逻辑和错误处理。
"""

from app.tools.weather import get_weather, _format_weather


class TestFormatWeather:
    """天气数据格式化测试。"""

    def test_format_basic(self) -> None:
        """测试基本天气数据格式化。"""
        data = {
            "name": "Beijing",
            "sys": {"country": "CN"},
            "main": {
                "temp": 25.5,
                "feels_like": 27.0,
                "humidity": 60,
            },
            "weather": [{"description": "晴"}],
            "wind": {"speed": 3.5},
        }
        result = _format_weather(data)
        assert "Beijing" in result
        assert "25.5" in result
        assert "27.0" in result
        assert "晴" in result
        assert "60" in result
        assert "3.5" in result


class TestGetWeather:
    """get_weather 函数测试。"""

    def test_empty_api_key(self) -> None:
        """API Key 未配置时返回错误。"""
        # 临时清空 API Key
        import app.config as config
        original = config.OPENWEATHER_API_KEY
        config.OPENWEATHER_API_KEY = ""

        try:
            result = get_weather("北京")
            assert "错误" in result
        finally:
            config.OPENWEATHER_API_KEY = original
