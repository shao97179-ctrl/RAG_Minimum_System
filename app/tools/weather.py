"""
天气工具模块。

调用 OpenWeather API 获取指定城市的当前天气信息。

LLM 只能看到:
    get_weather(city: str) -> str

API Key 等内部细节不暴露给 LLM。
"""

import logging
from typing import Any

import httpx

from app.config import OPENWEATHER_API_KEY, OPENWEATHER_API_URL

logger = logging.getLogger(__name__)

# 请求超时（秒）
REQUEST_TIMEOUT = 10.0


def get_weather(city: str) -> str:
    """
    获取指定城市的当前天气。

    参数:
        city: 城市名称，如 "北京"、"Tokyo"、"London"

    返回:
        天气信息字符串，包含温度、体感温度、天气描述、湿度、风速
    """
    if not OPENWEATHER_API_KEY:
        return "错误：OpenWeather API Key 未配置，无法获取天气信息。"

    logger.info("Getting weather for city: %s", city)

    # OpenWeather API 参数
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",       # 使用摄氏度
        "lang": "zh_cn",         # 返回中文天气描述
    }

    try:
        response = httpx.get(
            OPENWEATHER_API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        logger.error("OpenWeather API timeout for city: %s", city)
        return f"错误：获取 {city} 的天气超时，请稍后重试。"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.error("City not found: %s", city)
            return f"错误：找不到城市 '{city}'，请检查城市名称是否正确。"
        elif e.response.status_code == 401:
            logger.error("OpenWeather API Key invalid")
            return "错误：OpenWeather API Key 无效，请检查配置。"
        else:
            logger.error("OpenWeather API error: %d", e.response.status_code)
            return f"错误：天气 API 返回错误 ({e.response.status_code})。"
    except Exception as e:
        logger.error("OpenWeather request failed: %s", e)
        return f"错误：获取天气失败 - {e}"

    # 解析响应
    try:
        data: dict[str, Any] = response.json()
        return _format_weather(data)
    except Exception as e:
        logger.error("Failed to parse weather data: %s", e)
        return "错误：天气数据解析失败。"


def _format_weather(data: dict[str, Any]) -> str:
    """
    将 OpenWeather API 响应格式化为可读字符串。

    这是内部函数，不暴露给 LLM。
    """
    city_name = data.get("name", "未知")
    country = data.get("sys", {}).get("country", "")

    main = data.get("main", {})
    temp = main.get("temp", "N/A")
    feels_like = main.get("feels_like", "N/A")
    humidity = main.get("humidity", "N/A")

    weather_list = data.get("weather", [])
    description = weather_list[0].get("description", "N/A") if weather_list else "N/A"

    wind = data.get("wind", {})
    wind_speed = wind.get("speed", "N/A")

    result = (
        f"城市：{city_name}, {country}\n"
        f"温度：{temp}°C\n"
        f"体感温度：{feels_like}°C\n"
        f"天气：{description}\n"
        f"湿度：{humidity}%\n"
        f"风速：{wind_speed} m/s"
    )

    logger.info("Weather data formatted for %s", city_name)
    return result
