from datetime import datetime
from typing import Optional, Literal
import random


async def get_current_time() -> str:
    """查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“

    Returns:
        str: 当前时间
    """

    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 格式化当前日期和时间
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # 返回格式化后的当前时间
    return f"当前时间：{formatted_time}。"


async def get_current_weather(
        location: str,
        units: Optional[Literal["celsius", "fahrenheit"]] = "celsius") -> str:
    """查询当前天气的工具。返回结果示例：“北京今天是雨天。”

    Args:
        location (str): 城市名称
        units (Optional[Literal["celsius", "fahrenheit"]], optional): 温度单位. 
        默认是摄氏度，可选摄氏度或华氏度

    Returns:
        str: 当前天气
    """

    # 随机返回一个天气，做接口调试用
    weathers = ["晴天", "多云", "阴天", "小雨", "大雨", "雷阵雨", "冰雹", "大雪", "雾霾"]
    weather = random.choice(weathers)
    return f"{location}今天是{weather}。"
