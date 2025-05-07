from datetime import datetime


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
