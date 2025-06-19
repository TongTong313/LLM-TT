from typing import List


async def add(numbers: List[float]) -> float:
    """对任意个数的数字进行加法运算
    
    Args:
        numbers (List[float]): 加数列表

    Returns:
        float: 和
    """

    # 转换为字符串
    return f"相加的结果是：{sum(numbers)}"


if __name__ == "__main__":
    import asyncio
    result = asyncio.run(add([1, 2, 3, 4, 5]))
    print(result)
