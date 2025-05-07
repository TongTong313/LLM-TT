async def add(*args: float) -> float:
    """对任意个数的数字进行加法运算
    
    Args:
        *args (float): 加数列表

    Returns:
        float: 和
    """

    # 转换为字符串
    return f"相加的结果是：{sum(args)}"


if __name__ == "__main__":
    print(add(1, 2, 3, 4, 5))
