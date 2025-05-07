async def terminate():
    """这是一个特殊的工具，这个工具的作用就是一旦调用，就意味着智能体已经解决了所有的问题，返回一个固定回答
    
    Returns:
        str: 固定回答
    """
    return f"开始总结用户结果"
