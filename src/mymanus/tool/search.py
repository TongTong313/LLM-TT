from baidusearch.baidusearch import search
import json


async def baidu_search(query: str, num_results: int = 3) -> str:
    """百度搜索工具

    Args:
        query (str): 搜索关键词
        num_results (int, optional): 搜索结果数量，默认3条.

    Returns:
        str: 搜索结果
    """
    results = search(query, num_results=num_results)
    # 转换为json
    results = json.dumps(results, ensure_ascii=False)
    return results
