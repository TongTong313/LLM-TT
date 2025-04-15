from baidusearch.baidusearch import search
import json


async def baidu_search(query: str, num_results: int = 10) -> str:
    """百度搜索工具

    Args:
        query (str): 搜索关键词
        num_results (int, optional): 搜索结果数量，默认10条.

    Returns:
        str: 格式化的搜索结果
    """
    results = search(query, num_results=num_results)

    # 格式化搜索结果
    formatted_results = []
    for i, result in enumerate(results, 1):
        title = result.get('title', '无标题')
        abstract = result.get('abstract', '无摘要')
        url = result.get('url', '无链接')

        # 清理摘要中的特殊字符和多余空格
        abstract = abstract.replace('\n', ' ').replace('\ue62b', '').replace(
            '\ue680', '').replace('\ue67d', '').strip()

        # 构建格式化的结果
        formatted_result = f"""
                            第{i}条搜索结果：
                            标题：{title}
                            链接：{url}
                            摘要：{abstract}
                            """
        formatted_results.append(formatted_result)

    # 将所有结果合并成一个字符串
    final_result = "搜索结果：\n" + "\n".join(formatted_results)
    return final_result
