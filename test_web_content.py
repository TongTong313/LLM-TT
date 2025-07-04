#!/usr/bin/env python3
"""
测试网页内容获取功能
"""

import asyncio
import sys
import os

# 添加src目录到Python路径
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from graphagent.tool.web_content import get_web_content, get_web_content_simple


async def test_web_content():
    """测试网页内容获取功能"""

    # 测试URL
    test_url = "https://www.python.org"

    print("=" * 50)
    print("测试网页内容获取功能")
    print("=" * 50)

    print(f"\n正在获取网页内容：{test_url}")
    print("-" * 30)

    # 测试完整版本
    print("1. 完整版本结果：")
    result = await get_web_content(test_url)
    print(result)

    print("\n" + "=" * 50)

    # 测试简化版本
    print("2. 简化版本结果：")
    simple_result = await get_web_content_simple(test_url)
    print(simple_result)


async def test_custom_url():
    """测试自定义URL"""
    print("\n" + "=" * 50)
    print("测试自定义URL")
    print("=" * 50)

    # 让用户输入URL
    url = input("请输入要获取内容的网页URL（直接回车使用默认URL）：").strip()
    if not url:
        url = "https://httpbin.org/html"

    print(f"\n正在获取网页内容：{url}")
    print("-" * 30)

    try:
        result = await get_web_content(url)
        print(result)
    except Exception as e:
        print(f"获取失败：{e}")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_web_content())

    # 询问是否测试自定义URL
    choice = input("\n是否要测试自定义URL？(y/n): ").strip().lower()
    if choice in ['y', 'yes', '是']:
        asyncio.run(test_custom_url())
