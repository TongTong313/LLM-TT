import requests
from bs4 import BeautifulSoup
from typing import Optional
import re


async def get_web_content(url: str, timeout: Optional[int] = 30) -> str:
    """获取指定URL的网页全部内容
    
    Args:
        url (str): 要获取内容的网页URL
        timeout (int, optional): 请求超时时间，默认30秒
        
    Returns:
        str: 网页的完整内容，包括标题、正文等
    """
    try:
        # 设置请求头，模拟浏览器访问
        headers = {
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # 发送HTTP请求
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # 检查请求是否成功

        # 设置编码
        response.encoding = response.apparent_encoding

        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 移除script和style标签
        for script in soup(["script", "style"]):
            script.decompose()

        # 获取网页标题
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "无标题"

        # 获取正文内容
        # 尝试多种方式获取主要内容
        content = ""

        # 方法1：获取body内容
        body = soup.find('body')
        if body:
            content = body.get_text()

        # 方法2：如果没有body，获取html内容
        if not content:
            content = soup.get_text()

        # 清理文本内容
        content = re.sub(r'\s+', ' ', content)  # 将多个空白字符替换为单个空格
        content = re.sub(r'\n\s*\n', '\n', content)  # 移除多余的空行
        content = content.strip()

        # 限制内容长度，避免返回过长的内容
        max_length = 10000
        if len(content) > max_length:
            content = content[:max_length] + "...(内容已截断)"

        # 格式化输出
        result = f"""
网页内容获取成功：

URL：{url}
标题：{title_text}

内容：
{content}
        """

        return result.strip()

    except requests.exceptions.RequestException as e:
        return f"获取网页内容失败：{str(e)}"
    except Exception as e:
        return f"解析网页内容时出错：{str(e)}"


async def get_web_content_simple(url: str) -> str:
    """简化版本的网页内容获取函数
    
    Args:
        url (str): 要获取内容的网页URL
        
    Returns:
        str: 网页的文本内容
    """
    try:
        headers = {
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, 'html.parser')

        # 移除script和style
        for script in soup(["script", "style"]):
            script.decompose()

        # 获取文本内容
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()

        # 限制长度
        if len(text) > 5000:
            text = text[:5000] + "..."

        return text

    except Exception as e:
        return f"获取网页内容失败：{str(e)}"
