# 测试大模型function calling
from openai import OpenAI, AsyncOpenAI, AsyncStream
import os
import asyncio
import random
from datetime import datetime
import json
from typing import Optional, List, Dict
import requests
from bs4 import BeautifulSoup
from baidusearch.baidusearch import search

client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


# 模拟天气查询工具。返回结果示例：“北京今天是雨天。”
async def get_current_weather(location: str) -> str:
    # 定义备选的天气条件列表
    weather_conditions = ["晴天", "多云", "雨天"]
    # 随机选择一个天气条件
    random_weather = random.choice(weather_conditions)
    # 返回格式化的天气信息
    return f"{location}今天是{random_weather}。"


async def baidu_search(query: str, num_results: int = 3) -> str:
    """百度搜索工具

    Args:
        query (str): 搜索关键词

    Returns:
        str: 搜索结果
    """
    results = search(query, num_results=num_results)
    # 转换为json
    results = json.dumps(results, ensure_ascii=False)
    return results


# 查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“
async def get_current_time() -> str:
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 格式化当前日期和时间
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # 返回格式化后的当前时间
    return f"当前时间：{formatted_time}。"


tools = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "当你想知道现在的时间时非常有用。",
    }
}, {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "当你想查询指定城市的天气时非常有用。",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市或县区，比如北京市、杭州市、余杭区等。",
                }
            },
            "required": ["location"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "baidu_search",
        "description": "对于用户提出的问题，如果需要使用搜索引擎查询，请使用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                },
                "num_results": {
                    "type": "integer",
                    "description": "搜索结果数量",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
}]


# 异步任务
async def function_calling(query: str) -> tuple[str, str, str]:
    """函数调用函数，采用流式输出，兼容普通问答

    Args:
        query (str): 用户输入的query

    Returns:
        tuple[str, str, str]: 工具名称、工具入参、回答
    """

    origin_messages = [{
        "role": "system",
        "content": "你是一个AI助手，请根据用户的问题给出回答，可以采用工具调用帮助回答问题"
    }, {
        "role": "user",
        "content": query
    }]

    response = await client.chat.completions.create(model="qwen-plus",
                                                    messages=origin_messages,
                                                    tools=tools,
                                                    tool_choice="auto",
                                                    stream=True)
    function_name = ""
    function_arguments = ""
    response_content = ""
    fun_id = None
    first_chunk = True
    # 处理流式输出：当成标准模板背诵！
    async for chunk in response:
        if chunk.choices[0].delta.tool_calls:
            if first_chunk:  # 第一个chunk提取工具名称，同时开始累积函数入参
                function_name = chunk.choices[0].delta.tool_calls[
                    0].function.name
                function_arguments += chunk.choices[0].delta.tool_calls[
                    0].function.arguments
                fun_id = chunk.choices[0].delta.tool_calls[0].id
                first_chunk = False
            else:
                if chunk.choices[0].delta.tool_calls[0].function.arguments:
                    function_arguments += chunk.choices[0].delta.tool_calls[
                        0].function.arguments
        else:
            # 不是函数调用，正常回答
            if chunk.choices[0].delta.content:
                response_content += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="", flush=True)

    # 返回工具名称、工具入参、回答
    return function_name, function_arguments, fun_id, origin_messages, response_content


# 输出信息都是字符串，需要根据字符串信息执行函数
# 需要字符串到函数名称的映射 -> 使用字典实现
tool_mapping = {
    "get_current_time": get_current_time,
    "get_current_weather": get_current_weather,
    "baidu_search": baidu_search
}

assistant_messages_template = {
    "content":
    "",
    "refusal":
    None,
    "role":
    "assistant",
    "audio":
    None,
    "function_call":
    None,
    "tool_calls": [{
        "id": "call_xxx",
        "function": {
            "arguments": "",
            "name": "",
        },
        "type": "function",
        "index": 0,
    }],
}


async def main():
    # 用户问题
    query = input("请输入问题：")

    # 1. 大模型根据用户问题以字符串形式返回调用工具名称和入参
    function_name, function_arguments, fun_id, origin_messages, response_content = await function_calling(
        query)

    if function_name:
        print(
            f"执行函数调用：工具名称：{function_name}，工具参数：{function_arguments}，工具调用id：{fun_id}"
        )

    # 函数执行过程
    # 2. 根据函数映射获取函数实体
    function = tool_mapping[function_name]
    # 3. 解析函数入参（将字符串转换为字典）
    function_arguments_dict = json.loads(function_arguments)
    # 4. 执行函数
    function_result = await function(**function_arguments_dict)
    # 5. 打印函数结果
    print(function_result)

    # 将函数执行结果告诉大模型，让大模型能够根据函数执行结果得到更准确的答案
    # 6. 更新messages
    # 6.1 依据assistant_messages_template生成assistant_messages
    assistant_messages = assistant_messages_template.copy()
    assistant_messages["tool_calls"][0]["id"] = fun_id
    assistant_messages["tool_calls"][0]["function"].update({
        'arguments':
        function_arguments,
        'name':
        function_name
    })

    # # 6.2 将assistant_messages添加到origin_messages中
    origin_messages.append(assistant_messages)

    # 6.3 将函数的输出信息添加到origin_messages中
    origin_messages.append({
        'role': 'tool',
        'content': function_result,
        'tool_call_id': fun_id
    })

    # 6.4 将拼接后的messages发送给大模型，现在包括原始messages、assistant_messages、function_result
    print(origin_messages)
    function_name, function_arguments, fun_id, origin_messages, response_content = await function_calling(
        query, origin_messages)
    if function_name:
        print(
            f"执行函数调用：工具名称：{function_name}，工具参数：{function_arguments}，工具调用id：{fun_id}"
        )


if __name__ == "__main__":
    asyncio.run(main())
    # search_results = asyncio.run(baidu_search("杭州市"))
    # print(search_results[0])
