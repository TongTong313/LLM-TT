# 测试大模型function calling
from openai import OpenAI, AsyncOpenAI
import os
import asyncio

client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


# 异步任务
async def ask(query):
    print(f"正在提问: {query}")
    messages = [{
        "role": "system",
        "content": "你是一个AI助手，请根据用户的问题给出回答。"
    }, {
        "role": "user",
        "content": query
    }]
    response = await client.chat.completions.create(model="qwen-plus",
                                                    messages=messages,
                                                    stream=True)
    full_response = ""
    async for chunk in response:
        if chunk.choices:
            full_response += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="", flush=True)

    # return full_response


async def main():
    querys = [
        "你好，请帮我写一个冒泡排序的代码",
        "你好，请帮我写一个快速排序的代码",
    ]
    tasks = [ask(query) for query in querys]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
