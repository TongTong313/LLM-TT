# 实现MCP客户端，参考https://modelcontextprotocol.io/quickstart/client
from typing import Optional
# from anthropic import Anthropic
from contextlib import AsyncExitStack
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
from openai import OpenAI, AsyncOpenAI
import os


class MCPClient:

    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.qwen = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        # self.anthropic = Anthropic()

    async def connect_to_mcp_server(self, server_script_path: str):
        """连接到一个MCP服务端

        Args:
            server_script_path (str): Path to the server script (.py or .js)
        """

        # 判断文件格式
        is_python = server_script_path.endswith('.py')

        if not is_python:
            raise ValueError("Only Python scripts are supported")

        command = "python" if is_python else None
        # StdioServerParameters类就是封装一句命令行的命令，比如python server.py
        server_params = StdioServerParameters(command=command,
                                              args=[server_script_path],
                                              env=None)

        # 使用AsyncExitStack管理异步资源，建立与该进程的标准输入输出通信管道
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))  # 执行命令起服务，并和它连接上
        self.stdio, self.write = stdio_transport  # 获取对服务端进程的读取和写入接口
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write))

        # 整个过程的数据流如下：
        # 客户端代码（比如发送消息） -> self.write -> 服务端进程的标准输入
        # 服务端进程的标准输出 -> self.stdio -> 客户端代码（比如接收消息）

        # 初始化会话，保障客户端<->服务端通信正常
        await self.session.initialize()

        # 客户端向服务端发送一个请求获取工具列表，服务端返回工具列表
        response = await self.session.list_tools()
        tools = response.tools

        print('\n已和MCP服务端连接完成，工具包含：', [tool.name for tool in tools])

    async def process_query(self, query: str):
        """处理用户需求，结合大模型和MCP Server的工具调用
        客户端向服务端发送一个query请求，服务端返回执行结果
        """
        messages = [{"role": "user", "content": query}]

        # 这里客户端向服务器发送一个请求获取工具列表，服务端返回工具列表
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # 初始化OpenAI的API
        response = await self.qwen.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
            stream=True)

        # 获取流式response的调用工具信息，然后拼接
        async for chunk in response:
            if chunk.choices[0].message.tool_calls:
