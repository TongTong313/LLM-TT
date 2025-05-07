# 实现MCP客户端，参考https://modelcontextprotocol.io/quickstart/client
from typing import Optional, List
import json
from contextlib import AsyncExitStack
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
from openai import OpenAI, AsyncOpenAI
from .mcp_adapter import BaseMCPAdapter


# 写一个adapter函数，将MCP tool中的字段转换为openai接口需要的tool_schema
class MCPClient:

    def __init__(self, api_key: str, base_url: str, adapter: BaseMCPAdapter):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.adapter = adapter
        # self.anthropic = Anthropic()

    async def connect_to_mcp_server_stdio(self, server_script_path: str):
        """连接到一个MCP服务端 stdio模式

        Args:
            server_script_path (str): Path to the server script (.py or .js)
        """

        # 判断文件格式
        is_python = server_script_path.endswith('.py')

        # 目前仅支持python脚本
        if not is_python:
            raise ValueError("仅支持python脚本")

        command = "python" if is_python else None
        # StdioServerParameters类就是封装一句命令行的命令（类），比如python server.py
        server_params = StdioServerParameters(command=command,
                                              args=[server_script_path],
                                              env=None)

        # 官方提供，建议背板：使用AsyncExitStack管理异步资源，建立与该进程的标准输入输出通信管道
        # 将stdio客户端添加到异步上下文栈中，这个过程会：
        # 1.启动MCP服务器进程
        # 2.建立与该进程的标准输入输出通信管道
        # 3.返回一个包含读写接口的传输对象
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))  # 执行命令起服务，并和它连接上
        self.stdio, self.write = stdio_transport  # 获取对服务端进程的读取和写入接口
        # 将客户端会话添加到异步上下文栈中，这个过程会：
        # 1.初始化客户端会话
        # 2.建立与服务器的会话连接
        # 3.设置必要的会话参数和状态
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

    async def connect_to_mcp_server_sse(self, server_url: str):
        """使用SSE模式连接到MCP服务端(测试中……)

        Args:
            server_url (str): MCP服务端的SSE URL
        """
        from mcp.client.sse import sse_client

        # 使用AsyncExitStack管理异步资源，建立与SSE服务器的连接
        streams_context = sse_client(url=server_url)
        streams = await self.exit_stack.enter_async_context(streams_context)

        # 创建会话
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(*streams))

        # 初始化会话，保障客户端<->服务端通信正常
        await self.session.initialize()

        # 获取工具列表
        response = await self.session.list_tools()
        tools = response.tools

        print('\n已和MCP服务端SSE连接完成，工具包含：', [tool.name for tool in tools])

    async def process_query(self, query: str):
        """处理用户需求，结合大模型和MCP Server的工具调用
        客户端向服务端发送一个query请求，服务端返回执行结果
        """
        messages = [{"role": "user", "content": query}]

        # 这里客户端向服务器发送一个请求获取工具列表，服务端返回工具列表
        response = await self.session.list_tools()
        # 这里要调用adapter的convert_to_tool_schema方法，将工具列表转换为openai接口需要的tool_schema
        available_tools = self.adapter.convert_to_tool_schema(response.tools)

        # available_tools = [{
        #     "name": tool.name,
        #     "description": tool.description,
        #     "input_schema": tool.inputSchema
        # } for tool in response.tools]

        # 初始化OpenAI的API
        response = await self.llm.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
            stream=True)

        # 获取流式response的调用工具信息，然后拼接
        collected_content = []
        collected_tool_calls = []
        current_tool_call = None

        async for chunk in response:
            # 处理内容部分
            if chunk.choices[0].delta.content:
                chunk_content = chunk.choices[0].delta.content
                collected_content.append(chunk_content)
                print(chunk_content, end="", flush=True)

            # 处理工具调用部分：工具调用部分第一个返回的流式输出对象可以获得工具名称（name），但工具的入参需要拼接
            if chunk.choices[0].delta.tool_calls:
                for tool_call in chunk.choices[0].delta.tool_calls:
                    # 新工具调用的开始
                    if tool_call.index is not None:
                        # 如果是新的工具调用，保存当前工具调用并创建新的
                        if current_tool_call is None or tool_call.index != current_tool_call[
                                "index"]:
                            if current_tool_call:
                                collected_tool_calls.append(current_tool_call)
                            current_tool_call = {
                                "id": tool_call.id or "",
                                "type": "function",
                                "index": tool_call.index,
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            }

                    # 更新工具名称（实际上只在第一次获取时设置）
                    if tool_call.function and tool_call.function.name:
                        current_tool_call["function"][
                            "name"] = tool_call.function.name
                    # 更新工具参数（需要拼接）
                    if tool_call.function and tool_call.function.arguments:
                        current_tool_call["function"][
                            "arguments"] += tool_call.function.arguments

        # 如果存在未完成的工具调用，将其添加到已收集的工具调用列表中
        if current_tool_call:
            collected_tool_calls.append(current_tool_call)

        # 首先搞一个assistant message
        assistant_message = {
            "role": "assistant",
            "content": "".join(collected_content).strip(),
            "tool_calls": collected_tool_calls
        }
        # 再调用工具
        tool_response = await self.session.call_tool(
            name=current_tool_call["function"]["name"],
            input=current_tool_call["function"]["arguments"])

        # 再搞一个tool message
        tool_message = {
            "role": "tool",
            "content": tool_response,
            "tool_call_id": tool_call.id
        }

        # 将assistant message和tool message添加到messages中
        messages.append(assistant_message)
        messages.append(tool_message)

        # 再调用大模型
        response = await self.llm.chat.completions.create(model="qwen-plus",
                                                          messages=messages,
                                                          stream=True)

        collected_content = []
        # 返回最终的response，流式输出
        async for chunk in response:
            if chunk.choices[0].delta.content:
                chunk_content = chunk.choices[0].delta.content
                collected_content.append(chunk_content)
                print(chunk_content, end="", flush=True)

        return "".join(collected_content).strip()
