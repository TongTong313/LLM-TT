from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
import os
from typing import List, Dict, Optional, Literal
import asyncio


class LLM:
    """大模型类，可用于agent中作为大脑用来规划和执行任务，基于openai接口规范
    
    Args:
        api_key (str): 大模型api key，如果是本地部署的大模型例如vllm，需要设置api_key为"EMPTY"。
        base_url (str): 大模型base url。
        model (str, optional): 大模型名称，默认"qwen-plus"。
        tool_choice (str, optional): 工具选择模式，包括"auto", "required", "none"，默认"auto"。
        temperature (float, optional): 温度，控制大模型生成结果的随机性，越大随机性越强，默认0.7。
        max_tokens (int, optional): 最大tokens，默认1000。
        stream (bool, optional): 是否流式输出，默认False。
        enable_thinking (bool, optional): 是否启用大模型思考模式，仅在使用qwen3模型时有效，默认是None，表示该大模型不具备思考模式切换能力。
    """

    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model: str = "qwen-plus",
                 tool_choice: Literal["auto", "required", "none"] = "auto",
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 stream: bool = False,
                 enable_thinking: Optional[bool] = None):

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_choice = tool_choice
        self.stream = stream
        self.enable_thinking = enable_thinking

    async def chat(self,
                   messages: List[Dict],
                   tools: Optional[List[Dict]] = None,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   tool_choice: Optional[Literal["auto", "required",
                                                 "none"]] = None,
                   stream: Optional[bool] = None,
                   enable_thinking: Optional[bool] = None) -> Dict:
        """与大模型进行交互对话

        Args:
            messages (List[Dict]): 对话历史记录
            tools (List[Dict], optional): 可用的工具列表（function schema格式）。 Defaults to None.
            temperature (float, optional): 温度，控制大模型生成结果的随机性，越大随机性越强，默认使用类初始化时的温度。
            max_tokens (int, optional): 最大tokens，默认使用类初始化时的最大tokens。
            tool_choice (str, optional): 工具选择模式，包括"auto", "required", "none"，默认使用类初始化时的工具选择模式。
            stream (bool, optional): 是否流式输出，默认使用类初始化时的流式输出。
            enable_thinking (bool, optional): 是否启用大模型思考模式，仅在使用qwen3模型时有效，默认使用类初始化时的思考模式。

        Returns:
            ChatCompletionMessage: openai格式的回复类
        """
        try:
            # 构建请求参数，字典形式
            request_params = {
                "model": self.model,
                "messages": messages,
                "tool_choice":
                self.tool_choice if tool_choice is None else tool_choice,
                "max_tokens":
                self.max_tokens if max_tokens is None else max_tokens,
                "temperature":
                self.temperature if temperature is None else temperature,
                "stream": self.stream if stream is None else stream,
            }

            if enable_thinking is not None:
                request_params["extra_body"] = {
                    "enable_thinking": enable_thinking
                }
            elif self.enable_thinking is not None:
                request_params["extra_body"] = {
                    "enable_thinking": self.enable_thinking
                }

            # 如果有工具,添加工具相关参数
            if tools:
                request_params["tools"] = tools

            # 调用API
            if not request_params["stream"]:
                # 非流式请求
                response = await self.client.chat.completions.create(
                    **request_params)
                # 更新：把推理过程print出来但不保存
                print(f"推理过程：{response.choices[0].message.reasoning_content}")
                return response.choices[0].message
            else:
                # 流式请求
                response = await self.client.chat.completions.create(
                    **request_params)
                collected_content = []
                collected_tool_calls = []
                current_tool_call = None

                async for chunk in response:
                    # 处理内容部分
                    if chunk.choices[0].delta.content:
                        chunk_content = chunk.choices[0].delta.content
                        collected_content.append(chunk_content)
                        print(chunk_content, end="", flush=True)
                    # 更新：处理推理内容部分，但最后不作为上下文返回给人看或提供给大模型，需要确认reasoning_content字段是否存在，不是每个大模型都有这个字段
                    if hasattr(chunk.choices[0].delta, "reasoning_content"):
                        if chunk.choices[0].delta.reasoning_content:
                            reasoning_content = chunk.choices[
                                0].delta.reasoning_content
                            print(reasoning_content, end="", flush=True)
                    # if chunk.choices[0].delta.reasoning_content:
                    #     reasoning_content = chunk.choices[
                    #         0].delta.reasoning_content
                    #     print(reasoning_content, end="", flush=True)

                    # 处理工具调用部分：工具调用部分第一个返回的流式输出对象可以获得工具名称（name），但工具的入参需要拼接
                    if chunk.choices[0].delta.tool_calls:
                        for tool_call in chunk.choices[0].delta.tool_calls:
                            # 新工具调用的开始
                            if tool_call.index is not None:
                                # 如果是新的工具调用，保存当前工具调用并创建新的
                                if current_tool_call is None or tool_call.index != current_tool_call[
                                        "index"]:
                                    if current_tool_call:
                                        collected_tool_calls.append(
                                            current_tool_call)
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

                # 添加最后一个工具调用
                if current_tool_call:
                    collected_tool_calls.append(current_tool_call)

                # 把字典转换为openai的ChatCompletionMessage对象
                return ChatCompletionMessage(
                    role="assistant",
                    content="".join(collected_content).strip()
                    if collected_content else "",
                    tool_calls=collected_tool_calls
                    if collected_tool_calls else None)

        except Exception as e:
            raise Exception(f"调用大模型API失败: {str(e)}")


if __name__ == "__main__":
    # 测试代码
    llm = LLM(api_key=os.getenv("DASHSCOPE_API_KEY"),
              base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
              stream=False)
    messages = []
    # 添加系统prompt
    messages.append({
        "role":
        "system",
        "content":
        "你是一个AI助手，请根据用户的问题给出回答，必要时可以调用多个工具完成任务。你现在拥有两个工具，一个获取用户信息，一个获取用户朋友圈。"
    })
    # 添加用户消息
    messages.append({"role": "user", "content": "你好，我的名字是童发发。"})
    # 模拟多个tools
    tools = [{
        "type": "function",
        "function": {
            "name": "get_user_info",
            "description": "获取用户信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    }
                }
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "get_user_friends",
            "description": "获取用户朋友圈",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    }
                }
            }
        }
    }]

    print(asyncio.run(llm.chat(messages, tools)))
