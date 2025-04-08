from openai import AsyncOpenAI
import os
from typing import List, Dict, Optional, Literal


class LLM:
    """大模型类，可用于agent中作为规划器使用，支持openai接口规范
    
    Args:
        api_key (str): 大模型api key。
        base_url (str): 大模型base url。
        tool_choice (str, optional): 工具选择模式，包括"auto", "required", "none"，默认"auto"。
        temperature (float, optional): 温度，控制大模型生成结果的随机性，越大随机性越强，默认0.7。
        max_tokens (int, optional): 最大tokens，默认1000。
    """

    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model: str = "qwen-plus",
                 tool_choice: Literal["auto", "required", "none"] = "auto",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_choice = tool_choice

    async def chat(self,
                   messages: List[Dict],
                   tools: Optional[List[Dict]] = None,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   stream: bool = False) -> Dict:
        """与大模型进行交互对话

        Args:
            messages (List[Dict]): 对话历史记录
            tools (List[Dict], optional): 可用的工具列表（function schema格式）。 Defaults to None.
            temperature (float, optional): 温度，控制大模型生成结果的随机性，越大随机性越强，默认使用类初始化时的温度。
            max_tokens (int, optional): 最大tokens，默认使用类初始化时的最大tokens。
            stream (bool, optional): 是否流式输出，默认False。

        Returns:
            Dict: 大模型的回复
        """
        try:
            # 构建请求参数，字典形式
            request_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "stream": stream
            }

            # 如果有工具,添加工具相关参数
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = self.tool_choice

            # 调用API
            if not stream:
                # 非流式请求
                response = await self.client.chat.completions.create(
                    **request_params)
                # 返回大模型回复的message，形式为：
                # {
                #     "role": "assistant",
                #     "content": "你好，我是小明，很高兴认识你。"
                # }
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

                    # 处理工具调用部分
                    if chunk.choices[0].delta.tool_calls:
                        for tool_call in chunk.choices[0].delta.tool_calls:
                            # 新工具调用的开始
                            if tool_call.index is not None and (
                                    not current_tool_call or tool_call.index
                                    != current_tool_call["index"]):
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

                            # 更新工具名称
                            if tool_call.function and tool_call.function.name:
                                current_tool_call["function"][
                                    "name"] = tool_call.function.name
                                print(f"[工具调用: {tool_call.function.name}]",
                                      end="",
                                      flush=True)

                            # 更新工具参数
                            if tool_call.function and tool_call.function.arguments:
                                current_tool_call["function"][
                                    "arguments"] += tool_call.function.arguments
                                # 可以选择是否打印参数
                                # print(tool_call.function.arguments, end="", flush=True)

                # 添加最后一个工具调用
                if current_tool_call:
                    collected_tool_calls.append(current_tool_call)

                print()  # 换行

                # 构建完整的响应消息
                full_message = {
                    "role":
                    "assistant",
                    "content":
                    "".join(collected_content).strip()
                    if collected_content else None
                }

                # 如果有工具调用，添加到响应中
                if collected_tool_calls:
                    full_message["tool_calls"] = collected_tool_calls

                return full_message

        except Exception as e:
            raise Exception(f"调用大模型API失败: {str(e)}")
