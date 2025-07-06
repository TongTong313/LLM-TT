from typing import List, Dict, Any, TypedDict, Optional
from graphagent.message.openai import OpenAIMessage
import os
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from graphagent.node.base import BaseNode
from openai import AsyncOpenAI
from graphagent.prompt.system_prompt import DEFAULT_SYSTEM_PROMPT_FOR_SUMMARY_NODE
from loguru import logger


class SummaryNodeState(TypedDict):
    """
    总结节点状态，包含总结消息列表
    """

    messages: List[OpenAIMessage | Dict[str, Any]]


class SummaryNodeConfig(BaseModel):
    """
    总结节点配置，包含总结消息列表，静态配置
    
    Args:
        api_key (str): 阿里云API密钥，默认从环境变量DASHSCOPE_API_KEY中获取
        base_url (str): 阿里云API地址，默认是阿里云API地址
        stream (bool): 是否流式输出，默认是False
        enable_thinking (Optional[bool]): 是否启用思考，针对Qwen3系列模型，默认是None
    """
    api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    stream: bool = False
    enable_thinking: Optional[bool] = None


class SummaryNodeRunnableConfig(TypedDict):
    """总结节点运行时配置，动态配置
    
    Args:
        summary_model (str): 大模型名称
        summary_system_prompt (str): 系统提示词
        summary_temperature (float): 温度
        summary_max_tokens (int): 最大tokens
    """
    summary_model: str
    summary_system_prompt: str
    summary_temperature: float
    summary_max_tokens: int


class SummaryNode(BaseNode):
    """
    总结节点，根据messages中所有的内容，给出总结性步骤
    """

    def __init__(self,
                 *,
                 api_key: str,
                 base_url: str,
                 stream: bool = False,
                 enable_thinking: Optional[bool] = None,
                 **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.stream = stream
        self.enable_thinking = enable_thinking
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        # 定死，一定不能调用工具
        self.tool_choice = "none"  # 一定不能调用工具

    async def __call__(self, state, config: RunnableConfig):

        logger.info(f"正在执行总结节点")

        # 比较简单，把所有信息总结一下就好了
        system_message = OpenAIMessage.system_message(
            content=config["configurable"].get(
                "summary_system_prompt",
                DEFAULT_SYSTEM_PROMPT_FOR_SUMMARY_NODE))
        user_message = OpenAIMessage.user_message(
            content="请根据之前的所有对话内容，结合用户的需求给出最后要提供给用户的内容")

        state["messages"].append(system_message)
        state["messages"].append(user_message)

        request_params = {
            "model": config["configurable"].get("summary_model", "qwen-plus"),
            "messages": state["messages"],
            "tool_choice": self.tool_choice,
            "temperature": config["configurable"].get("summary_temperature",
                                                      0.7),
            "max_tokens": config["configurable"].get("summary_max_tokens",
                                                     1024),
            "stream": self.stream
        }

        if self.enable_thinking is not None:
            request_params["extra_body"] = {
                "enable_thinking": self.enable_thinking
            }
        try:
            # 调用API
            if not request_params["stream"]:
                # 非流式请求
                response = await self.client.chat.completions.create(
                    **request_params)
                # 更新：把推理过程print出来但不保存
                if hasattr(response.choices[0].message, "reasoning_content"
                           ) and response.choices[0].message.reasoning_content:
                    print(
                        f"推理过程：{response.choices[0].message.reasoning_content}"
                    )

                assistant_message = OpenAIMessage.assistant_message(
                    content=response.choices[0].message.content,
                    tool_calls=None)

                state["messages"].append(assistant_message)

                return state
            else:
                # 流式请求
                response = await self.client.chat.completions.create(
                    **request_params)
                collected_content = []

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

                assistant_message = OpenAIMessage.assistant_message(
                    content="".join(collected_content).strip()
                    if collected_content else "",
                    tool_calls=None)

                state["messages"].append(assistant_message)

                return state

        except Exception as e:
            raise Exception(f"调用大模型API失败: {str(e)}")
