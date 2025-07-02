from typing import TypedDict, List, Dict, Any, Optional, Callable
from graphagent.message.openai import OpenAIMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from graphagent.node.base import BaseNode
from graphagent.node.tool import BaseTool, FunctionTool
import os
from openai import AsyncOpenAI


class RouterNodeState(TypedDict):
    """
    路由节点状态，包含路由消息列表
    """

    messages: List[OpenAIMessage | Dict[str, Any]]


class RouterNodeConfig(BaseModel):
    """
    路由节点配置，包含路由消息列表，静态配置
    
    Args:
        api_key (str): 阿里云API密钥，默认从环境变量DASHSCOPE_API_KEY中获取
        base_url (str): 阿里云API地址，默认是阿里云API地址
        tools (Optional[List[BaseTool | Callable]]): 工具列表，默认是None
        stream (bool): 是否流式输出，默认是False
        enable_thinking (Optional[bool]): 是否启用思考，针对Qwen3系列模型，默认是None
    """
    api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    tools: Optional[List[BaseTool | Callable]] = None
    stream: bool = False
    enable_thinking: Optional[bool] = None
    system_prompt: str = ""


class RouterNodeRunnableConfig(TypedDict):
    """路由节点运行时配置，动态配置
    
    Args:
        router_model (str): 大模型名称
        router_system_prompt (str): 系统提示词
        router_temperature (float): 温度
        router_max_tokens (int): 最大tokens
    """
    router_model: str
    router_system_prompt: str
    router_temperature: float
    router_max_tokens: int


# router要多一个路由函数，便于需要调用工具的时候将条件边路由到工具节点
def router_function(state) -> str:
    """路由函数，根据状态决定路由到哪个节点
    """
    # 检查是否有工具调用
    if state["messages"][-1].get("tool_calls"):
        return "tool"

    # 检查是否还有待执行的规划步骤
    plan = state.get("plan", [])
    for step in plan:
        if step["status"] == "pending":
            return "router"  # 还有待执行的步骤，继续路由

    # 检查是否刚执行完工具，需要继续分析
    if state["messages"][-1].get("role") == "tool":
        return "router"  # 工具执行完成，继续路由分析

    return "end"  # 所有步骤都完成了，结束


class RouterNode(BaseNode):
    """路由节点，根据PlanningNode的规划结果，给出工具调用方案，运行工具是ToolNode来做
    """

    def __init__(self,
                 *,
                 api_key: str,
                 base_url: str,
                 tools: Optional[List[BaseTool | Callable]] = None,
                 stream: bool = False,
                 enable_thinking: Optional[bool] = None,
                 **kwargs):
        # 定死
        self.tool_choice = "auto"  # 必须可以调工具

        self.api_key = api_key
        self.base_url = base_url
        self.stream = stream
        self.enable_thinking = enable_thinking
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        if tools:
            self.tool_schema = []
            for tool in tools:
                if isinstance(tool, (BaseTool, FunctionTool)):
                    self.tool_schema.append(tool.tool_schema)
                elif callable(tool):
                    self.tool_schema.append(
                        FunctionTool(tool=tool).tool_schema)
                elif isinstance(tool, dict):
                    if 'tool_schema' in tool:
                        self.tool_schema.append(tool['tool_schema'])
                    else:
                        raise ValueError(f"工具字典中没有tool_schema字段: {tool}")
                else:
                    raise ValueError(f"未知工具类型: {type(tool)}")
        else:
            self.tool_schema = None

    async def __call__(self, state, config: RunnableConfig):
        """路由节点运行，根据PlanningNode的规划结果，给出工具调用方案
        """
        try:
            # 按顺序取一步的规划，如果这一步的plan的status是pending，则需要调用大模型给出工具调用方案
            plan = state["plan"]
            for step in plan:
                # 已提取可规划
                if step["status"] == "pending":
                    plan_text = step["step"]
                    # 更新state中plan的该步骤状态为running
                    state["plan"][plan.index(step)]["status"] = "running"
                    break

            if plan_text:
                system_message = {
                    "role":
                    "system",
                    "content":
                    config["configurable"].get("router_system_prompt", "")
                }
                user_message = {"role": "user", "content": plan_text}
                state["messages"].append(system_message)
                state["messages"].append(user_message)
            else:
                # 没有可规划的步骤，直接结束
                return state

            # 构建请求参数，字典形式
            request_params = {
                "model":
                config["configurable"].get("router_model", "qwen-plus"),
                "messages":
                state["messages"],
                "tool_choice":
                self.tool_choice,
                "max_tokens":
                config["configurable"].get("router_max_tokens", 1000),
                "temperature":
                config["configurable"].get("router_temperature", 0.7),
                "stream":
                self.stream,
            }

            if self.enable_thinking is not None:
                request_params["extra_body"] = {
                    "enable_thinking": self.enable_thinking
                }

            # 如果有工具,添加工具相关参数
            if self.tool_schema:
                request_params["tools"] = self.tool_schema

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

                state["messages"].append(response.choices[0].message)
                return state
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

                assistant_message = OpenAIMessage.assistant_message(
                    content="".join(collected_content).strip()
                    if collected_content else "",
                    tool_calls=collected_tool_calls
                    if collected_tool_calls else None)

                state["messages"].append(assistant_message)

                return state

        except Exception as e:
            raise Exception(f"调用大模型API失败: {str(e)}")
