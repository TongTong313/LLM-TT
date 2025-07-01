from typing import Literal, Optional, List, Dict, Any, TypedDict, Callable
from graphagent.message.openai import OpenAIMessage
from graphagent.prompt.system_prompt import DEFAULT_SYSTEM_PROMPT_FOR_PLANNING_NODE
from openai import AsyncOpenAI
from langchain_core.runnables.config import RunnableConfig
from graphagent.node.base import BaseNode
from pydantic import BaseModel
import os
from graphagent.node.tool import FunctionTool, BaseTool


class PlanningNodeState(TypedDict):
    """
    规划节点状态，包含规划消息列表
    
    Args:
        messages (List[OpenAIMessage | Dict[str, Any]]): 规划消息列表
        plan (List[Dict[str, Any]]): 规划列表，每个元素是字典，字典有两个键值对，分别是步骤和该步骤的状态，状态有：
            - "pending": 未开始
            - "running": 正在运行
            - "completed": 已完成
            - "failed": 失败
    """

    messages: List[OpenAIMessage | Dict[str, Any]]
    plan: List[Dict[str, Any]]


class PlanningNodeConfig(BaseModel):
    """预制Planning节点配置，pydantic模型
    
    Args:
        api_key (str): 大模型api key，如果是本地部署的大模型例如vllm，需要设置api_key为"EMPTY"，属于静态配置。
        base_url (str): 大模型base url，属于静态配置。
        tools (List[BaseTool | Callable], optional): 工具列表，属于静态配置。
        stream (bool, optional): 是否流式输出，默认False，属于静态配置。
        enable_thinking (bool, optional): 是否启用大模型思考模式，仅在使用qwen3模型时有效，默认是None，表示该大模型不具备思考模式切换能力，属于静态配置。
        step_start_token (str, optional): 规划的步骤开始标记，默认是"<step>"，属于静态配置。
        step_end_token (str, optional): 规划的步骤结束标记，默认是"</step>"，属于静态配置。
    """

    api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    tools: Optional[List[BaseTool | Callable]] = None
    stream: bool = False
    enable_thinking: Optional[bool] = None
    step_start_token: str = "<step>"
    step_end_token: str = "</step>"


class PlanningNodeRunnableConfig(TypedDict):
    """
    规划节点运行时配置，pydantic模型，支持动态修改，每次运行必须设置，无默认值！
    Args:
        planning_model (str, optional): 大模型名称，推荐"qwen-plus"，属于动态配置。
        planning_system_prompt (str, optional): 系统提示词，属于动态配置。
        planning_temperature (float, optional): 温度，控制大模型生成结果的随机性，越大随机性越强，属于动态配置，推荐0.7
        planning_max_tokens (int, optional): 最大tokens，属于动态配置，推荐1000。
    """

    planning_model: str
    planning_system_prompt: str
    planning_temperature: float
    planning_max_tokens: int


class PlanningNode(BaseNode):
    """
    使用大模型进行任务规划的节点
    
    Args:
        api_key (str): 大模型api key，如果是本地部署的大模型例如vllm，需要设置api_key为"EMPTY"。
        base_url (str): 大模型base url。
        tools (List[BaseTool | Callable], optional): 工具列表，属于静态配置。
        stream (bool, optional): 是否流式输出，默认False，属于静态配置。
        enable_thinking (bool, optional): 是否启用大模型思考模式，仅在使用qwen3模型时有效，默认是None，表示该大模型不具备思考模式切换能力，属于静态配置。
        step_start_token (str, optional): 规划的步骤开始标记，默认是"<step>"，属于静态配置。
        step_end_token (str, optional): 规划的步骤结束标记，默认是"</step>"，属于静态配置。
    """

    def __init__(self,
                 *,
                 api_key: str,
                 base_url: str,
                 tools: Optional[List[BaseTool | Callable]] = None,
                 stream: bool = False,
                 enable_thinking: Optional[bool] = None,
                 step_start_token: str = "<step>",
                 step_end_token: str = "</step>",
                 **kwargs):

        self.api_key = api_key
        self.base_url = base_url
        self.step_start_token = step_start_token
        self.step_end_token = step_end_token
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

        self.tool_choice = 'auto'  # 虽然不调用，但仍然需要设置为auto才能被大模型感知到有工具的存在
        self.stream = stream
        self.enable_thinking = enable_thinking

        # 初始化大模型client
        self.llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def run(self, *, state, config: RunnableConfig):
        """
        调用大模型产生规划结果，支持流式和非流式

        Args:
            state (PlanningNodeState): 规划节点的状态
                具体包含：
                    - planning_messages: 规划消息列表
            config (RunnableConfig): 运行时配置，可以在每次调用图的时候修改一些参数，也可以在langsmith前端进行修改
                具体包含：
                    - model: 大模型名称，如果配置了，则使用配置的模型，否则使用节点初始化时配置的模型
                    - temperature: 温度，如果配置了，则使用配置的温度，否则使用节点初始化时配置的温度
                    - max_tokens: 最大tokens，如果配置了，则使用配置的最大tokens，否则使用节点初始化时配置的最大tokens
                    - system_prompt: 系统提示词

        Returns:
            PlanningNodeState: 规划节点的状态，包含规划消息列表
        """
        # 这里是不需要考虑工具调用的，只用来做规划，但可能会含有工具，根据工具信息生成规划
        try:
            system_message = {
                "role": "system",
                "content": config["configurable"].get("planning_system_prompt")
            }
            messages = [system_message] + state["messages"]
            # 增加工具，通过工具分析流程
            request_params = {
                "model": config["configurable"].get("planning_model"),
                "messages": messages,
                "temperature":
                config["configurable"].get("planning_temperature"),
                "max_tokens":
                config["configurable"].get("planning_max_tokens"),
                "stream": self.stream,
                "tool_choice": self.tool_choice,
                "tools": self.tool_schema,
            }
            if self.enable_thinking is not None:
                request_params["extra_body"] = {
                    "enable_thinking": self.enable_thinking
                }

            # 分成流式和非流式两种情况，暂时不考虑工具调用
            if not self.stream:  # 非流式请求
                response = await self.llm_client.chat.completions.create(
                    **request_params)
                return response.choices[0].message
            else:  # 流式请求
                response = await self.llm_client.chat.completions.create(
                    **request_params)
                collected_content = []
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        chunk_content = chunk.choices[0].delta.content
                        collected_content.append(chunk_content)
                        print(chunk_content, end="", flush=True)

                # 把规划文本单独抽出来
                plan_text = "".join(collected_content).strip()

                # 把消息加进去
                state["messages"].append(
                    OpenAIMessage.assistant_message(
                        content="".join(collected_content).strip(
                        ) if collected_content else ""))

                # 把规划按照step_start_token和step_end_token拆分成步骤
                # state["plan"]是列表，列表的每个元素是字典，字典有两个键值对，分别是步骤和该步骤的状态，状态有：
                # - "pending": 未开始
                # - "running": 正在运行
                # - "completed": 已完成
                # - "failed": 失败
                # - "skipped": 跳过
                # - "cancelled": 取消
                plan_list = []
                for step in plan_text.split(self.step_start_token):
                    if step.strip():
                        plan_list.append({
                            "step":
                            step.split(self.step_end_token)[0].strip(),
                            "status":
                            "pending"  # 默认未开始
                        })

                state["plan"] = plan_list

                return state

        except Exception as e:
            raise Exception(f"调用大模型API失败: {str(e)}")

    async def __call__(self, state, config: RunnableConfig):
        return await self.run(state=state, config=config)
