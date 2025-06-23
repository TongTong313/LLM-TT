from typing import Literal, Optional, List, Dict, Any, TypedDict
from graphagent.message.openai import OpenAIMessage
from graphagent.prompt.system_prompt import DEFAULT_SYSTEM_PROMPT_FOR_PLANNING_NODE
from openai import AsyncOpenAI
from langchain_core.runnables.config import RunnableConfig
from graphagent.node.base import BaseNode
from pydantic import BaseModel
import os


class PlanningNodeState(TypedDict):
    """
    规划节点状态，包含规划消息列表
    """

    messages: List[OpenAIMessage | Dict[str, Any]]


class PlanningNodeConfig(BaseModel):
    """预制Planning节点配置，pydantic模型
    
    Args:
        api_key (str): 大模型api key，如果是本地部署的大模型例如vllm，需要设置api_key为"EMPTY"，属于静态配置。
        base_url (str): 大模型base url，属于静态配置。
        model (str, optional): 大模型名称，默认"qwen-plus"，属于动态配置。
        system_prompt (str, optional): 系统提示词，默认""，属于静态配置。
        tool_choice (str, optional): 工具选择模式，包括"auto", "required", "none"，默认"auto"，属于静态配置。
        temperature (float, optional): 温度，控制大模型生成结果的随机性，越大随机性越强，默认0.7，属于动态配置。
        max_tokens (int, optional): 最大tokens，默认1000，属于动态配置。
        stream (bool, optional): 是否流式输出，默认False，属于静态配置。
        enable_thinking (bool, optional): 是否启用大模型思考模式，仅在使用qwen3模型时有效，默认是None，表示该大模型不具备思考模式切换能力，属于静态配置。
    """

    api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus-latest"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT_FOR_PLANNING_NODE
    tool_choice: Literal["auto", "required", "none"] = "auto"
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = True
    enable_thinking: bool = False


class PlanningNode(BaseNode):
    """
    使用大模型进行任务规划的节点
    
    Args:
        api_key (str): 大模型api key，如果是本地部署的大模型例如vllm，需要设置api_key为"EMPTY"。
        base_url (str): 大模型base url。
        model (str, optional): 大模型名称，默认"qwen-plus"。
        tool_choice (str, optional): 工具选择模式，包括"auto", "required", "none"，默认"auto"。
        system_prompt (str, optional): 系统提示词，默认""。
        temperature (float, optional): 温度，控制大模型生成结果的随机性，越大随机性越强，默认0.7。
        max_tokens (int, optional): 最大tokens，默认1000。
        stream (bool, optional): 是否流式输出，默认False。
        enable_thinking (bool, optional): 是否启用大模型思考模式，仅在使用qwen3模型时有效，默认是None，表示该大模型不具备思考模式切换能力。
    """

    def __init__(self,
                 *,
                 api_key: str,
                 base_url: str,
                 model: str = "qwen-plus",
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT_FOR_PLANNING_NODE,
                 tool_choice: Literal["auto", "required", "none"] = "auto",
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 stream: bool = False,
                 enable_thinking: Optional[bool] = None):

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt
        self.tool_choice = tool_choice
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.enable_thinking = enable_thinking

        # 初始化大模型client
        self.llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def run(self, *, state: PlanningNodeState,
                  config: RunnableConfig) -> PlanningNodeState:
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

        Returns:
            PlanningNodeState: 规划节点的状态，包含规划消息列表
        """
        # 这里是不需要考虑工具调用的，只用来做规划，但可能会含有工具（后面加）
        try:
            system_message = {"role": "system", "content": self.system_prompt}
            messages = [system_message] + state["messages"]
            # 暂时不加工具，推理过程暂时不支持
            request_params = {
                "model":
                self.model if not config["configurable"].get("model") else
                config["configurable"].get("model"),
                "messages":
                messages,
                "temperature":
                self.temperature
                if not config["configurable"].get("temperature") else
                config["configurable"].get("temperature"),
                "max_tokens":
                self.max_tokens if not config["configurable"].get("max_tokens")
                else config["configurable"].get("max_tokens"),
                "stream":
                self.stream
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

                state["messages"].append(
                    OpenAIMessage.assistant_message(
                        content="".join(collected_content).strip(
                        ) if collected_content else ""))
                return state

        except Exception as e:
            raise Exception(f"调用大模型API失败: {str(e)}")

    async def __call__(self, state: PlanningNodeState,
                       config: RunnableConfig) -> PlanningNodeState:
        return await self.run(state=state, config=config)
