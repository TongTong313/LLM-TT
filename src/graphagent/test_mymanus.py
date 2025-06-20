# 将我实现的manus拆解为langgraph实现方法
from mymanus.prompt import SYSTEM_PROMPT as system_prompt
from mymanus.agent import ToolCallingAgent, ToolManager, MemoryManager, LLM
from mymanus.tool import *
import os
import traceback
from loguru import logger
from langgraph.graph.state import CompiledStateGraph, StateGraph, START, END
from typing import TypedDict, Annotated, Dict, Any, List, Union, Literal
from openai.types.chat import ChatCompletionMessage
from langchain_core.runnables import RunnableConfig
import asyncio
from IPython.display import Image, display

MAX_STEP = 5


def add_messages(left: List[Any], right: Union[Any, List[Any]]) -> List[Any]:
    if isinstance(right, list):
        return left + right
    return left + [right]


class State(TypedDict):
    messages: Annotated[List[ChatCompletionMessage | Dict[str, str]],
                        add_messages]


class AgentConfig(TypedDict):
    max_step: int = MAX_STEP
    api_key: str = os.getenv("DASHSCOPE_API_KEY")
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus-latest"
    max_tokens: int = 8000
    tool_choice: Literal["auto", "required", "none"] = "auto"
    stream: bool = True
    enable_thinking: bool = False


class AgentNode:

    def __init__(self, config: AgentConfig):
        llm = LLM(api_key=config.api_key,
                  base_url=config.base_url,
                  model=config.model,
                  max_tokens=config.max_tokens,
                  tool_choice=config.tool_choice,
                  stream=config.stream,
                  enable_thinking=config.enable_thinking)
        self.llm = llm
        self.tool_manager = ToolManager()
        self.memory_manager = MemoryManager(max_memory=20)
        self.agent = ToolCallingAgent(llm=llm,
                                      tool_manager=self.tool_manager,
                                      memory_manager=self.memory_manager,
                                      max_step=config.max_step)
        # 注册工具
        self.agent.add_tool(baidu_search, tool_name="baidu_search")
        self.agent.add_tool(get_current_time, tool_name="get_current_time")
        self.agent.add_tool(terminate, tool_name="terminate")
        self.agent.add_tool(add, tool_name="add")

    async def __call__(self, state: State) -> State:
        try:
            result = await self.agent.run(state["messages"])
            return {"messages": result}
        except Exception as e:
            logger.error(f"智能体运行错误: {e}")
            logger.error("错误堆栈信息:")
            logger.error(traceback.format_exc())
            raise e


class MyAgent:

    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent = AgentNode(config)
        self.graph = StateGraph(State, config_schema=self.config)

    def create_graph(self) -> "CompiledStateGraph":
        self.graph.add_node("agent", self.agent)
        self.graph.add_edge(START, "agent")
        self.graph.add_edge("agent", END)
        return self.graph.compile()


async def main():
    graph = MyAgent(config=AgentConfig).create_graph()
    try:
        display(
            Image(graph.get_graph().draw_mermaid_png(
                output_file_path="graph.png")))
    except Exception:
        pass
    while True:
        try:
            prompt_list = [{"role": "system", "content": system_prompt}]
            prompt = input(
                "我是TT开发的manus超级助手，请输入你的需求，我会尽力解决你的问题，输入quit/exit可退出：")
            if prompt.lower() in ["quit", "exit"]:
                logger.warning("再见!")
                break

            # 要把prompt变为字典送入
            prompt_dict = [{"role": "user", "content": prompt}]
            prompt_list.extend(prompt_dict)

            # 运行智能体
            logger.warning(f"智能体正在运行中……")
            result = await graph.ainvoke({"messages": prompt_list})

            if result:
                logger.warning(f"智能体执行完成")
        except KeyboardInterrupt:
            logger.warning("再见!")
            break
        except Exception as e:
            logger.error(f"智能体运行错误: {e}")
            logger.error("错误堆栈信息:")
            logger.error(traceback.format_exc())
            break


if __name__ == "__main__":
    asyncio.run(main())
