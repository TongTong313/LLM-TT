# 测试自己编写的节点是否可langgraph运行

from langgraph.graph.state import StateGraph, START, END, CompiledStateGraph
from typing import TypedDict, List, Dict, Any, Literal
from graphagent.node.planning import PlanningNode
import os
from graphagent.prompt.system_prompt import DEFAULT_SYSTEM_PROMPT_FOR_PLANNING_NODE
import asyncio
from graphagent.message.openai import UserMessage
from langchain_core.runnables.config import RunnableConfig


class State(TypedDict):
    messages: List[Dict[str, Any]]


class ConfigSchema(TypedDict):
    api_key: str = os.getenv("DASHSCOPE_API_KEY")
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus-latest"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT_FOR_PLANNING_NODE
    tool_choice: Literal["auto", "required", "none"] = "auto"
    temperature: float = 0.7
    max_tokens: int = 8000
    stream: bool = True
    enable_thinking: bool = False


# 复写节点逻辑，只需要写__call__方法和类内方法实现场景state的修改
class MyPlanningNode(PlanningNode):

    async def __call__(self, state: State, config: RunnableConfig) -> State:
        plan = await self._chat_with_llm(messages=state["messages"],
                                         config=config)
        state["messages"].append(plan)
        return state


class MyAgent:

    def __init__(self, config: RunnableConfig):
        self.config = config
        self.planning_node = MyPlanningNode(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            model=self.config.model,
            system_prompt=self.config.system_prompt,
            tool_choice=self.config.tool_choice,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=self.config.stream,
            enable_thinking=self.config.enable_thinking)

    def create_graph(self) -> CompiledStateGraph:
        self.graph = StateGraph(State, config_schema=self.config)
        self.graph.add_node("planning", self.planning_node)
        self.graph.add_edge(START, "planning")
        self.graph.add_edge("planning", END)
        return self.graph.compile()


if __name__ == "__main__":
    config = ConfigSchema
    my_agent = MyAgent(config)
    graph = my_agent.create_graph()

    runnable_config = {'configurable': {'max_tokens': 1000}}

    async def main():
        async for chunk in graph.astream(
            {"messages": [UserMessage(content="你好，请帮我规划一个去北京旅游的行程")]},
                stream_mode="updates",
                config=runnable_config):
            print(chunk)

    asyncio.run(main())
