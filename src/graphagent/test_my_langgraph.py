# 测试自己编写的节点是否可langgraph运行
from langgraph.graph.state import StateGraph, START, END, CompiledStateGraph
from typing import TypedDict, Literal
from graphagent.node.planning import PlanningNode, PlanningNodeState, PlanningNodeConfig
from graphagent.node.tool import ToolNodeState
import os
from graphagent.prompt.system_prompt import DEFAULT_SYSTEM_PROMPT_FOR_PLANNING_NODE
import asyncio
from graphagent.message.openai import OpenAIMessage
from langchain_core.runnables.config import RunnableConfig


# 1. 预制节点配套预制状态(TypedDict)，用户可以直接使用，也支持自定义
class State(PlanningNodeState):
    pass


print(State.__annotations__)


# 2. 用户可以拿到预制配置(Pydantic模型)，也可以自己设定，甚至修改参数
# TypedDict不支持默认值语法
class ConfigSchema(PlanningNodeConfig):
    pass


# 3. 用户拿到预制节点形成智能体，或直接使用智能体模板
class MyAgent:

    def __init__(self, config: PlanningNodeConfig):
        self.config = config
        # 配置静态参数
        self.planning_node = PlanningNode(**config.model_dump())

    def create_graph(self) -> CompiledStateGraph:
        self.graph = StateGraph(State, config_schema=self.config)
        self.graph.add_node("planning", self.planning_node)
        self.graph.add_edge(START, "planning")
        self.graph.add_edge("planning", END)
        return self.graph.compile()


if __name__ == "__main__":
    # 创建配置实例 - 这是关键修复
    config = ConfigSchema()

    my_agent = MyAgent(config)
    graph = my_agent.create_graph()

    runnable_config = {'configurable': {'max_tokens': 1000}}

    async def main():
        events = graph.astream(
            {
                "messages":
                [OpenAIMessage.user_message(content="你好，请帮我规划一个去北京旅游的行程")]
            },
            stream_mode="values",
            config=runnable_config)
        async for event in events:
            if "messages" in event:
                pass
                # print(event["planning_messages"][-1]["content"])

        # async for chunk in graph.astream(
        #     {
        #         "planning_messages":
        #         [OpenAIMessage.user_message(content="你好，请帮我规划一个去北京旅游的行程")]
        #     },
        #         stream_mode="updates",
        #         config=runnable_config):

        #     for event in chunk:
        #         print(event)
        #         if "planning_messages" in event:
        #             print(event["planning_messages"][-1]["content"])

    asyncio.run(main())
