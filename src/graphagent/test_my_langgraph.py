# 测试自己编写的节点是否可langgraph运行
from langgraph.graph.state import StateGraph, START, END, CompiledStateGraph
from typing import TypedDict, Literal, List, Callable
from graphagent.node.planning import PlanningNode, PlanningNodeState, PlanningNodeConfig, PlanningNodeRunnableConfig
from graphagent.node.tool import ToolNode, ToolNodeState, ToolNodeConfig, FunctionTool
import os
from graphagent.prompt.system_prompt import DEFAULT_SYSTEM_PROMPT_FOR_PLANNING_NODE, DEFAULT_SYSTEM_PROMPT_FOR_ROUTER_NODE
import asyncio
from graphagent.message.openai import OpenAIMessage
from langchain_core.runnables.config import RunnableConfig
from graphagent.tool import add, baidu_search, get_current_time
from graphagent.node.tool import BaseTool
from graphagent.node.router import RouterNodeState, RouterNodeConfig, RouterNodeRunnableConfig
from graphagent.node.router import RouterNode


# 1. 预制节点配套预制状态(TypedDict)，用户可以直接使用，也支持自定义
class State(PlanningNodeState, ToolNodeState, RouterNodeState):
    pass


# 2. 用户可以拿到静态预制配置(Pydantic模型)，也可以自己设定，甚至修改参数
# TypedDict不支持默认值语法
class ConfigSchema(PlanningNodeConfig, ToolNodeConfig, RouterNodeConfig):
    stream: bool = True
    tools: List[BaseTool | Callable] = [
        FunctionTool(tool=add),
        FunctionTool(tool=baidu_search),
        FunctionTool(tool=get_current_time)
    ]


# 3. 用户可以拿到预制运行时配置(TypedDict模型)，也可以自己设定，甚至修改参数
class RunnableConfigSchema(PlanningNodeRunnableConfig):
    pass


# 4. 用户拿到预制节点形成智能体，或直接使用智能体模板
class MyAgent:

    def __init__(self, config: ConfigSchema, runnable_config: RunnableConfig):
        self.config = config
        self.runnable_config = runnable_config
        # 配置静态参数
        self.planning_node = PlanningNode(
            api_key=config.api_key,
            base_url=config.base_url,
            tools=config.tools,
            stream=config.stream,
            enable_thinking=config.enable_thinking,
            step_start_token=config.step_start_token,
            step_end_token=config.step_end_token)
        self.tool_node = ToolNode(**config.model_dump())
        self.router_node = RouterNode(**config.model_dump())

    def create_graph(self) -> CompiledStateGraph:
        self.graph = StateGraph(State, config_schema=self.runnable_config)
        self.graph.add_node("planning", self.planning_node)
        self.graph.add_node("tool", self.tool_node)
        self.graph.add_node("router", self.router_node)
        self.graph.add_edge(START, "planning")
        self.graph.add_edge("planning", "router")
        self.graph.add_edge("router", "tool")
        self.graph.add_edge("tool", END)

        return self.graph.compile()


if __name__ == "__main__":
    # 创建配置实例 - 这是关键修复
    config = ConfigSchema()
    runnable_config = RunnableConfigSchema()

    my_agent = MyAgent(config, runnable_config)
    graph = my_agent.create_graph()

    runnable_config = {
        'configurable': {
            'planning_max_tokens': 8000,
            'planning_model': 'qwen-plus',
            'planning_system_prompt': DEFAULT_SYSTEM_PROMPT_FOR_PLANNING_NODE,
            'planning_temperature': 0.7,
            'router_max_tokens': 8000,
            'router_model': 'qwen-plus',
            'router_system_prompt': DEFAULT_SYSTEM_PROMPT_FOR_ROUTER_NODE,
            'router_temperature': 0.7
        }
    }

    async def main():
        events = graph.astream(
            {
                "messages":
                [OpenAIMessage.user_message(content="你好，请帮我生成一份扩散模型的综述报告")]
            },
            stream_mode="updates",
            config=runnable_config)
        async for event in events:
            print(event)
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
