from langgraph.graph.state import CompiledStateGraph, StateGraph, START, END
# from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Dict, Any, List, Union
from openai import AsyncOpenAI
from typing import Literal, Optional
from openai.types.chat import ChatCompletionMessage
# from graphagent.tool import baidu_search
import os
import asyncio

# def add_messages(left: List[Any], right: Union[Any, List[Any]]) -> List[Any]:
#     if isinstance(right, list):
#         return left + right
#     return left + [right]


class State(TypedDict):
    messages: List[Dict[str, Any]]


class ChatNode:

    def __init__(self,
                 *,
                 api_key: str,
                 base_url: str,
                 model: str = "qwen-plus",
                 tool_choice: Literal["auto", "required", "none"] = "auto",
                 tools: Optional[List[Dict]] = None,
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
        self.tools = tools

    async def __call__(self, state: State) -> State:
        try:
            # 构建请求参数，字典形式
            request_params = {
                "model": self.model,
                "messages": state["messages"],
                "tool_choice": self.tool_choice,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": self.stream,
            }

            if self.enable_thinking is not None:
                request_params["extra_body"] = {
                    "enable_thinking": self.enable_thinking
                }

            # 如果有工具,添加工具相关参数
            if self.tools:
                request_params["tools"] = self.tools

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
                state["messages"].append({
                    "role":
                    "assistant",
                    "content":
                    "".join(collected_content).strip()
                    if collected_content else "",
                    "tool_calls":
                    collected_tool_calls if collected_tool_calls else None
                })
                return state

        except Exception as e:
            raise Exception(f"调用大模型API失败: {str(e)}")


class MyAgent:

    def __init__(self):
        self.graph = StateGraph(State)
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def create_graph(self) -> "CompiledStateGraph":
        self.graph.add_node(
            "chat",
            ChatNode(api_key=self.api_key,
                     base_url=self.base_url,
                     model="qwen-plus-latest",
                     max_tokens=8000,
                     tool_choice="auto",
                     stream=True,
                     enable_thinking=False))
        self.graph.add_edge(START, "chat")

        return self.graph.compile()


async def stream_graph_updates(graph, user_input: str):
    user_message = {"role": "user", "content": user_input}
    async for chunk in graph.astream({"messages": [user_message]},
                                     stream_mode="updates"):
        # if chunk.event != "messages":
        #     print("跳过")
        #     continue
        print(chunk)
        # message_chunk, metadata = chunk.data
        # if message_chunk["content"]:
        #     print(message_chunk["content"], end="|", flush=True)
        # for value in event.values():
        #     print("Assistant:", value["messages"][-1].content)


async def main():
    graph = MyAgent().create_graph()

    while True:
        # try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        await stream_graph_updates(graph, user_input)
        # except:
        #     # fallback if input() is not available
        #     user_input = "What do you know about LangGraph?"
        #     print("User: " + user_input)
        #     await stream_graph_updates(graph, user_input)
        #     break


if __name__ == "__main__":
    asyncio.run(main())
